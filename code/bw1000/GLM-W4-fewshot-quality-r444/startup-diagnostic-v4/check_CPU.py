"""New startup glue only. Real CLI/Run/helper; Native/process leaves recorded."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch

S=Path(__file__).resolve().parent
R=Path('/refactor')
B=R/'candidates/GLM444-startup-diagnostic-v4-CPU'
D=S/'deployment';P=S/'provider'
sys.path.insert(0,str(D));sys.path.insert(0,str(P))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def configuration(folder,name):
    q=json.loads((D/'config-slot0.json').read_bytes())
    q.update(stage=1,run=name,runtime_directory=str(folder),actual_step='CPU.recording',
             source_manifest_SHA256=sha(P/'SOURCE_MANIFEST.json'),
             provider_config_SHA256=sha(D/'config-slot0.json'),gpu_start_epoch=1.)
    return q


def publish_ready(run):
    for stage in range(2):
        for rank in range(8):
            run.write('ready-%d-%d.json'%(stage,rank),dict(stage=stage,rank=rank,
                Native_SHA256=run.config['Native_SHA256'],config_SHA256=run.config['config_SHA256'],
                loaded=True,postload_4GiB=True,service_agreement=True,
                affinity={'actual_owner_report':{'owners':[dict(device=rank,changes=1,
                    verified_mask=','.join(map(str,range(rank*4,rank*4+4))))]}}))


def watchdog_child():
    from public_glm_native.startup_diagnostic import arm_after_owner_ready,resolve
    cfg=configuration(B/'watchdog','real-C-watchdog')
    folder=Path(cfg['runtime_directory'])/cfg['run'];folder.mkdir(parents=True)
    end=threading.Event()
    def emitter_wait():end.wait(48)
    def monitor_wait():end.wait(48)
    threads=[threading.Thread(target=emitter_wait,name='recorded-emitter'),
             threading.Thread(target=monitor_wait,name='recorded-monitor')]
    for t in threads:t.start()
    arm_after_owner_ready(cfg,0)
    try:time.sleep(42)
    finally:
        resolve(cfg,'CPU.actual_watchdog_cancel');end.set()
        for t in threads:t.join(2)
    files=list(folder.glob('startup-stack-*.txt'));assert len(files)==1
    raw=files[0].read_bytes();assert raw.count(b'Timeout (0:00:20)!')>=2
    assert b'emitter_wait'in raw and b'monitor_wait'in raw and b'watchdog_child'in raw
    assert len(raw)<1024**2
    print(json.dumps(dict(passed=True,actual_C_watchdog=True,repeated=True,
        cancelled=True,trace=str(files[0]),bytes=len(raw),SHA256=sha(files[0]),
        Native_stack=False,Python_locals=False)))


def main():
    B.mkdir(parents=True,exist_ok=True)
    result=B/'CPU-result.json';assert not result.exists()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        GPU=0,Model=False,Native_dlopen=False,SDK_executed=False,weights=0,
        Native_compile=0,old_CPU918_922_926_repeated=False,cases=[],errors=[],
        source_manifest_SHA256=sha(S/'SOURCE_MANIFEST.json'),
        env={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']},
        recorded_leaves=['Slurm read-only queries in launch_once','actor Process/owner body and affinity reports',
                         'owned cleanup process and signal leaves'],
        proc_maps_before=Path('/proc/self/maps').read_text())
    def save():
        tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(result)
    save()
    try:
        assert all(v=='-1'for v in q['env'].values())
        for p in S.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
        root=json.loads((S/'SOURCE_MANIFEST.json').read_bytes())
        for n,h in root['files_SHA256'].items():assert sha(S/n)==h,n
        import controller as C
        C.configure_slot(0);m=C.verify(sha(D/'SOURCE_MANIFEST.json'))
        # Real __main__ subprocess and source verifier; no mocked bind_identity.
        argv=[sys.executable,'-B',str(D/'controller.py'),'--manifest-sha',sha(D/'SOURCE_MANIFEST.json'),
              '--publication-commit','0'*40,'--slot','0','--preflight-only']
        z=subprocess.run(argv,capture_output=True,text=True,timeout=100)
        write(B/'actual-controller-main.json',dict(argv=argv,rc=z.returncode,stdout=z.stdout,stderr=z.stderr))
        assert z.returncode==0,z.stderr
        proof=json.loads(z.stdout);assert proof['passed']and proof['GPU']==proof['Slurm']==0
        q['cases'].append(dict(name='actual_controller___main__',passed=True,proof=proof));save()
        # Full launcher __main__ -> actual controller subprocess -> Popen-before
        # return. Only its external read-only scheduler replies are recorded.
        scheduler=[];original_check_output=subprocess.check_output
        def query(argv,**kwargs):
            scheduler.append(argv)
            if argv[0]=='sacct':return ''
            if argv[:3]==['scontrol','show','job']:
                nodes=','.join(C.NODES)if str(argv[3])==str(C.PARENT)else'a14r2n20'
                return 'JobState=RUNNING UserId=aclsylqidf(1000) NodeList='+nodes
            if argv[:3]==['scontrol','show','hostnames']:return '\n'.join(argv[3].split(','))+'\n'
            raise AssertionError(('Unexpected scheduler query',argv))
        launch_argv=[str(D/'launch_once.py'),'--manifest-sha',sha(D/'SOURCE_MANIFEST.json'),
                     '--publication-commit','0'*40,'--slot','0','--preflight-only']
        output=io.StringIO()
        with patch.object(subprocess,'check_output',query),patch.object(sys,'argv',launch_argv),contextlib.redirect_stdout(output):
            runpy.run_path(str(D/'launch_once.py'),run_name='__main__')
        value=json.loads(output.getvalue());assert value['Slurm_submitted']==0 and not value['intent_created']
        assert value['slot']==0 and value['HTTP']==64 and value['global_HTTP']==512
        write(B/'actual-launcher-main-recording-queries.json',dict(argv=launch_argv,output=value,queries=scheduler))
        q['cases'].append(dict(name='actual_launch_once___main___no_Popen',passed=True,recorded_Slurm_queries=scheduler));save()
        from public_glm_native.protocol import Run
        from startup_gate import StartupGate,StartupTimeout,drain_after_stop,seal_startup_records
        run=Run(configuration(B/'gates','ready120'));publish_ready(run)
        gate=StartupGate(run);assert not gate.observe(100.)and not gate.observe(219.)
        try:gate.observe(220.)
        except StartupTimeout:pass
        else:raise AssertionError('Missing all16ready120 bound')
        run.write('MODEL_READY.json',dict(Model_READY=True,actual_owner_count=16))
        try:gate.observe(221.)
        except StartupTimeout:pass
        else:raise AssertionError('Late READY enabled client after failed startup')
        r2=Run(configuration(B/'gates','epoch2700'));g2=StartupGate(r2)
        try:g2.observe(2701.)
        except StartupTimeout:pass
        else:raise AssertionError('Missing originalepoch2700 bound')
        r3=Run(configuration(B/'gates','success'));publish_ready(r3)
        r3.write('MODEL_READY.json',dict(Model_READY=True,actual_owner_count=16));g3=StartupGate(r3)
        assert g3.observe(100.)and g3.observe(5000.) # Startup cap is not a new service deadline.
        q['cases'].append(dict(name='actual_Run_all16_ready_and_dual_bound_latch',passed=True,
                               ready120=gate.state(),epoch2700=g2.state(),success=g3.state(),recorded_affinity_not_CPU_or_GPU_qualification=True));save()
        class Proc:
            pid=987654321
            def poll(self):return 0
        ticks=[0.]
        def sleep(s):ticks[0]+=s
        def exact(label,timeout=25):return ['910376.123',label,'COMPLETED','0:0']if ticks[0]>=2 else None
        normal=drain_after_stop(Proc(),exact,'owned','910376.123',clock=lambda:ticks[0],sleep=sleep)
        assert normal['normal_GPU_STOP_drain']and ticks[0]==2
        ticks[0]=0
        unknown=drain_after_stop(Proc(),lambda label,timeout:None,'owned','910376.123',clock=lambda:ticks[0],sleep=sleep)
        assert not unknown['normal_GPU_STOP_drain']and ticks[0]==60
        # Execute the original failstop implementation with only signal/process
        # leaves recorded; confirms TERM then KILL on exact owned step/PGID.
        sent=[];ticks[0]=0
        class LiveProc:
            pid=987654321
            def poll(self):return 0 if any(x[0]=='pg'and x[-1]==9 for x in sent)else None
        def killpg(pid,kind):
            if kind==0:
                if LiveProc().poll()==0:raise ProcessLookupError()
            else:sent.append(('pg',pid,int(kind)))
        def cancel(argv,**kw):
            sent.append(('scancel',*argv));return subprocess.CompletedProcess(argv,0,'','')
        def accounting(label):return ['910376.123',label,'CANCELLED'if LiveProc().poll()==0 else'RUNNING','0:9'if LiveProc().poll()==0 else'0:0']
        with patch.object(C.os,'killpg',killpg),patch.object(C.subprocess,'run',cancel),patch.object(C.time,'monotonic',lambda:ticks[0]),patch.object(C.time,'sleep',sleep):
            closed=C.close_owned('owned',LiveProc(),'910376.123',accounting,100.,term_s=2,kill_s=2)
        assert closed['closed']and [x[-1]for x in sent if x[0]=='pg']==[15,9]
        q['cases'].append(dict(name='STOP60_then_existing_exact_TERM_KILL',passed=True,normal=normal,unknown=unknown,closed=closed,signals_recorded_not_sent=sent));save()
        # Execute the actual changed Python startup producers. The process and
        # loaded-owner bodies are recorded leaves; actual Run all_ready and
        # MODEL_READY publication, thread starts, markers and stop are not.
        from public_glm_native import actors
        from public_glm_native.provider import GLMProvider
        class Engine:
            def __init__(self):
                self._admission_lock=threading.RLock();self._lock=threading.RLock();self._requests={}
            def start(self):pass
            def stop(self):pass
        cfg=configuration(B/'actor-flow','actual-startup');cfg['gpu_start_epoch']=time.time()
        actual_run=Run(cfg);publish_ready(actual_run)
        for rank in range(8):
            actual_run.write('closed-0-%d.json'%rank,dict(stage=0,rank=rank,normal=True,
                post_close_Torch_liveness=True,recorded_owner_body=True))
        actual_run.write('COMMUNICATORS.json',dict(payload={}))
        actual_run.write('PREPARED.json',dict(prepared_identity={'recorded_loaded_owner_leaf':True}))
        class RecordedProcess:
            def __init__(self,*,target,args,name):
                self.rank=args[0];self.name=name;self.pid=900000+self.rank;self.exitcode=None
            def start(self):
                actual_run.write('closed-1-%d.json'%self.rank,dict(stage=1,rank=self.rank,normal=True,
                    post_close_Torch_liveness=True,recorded_owner_body=True))
                self.exitcode=0
            def is_alive(self):return False
            def join(self,timeout=None):pass
            def terminate(self):raise AssertionError('Unexpected startup terminate')
            def kill(self):raise AssertionError('Unexpected startup kill')
        class RecordedSpawn:Process=RecordedProcess
        def owner_leaf(rank,config,prepared,stop_intent):
            assert rank==0
            actual_run.write('closed-1-0.json',dict(stage=1,rank=0,normal=True,
                post_close_Torch_liveness=True,recorded_owner_body=True))
        provider=GLMProvider(cfg,dict(engine=Engine(),identity={'recorded_prepared_identity':True}))
        with patch.object(actors.multiprocessing,'get_context',lambda method:RecordedSpawn()),patch.object(actors,'owner_main',owner_leaf):
            provider.start();assert provider._ready
            assert actual_run.read('MODEL_READY.json')['actual_owner_count']==16
            provider.stop()
        events=[]
        for f in actual_run.path.glob('startup-*.jsonl'):
            events.extend(json.loads(line)['event']for line in f.read_text().splitlines())
        assert all(v in events for v in ['actors.start.return','actors.monitor_thread_start.return',
            'provider.group_start.return','provider.wait_ready.return','provider.MODEL_READY_write.return'])
        # Retain a startup primary exception if diagnostic cancellation also fails.
        from public_glm_native import provider as provider_module
        bad_cfg=configuration(B/'actor-flow','primary-error')
        broken=GLMProvider(bad_cfg,dict(engine=Engine(),identity={}))
        primary=RuntimeError('CPU recorded startup primary')
        cancel_errors=[]
        def fail_start():raise primary
        def fail_cancel(*args):
            cancel_errors.append('CPU recorded diagnostic cancel error')
            raise OSError(cancel_errors[-1])
        broken.group.start=fail_start
        with patch.object(provider_module,'resolve',fail_cancel):
            try:broken.start()
            except RuntimeError as error:assert error is primary
            else:raise AssertionError('Lost startup primary')
        assert cancel_errors==['CPU recorded diagnostic cancel error']
        notes=getattr(primary,'__notes__',[])
        if hasattr(primary,'add_note'):assert any('diagnostic cancel error'in v for v in notes)
        q['cases'].append(dict(name='actual_Actors_provider_start_stop_and_primary',passed=True,
            events=events,primary_notes=notes,recorded_diagnostic_cancel_errors=cancel_errors,recorded_process_owner_affinity=True,
            not_Native_or_device_qualification=True));save()
        # Real marker helper cancellation-before-arm race, no new watchdog here.
        from public_glm_native import startup_diagnostic as diag
        cfg=configuration(B/'cancel','resolve-before-arm')
        diag.resolve(cfg,'CPU.before_arm');diag.arm_after_owner_ready(cfg,0)
        assert not list((Path(cfg['runtime_directory'])/cfg['run']).glob('startup-stack-*.txt'))
        snapshot=seal_startup_records(Path(cfg['runtime_directory'])/cfg['run'],B/'marker-originals.json')
        q['cases'].append(dict(name='actual_helper_resolve_before_late_arm',passed=True,snapshot=snapshot));save()
        child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--watchdog-child'],capture_output=True,text=True,timeout=60)
        write(B/'actual-C-watchdog-child.json',dict(rc=child.returncode,stdout=child.stdout,stderr=child.stderr))
        assert child.returncode==0,child.stderr
        q['cases'].append(dict(name='actual_repeated_C_watchdog_then_cancel',passed=True,proof=json.loads(child.stdout)));save()
        # The source-injected Native callback/order is inspected, not executed
        # against a fake SDK; actual control/GIL behavior remains device UNKNOWN.
        owner=(P/'public_glm_native/native_owner.py').read_text()
        assert owner.count('control(tensor)')==1
        assert owner.index("startup_marker('owner.first_idleControl.before'")<owner.index('control(tensor)')<owner.index("startup_marker('owner.first_idleControl.return'")
        assert not any(n in sys.modules for n in ['torch','_infinilm','transformers'])
        q['proc_maps_after']=Path('/proc/self/maps').read_text()
        assert 'libGLM_V_on_demand_Model_r403.so'not in q['proc_maps_after']
        q.update(completed=True,passed=True,source_ordering_only_not_Native_execution=True,
                 diagnostic_GPU_qualification=False,old_quality_or_encoding_suites_repeated=False)
    except BaseException as error:
        q.update(completed=True,passed=False);q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:save()
    print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],cases=len(q['cases']),receipt=str(result),SHA256=sha(result))))
    return 0 if q['passed']else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watchdog-child',action='store_true');a=p.parse_args()
    if a.watchdog_child:watchdog_child()
    else:raise SystemExit(main())
