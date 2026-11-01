"""Run the actual controller against recording Slurm and real temporary markers.

No Native import, Context, target worker, model weights, GPU, or real Slurm call.
Linux additionally exercises the real passive /proc listener selector using a
small recording child; it never imports or starts the r334 GPU worker.
"""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest import mock

import contract
import controller
import invoke_once
import resource_probe
import slurm_backend

HERE = Path(__file__).resolve().parent
NONCE = 'recording-controller-nonce-' + '7' * 48
CASES = []


def plan_copy():
    p = json.loads((HERE/'execution-plan.template.json').read_bytes())
    p['execute_once'] = True
    p['publication_commit'] = '1' * 40
    for key in list(p):
        if key.endswith('_SHA256') and set(p[key]) == {'0'}: p[key] = '2' * 64
    contract.validate_plan(p, execution=True)
    return p


def rejected(fn, kinds=(AssertionError, ValueError, RuntimeError, FileExistsError)):
    try: fn()
    except kinds: return
    raise AssertionError('Expected rejection did not occur')


def case(name, fn):
    try:
        result = fn()
        CASES.append(dict(name=name, passed=True, detail=result))
    except BaseException as error:
        CASES.append(dict(name=name, passed=False, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc()))


class RecordingClock:
    def __init__(self): self.now = 1000000.; self.ticks = 0; self.boundary = None
    def __call__(self): return self.now
    def sleep(self, seconds):
        self.now += seconds; self.ticks += 1
        self.boundary.advance()
        assert self.ticks <= 100, 'CPU test itself must remain bounded'


class RecordingSlurm:
    """External worker/state observations only; orchestration stays in Controller."""
    def __init__(self, root, plan, clock, scenario):
        self.root, self.plan, self.clock, self.scenario = root, plan, clock, scenario
        self.clock.boundary = self; self.starts = []; self.events = []; self.cancelled = []
        self.epoch = None; self.ctrl = None; self.finish_tick = None; self.listener_calls = 0

    def validate_resources(self):
        self.events.append('validate_resources')
        if self.scenario == 'preflight_failure': raise RuntimeError('recording invalid resource')

    def no_foreign(self): self.events.append('no_foreign')

    def idle(self, suffix):
        self.events.append('idle:' + suffix)
        if self.scenario == 'post_failure' and suffix in ['post', 'failure-post']:
            raise RuntimeError('recording post failure')
        return [dict(node=n, ib_ipv4=['10.0.0.'+str(i+1)], passed=True,
                     KFD_empty=True, kfd_pids=[], ports_bindable={'46947': True})
                for i,n in enumerate(contract.TARGET_NODES+[contract.DRAFT_NODE])]

    def start(self, kind, environment, epoch):
        assert kind not in self.starts, 'Second residency forbidden'
        if self.scenario == 'target_spawn_failure' and kind == 'target':
            raise OSError('recording Popen failure before child')
        if kind == 'draft':
            assert 'AR1_finished' in self.events and 'AR2_finished' in self.events
            assert contract.read_start(self.ctrl.start_marker, self.identity)
            assert not self.ctrl.listening_marker.exists()
            self.finish_tick = self.clock.ticks + 1
        else:
            self.epoch = epoch
            self.identity = contract.marker_identity(self.plan['run'],
                hashlib.sha256(environment['KIMI_LONG_SESSION_R347'].encode()).hexdigest(), epoch)
            assert environment['KIMI_LONG_NATIVE_DEADLINE_UNIX_R347'] == str(epoch+contract.NATIVE_SECONDS)
        self.starts.append(kind); self.events.append('start:' + kind)
        return dict(submitted=True, recording=True, srun_client_PID=100+len(self.starts))

    def advance(self):
        t = self.clock.ticks
        if t == 1: self.events.append('AR1_finished')
        if self.scenario == 'cancel' and t == 1:
            contract.publish_once(self.ctrl.cancel_marker, {'user_cancel': True}); return
        if self.scenario in ['no_marker', 'srun0_account_unknown', 'srun0_account_running']:
            if t == 3 and self.scenario == 'no_marker': self.clock.now = self.epoch+contract.GPU_SECONDS+1
            return
        if t == 2:
            self.events.append('AR2_finished')
            value = dict(self.identity)
            if self.scenario == 'bad_nonce': value['nonce_SHA256'] = 'f' * 64
            if self.scenario == 'bad_node': value['requested_by_node'] = 1
            if self.scenario == 'bad_epoch_type': value['original_GPU_start_unix'] = int(self.epoch)
            if self.scenario == 'bad_JSON': self.ctrl.start_marker.write_bytes(b'not-json\n')
            elif self.scenario in ['partial', 'partial_never']:
                self.ctrl.start_marker.write_text(json.dumps(value), encoding='utf8')
            else: contract.publish_once(self.ctrl.start_marker, value)
            if self.scenario == 'native_deadline': self.clock.now = self.epoch+contract.NATIVE_SECONDS-180
        if t == 3 and self.scenario == 'partial':
            assert self.starts == ['target'], 'Partial write launched a draft'
            self.ctrl.start_marker.write_bytes(self.ctrl.start_marker.read_bytes()+b'\n')
        if t == 3 and self.scenario == 'partial_never': self.clock.now = self.epoch+contract.GPU_SECONDS+1

    def observe(self, kind):
        t = self.clock.ticks
        parent = '910380' if kind == 'target' else '910371'
        nodes = ','.join(contract.TARGET_NODES) if kind == 'target' else contract.DRAFT_NODE
        row = [parent+'.101', kind, 'RUNNING', '0:0', nodes]; exit_code = None
        if kind == 'target' and self.scenario == 'target_failed': row[2:4] = ['FAILED', '1:0']; exit_code = 1
        elif kind == 'target' and self.scenario == 'target_early_complete': row[2] = 'COMPLETED'; exit_code = 0
        elif kind == 'target' and self.scenario == 'srun_nonzero_early': exit_code = 1
        elif kind == 'target' and self.scenario in ['srun0_account_unknown', 'srun0_account_running']:
            row = None if self.scenario == 'srun0_account_unknown' else row; exit_code = 0
        elif self.finish_tick is not None and t >= self.finish_tick:
            row[2] = 'COMPLETED'; exit_code = 0
            if self.scenario == 'target_exit_nonzero' and kind == 'target': exit_code = 1
            if self.scenario == 'draft_exit_nonzero' and kind == 'draft': exit_code = 1
            if self.scenario == 'draft_failed' and kind == 'draft': row[2:4] = ['FAILED', '1:0']; exit_code = 1
            if self.scenario == 'exit_lag' and t < self.finish_tick+2: exit_code = None
            if self.scenario == 'exit_never' and kind == 'target': exit_code = None
            if self.scenario == 'account_lag' and kind == 'target' and t < self.finish_tick+2: row = None
        self.events.append('observe:'+kind)
        return dict(row=row, srun_exit_code=exit_code)

    def draft_listening(self, epoch):
        self.listener_calls += 1
        assert self.starts == ['target', 'draft'] and epoch == self.epoch
        assert not self.ctrl.listening_marker.exists()
        if self.scenario == 'listener_failure': raise TimeoutError('recording passive listener failure')
        return dict(actual_draft_step='910371.101', listener_ready=True,
                    probe=dict(TCP_connections_made=0, passive_proc_observation=True))

    def cancel_owned(self):
        self.cancelled.extend(self.starts)
        if self.scenario == 'cleanup_failure': raise RuntimeError('recording context cleanup failure')
        return [dict(recording=True, kind=k, owned_only=True) for k in self.starts]

    def collect_success(self):
        assert self.starts == ['target', 'draft'] and self.listener_calls == 1
        ready = json.loads(self.ctrl.listening_marker.read_bytes())
        assert ready == dict(self.identity, actual_draft_step='910371.101', listener_ready=True)
        assert self.events[-1] == 'idle:post'
        if self.scenario in ['evidence_failure', 'cleanup_failure']:
            raise RuntimeError('recording quality/context close failure')
        return dict(recording_Slurm=True, actual_Native=False, actual_GPU=False)


def run_controller_scenario(scenario):
    with tempfile.TemporaryDirectory(prefix='Kimi347-controller-') as tmp:
        root = Path(tmp); (root/'results').mkdir(); (root/'control').mkdir()
        clock = RecordingClock(); plan = plan_copy(); backend = RecordingSlurm(root, plan, clock, scenario)
        ctl = controller.Controller(root, plan, backend, NONCE, clock=clock, sleep=clock.sleep); backend.ctrl = ctl
        result = ctl.run(); expected = scenario in ['normal', 'partial', 'exit_lag', 'account_lag']
        assert result['completed'] and result['passed'] == expected, result
        assert backend.starts.count('target') <= 1 and backend.starts.count('draft') <= 1
        no_draft = scenario in ['preflight_failure','target_spawn_failure','target_failed','target_early_complete',
            'srun_nonzero_early','bad_nonce','bad_node','bad_epoch_type','bad_JSON','no_marker',
            'partial_never','native_deadline','cancel','srun0_account_unknown','srun0_account_running']
        if no_draft: assert 'draft' not in backend.starts, backend.starts
        if expected:
            assert not backend.cancelled and result['post_empty']
            count = len(backend.starts); rejected(ctl.run); assert len(backend.starts) == count
        elif backend.starts:
            assert set(backend.cancelled) == set(backend.starts)
            assert ctl.cancel_marker.exists()
        if scenario == 'cancel': assert json.loads(ctl.cancel_marker.read_bytes()) == {'user_cancel': True}
        if scenario in ['exit_never','srun0_account_unknown','srun0_account_running']:
            assert 'unresolved' in result['error'] and 90 < clock.now-(ctl.epoch or clock.now) < 150
        if scenario == 'cleanup_failure': assert result['cleanup_error_type'] == 'RuntimeError' and result['post_empty']
        if scenario == 'post_failure': assert result['post_error_type'] == 'RuntimeError'
        return dict(expected_pass=expected, starts=backend.starts, passive_listener_calls=backend.listener_calls,
                    ticks=clock.ticks, final_state=result['state'], error_type=result.get('error_type'))


def test_existing_markers():
    for name in ['watch','start_marker','listening_marker','cancel_marker']:
        with tempfile.TemporaryDirectory(prefix='Kimi347-existing-') as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir();clock=RecordingClock()
            backend=RecordingSlurm(root,plan_copy(),clock,'normal');ctl=controller.Controller(root,plan_copy(),backend,NONCE,clock=clock,sleep=clock.sleep);backend.ctrl=ctl
            path=getattr(ctl,name);path.write_bytes(b'');rejected(ctl.run);assert not backend.starts and path.read_bytes()==b''
    # Force the old check/write race; actual exclusive first watch must still reject.
    with tempfile.TemporaryDirectory(prefix='Kimi347-exclusive-') as tmp:
        root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir();clock=RecordingClock()
        backend=RecordingSlurm(root,plan_copy(),clock,'normal');ctl=controller.Controller(root,plan_copy(),backend,NONCE,clock=clock,sleep=clock.sleep);backend.ctrl=ctl
        ctl.watch.write_bytes(b'original');original_exists=Path.exists
        with mock.patch.object(Path,'exists',lambda p:False if p==ctl.watch else original_exists(p)):
            rejected(ctl.run)
        assert ctl.watch.read_bytes()==b'original' and not backend.starts
    return dict(empty_predecessors=4, exclusive_firstwatch_race=True)


def test_real_backend_argv():
    commands=[]
    with tempfile.TemporaryDirectory(prefix='Kimi347-argv-') as tmp:
        root=Path(tmp);(root/'logs').mkdir();backend=slurm_backend.SlurmBackend(root,HERE,plan_copy())
        def popen(args,**kw):
            assert kw['start_new_session'] is True and kw['stdin']==subprocess.DEVNULL
            assert Path(kw['stdout'].name).is_absolute()
            commands.append((args,kw));return SimpleNamespace(pid=200+len(commands),poll=lambda:None)
        now=1000000.25;epoch=now-600.75
        with mock.patch.object(slurm_backend.time,'time',return_value=now),mock.patch.object(slurm_backend.subprocess,'Popen',side_effect=popen):
            backend.start('target',{'ORIGINAL_DEADLINE':'unchanged'},epoch)
            backend.start('draft',{'ORIGINAL_DEADLINE':'unchanged'},epoch)
            rejected(lambda:backend.start('draft',{},epoch))
        expected=math.floor(epoch+contract.GPU_SECONDS-now);wall='%02d:%02d:%02d'%(expected//3600,(expected%3600)//60,expected%60)
        target,draft=[x[0] for x in commands]
        assert '--jobid=910380' in target and '-N4' in target and '-n4' in target and '--gres=dcu:8' in target
        assert '--jobid=910371' in draft and '-N1' in draft and '-n1' in draft and '--gres=dcu:1' in draft
        assert ','.join(contract.TARGET_NODES) in target and contract.DRAFT_NODE in draft
        assert target[target.index('-t')+1]==draft[draft.index('-t')+1]==wall
        assert all(x.startswith('--output=') and Path(x.split('=',1)[1]).is_absolute() for a,_ in commands for x in a if x.startswith('--output='))
        assert commands[0][1]['env']['WAVE_INSIDE_R2323'].endswith('target_inside.sh')
        assert commands[1][1]['env']['WAVE_INSIDE_R2323'].endswith('draft_inside.sh')
        assert commands[0][1]['env']['ORIGINAL_DEADLINE']=='unchanged'
        other=slurm_backend.SlurmBackend(root,HERE,plan_copy())
        with mock.patch.object(slurm_backend.time,'time',return_value=epoch+contract.GPU_SECONDS-180),mock.patch.object(slurm_backend.subprocess,'Popen',side_effect=AssertionError('unexpected real launch')):
            rejected(lambda:other.start('target',{},epoch))
    return dict(recording_Popen_calls=2,target_cards=32,draft_cards=1,original_remaining_seconds=expected,wall=wall)


def test_backend_owned_cleanup():
    if not hasattr(signal,'SIGKILL'): return dict(skipped='POSIX signals unavailable; run in CPU container')
    killed=[];cancelled=[]
    with tempfile.TemporaryDirectory(prefix='Kimi347-cleanup-') as tmp:
        b=slurm_backend.SlurmBackend(Path(tmp),HERE,plan_copy());b.processes={'target':SimpleNamespace(pid=4321,poll=lambda:0,wait=lambda timeout:0)}
        b.account=lambda kind:['910380.42','target','RUNNING','0:0',','.join(contract.TARGET_NODES)]
        def killpg(pid,sig):assert pid==4321;killed.append(int(sig))
        def scancel(a,**kw):cancelled.append(a);return SimpleNamespace(returncode=0)
        with mock.patch.object(slurm_backend.os,'killpg',side_effect=killpg,create=True),mock.patch.object(slurm_backend.subprocess,'run',side_effect=scancel):b.cancel_owned()
        assert cancelled==[['scancel','--signal=TERM','910380.42']]
        assert killed==[int(signal.SIGTERM),0,int(signal.SIGKILL)],killed
    return dict(exited_leader_original_PGID_reaped=True,owned_step_only=True,signals=killed)


def test_actual_account_and_foreign():
    with tempfile.TemporaryDirectory(prefix='Kimi347-account-') as tmp:
        b=slurm_backend.SlurmBackend(Path(tmp),HERE,plan_copy())
        b.expand=lambda s:s.split(',')
        good=['910380.50',b.label('target'),'RUNNING','0:0',','.join(contract.TARGET_NODES)]
        b.account_rows=lambda parent:[good] if parent==910380 else []
        assert b.account('target')==good and b.actual['target']=='910380.50'
        for bad in [dict(id='910371.50'),dict(nodes=contract.DRAFT_NODE)]:
            bb=slurm_backend.SlurmBackend(Path(tmp),HERE,plan_copy());bb.expand=b.expand
            row=good.copy();row[0]=bad.get('id',row[0]);row[4]=bad.get('nodes',row[4]);bb.account_rows=lambda parent,r=row:[r]
            rejected(lambda:bb.account('target'))
        b.account_rows=lambda parent:[good,good] if parent==910380 else []
        rejected(lambda:b.account('target'))
        foreign=['910380.51','other-owner','RUNNING','0:0',contract.TARGET_NODES[0]]
        b.account_rows=lambda parent:[good,foreign] if parent==910380 else []
        rejected(b.no_foreign)
        b.account_rows=lambda parent:[good] if parent==910380 else []
        b.no_foreign()
        changed=good.copy();changed[0]='910380.99';b.account_rows=lambda parent:[changed]
        rejected(lambda:b.account('target'))
    return dict(wrong_parent_rejected=True,wrong_nodes_rejected=True,duplicate_label_rejected=True,
                active_foreign_rejected=True,actual_handle_change_rejected=True,owned_handle_allowed=True)


def test_actual_collect_success():
    scenarios=['normal','target_incomplete','wrong_source','wrong_step','leader_signature','quality_failed',
               'no_eos','budget_changed','draft_failed','draft_abnormal_close','owner_close_incomplete',
               'capture_invalid','draft_unauthenticated','draft_bad_GPU_count','phase_missing_prompt',
               'phase_begin_count','phase_query_count','commands_not_sum','feature_bytes_short']
    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix='Kimi347-evidence-') as tmp:
            root=Path(tmp);(root/'results').mkdir();plan=plan_copy();b=slurm_backend.SlurmBackend(root,HERE,plan)
            b.actual={'target':'910380.50','draft':'910371.51'}
            requests=[dict(name=name,mode=mode,completed=True,normal_reset=True,finish='eos',
                quality={'passed':True},output_budget=128,tokens=5,prompt_SHA256=('a' if name=='retained8635' else 'b')*64,
                ids_SHA256=str(i)*64) for i,(name,mode) in enumerate(plan['request_order'])]
            for node in range(4):
                q=dict(completed=True,all8_owned_workers_shutdown_completed=True,actual_step='910380.50',
                       source_manifest_SHA256=plan['target_manifest_SHA256'],Native_SHA256=plan['Native_SHA256'],
                       first_capture_validation=dict(passed=True),
                       pair_report=dict(completed=True,normal_draft_protocol_close=True,requests=copy.deepcopy(requests)))
                if node==2:
                    if scenario=='target_incomplete':q['completed']=False
                    if scenario=='wrong_source':q['source_manifest_SHA256']='f'*64
                    if scenario=='wrong_step':q['actual_step']='910380.99'
                    if scenario=='owner_close_incomplete':q['all8_owned_workers_shutdown_completed']=False
                    if scenario=='capture_invalid':q['first_capture_validation']['passed']=False
                    if scenario=='leader_signature':q['pair_report']['requests'][0]['ids_SHA256']='f'*64
                    if scenario=='quality_failed':q['pair_report']['requests'][0]['quality']['passed']=False
                    if scenario=='no_eos':q['pair_report']['requests'][0]['finish']='length'
                    if scenario=='budget_changed':q['pair_report']['requests'][0]['output_budget']=127
                (root/'results'/('Kimi-long-pair-r346-'+plan['run']+'-node%d.json'%node)).write_text(json.dumps(q))
            phases=dict(begin=2,seal=2,retire=2,close=1,append_prompt=81,query=2)
            draft=dict(completed=True,passed=True,normal_protocol_close=True,actual_step='910371.51',source_manifest_SHA256=contract.WORKER_SHA,
                       authenticated=True,ready=True,GPU_count=1,phase_counts=phases,
                       commands=sum(phases.values()),feature_bytes=(8635+32638)*35840*2)
            if scenario=='draft_failed':draft['passed']=False
            if scenario=='draft_abnormal_close':draft['normal_protocol_close']=False
            if scenario=='draft_unauthenticated':draft['authenticated']=False
            if scenario=='draft_bad_GPU_count':draft['GPU_count']=2
            if scenario=='phase_missing_prompt':phases['append_prompt']=80
            if scenario=='phase_begin_count':phases['begin']=1
            if scenario=='phase_query_count':phases['query']=1
            if scenario=='commands_not_sum':draft['commands']+=1
            if scenario=='feature_bytes_short':draft['feature_bytes']-=1
            (root/'results'/('Kimi-long-draft-r334-'+plan['run']+'.json')).write_text(json.dumps(draft))
            if scenario=='normal':
                q=b.collect_success();assert q['all32_normal'] and q['four_request_signatures_equal'] and not q['client_SLO_qualified']
            else:rejected(b.collect_success)
    return dict(actual_backend_validator=True,cases=len(scenarios),four_leaders=True,
                target_cards=32,draft_cards=1,actual_GPU=False,synthetic_reports=True)


def test_actual_invoker_exclusive_intent():
    total=0
    for scenario in ['normal_then_duplicate','existing_empty_intent','existing_empty_watch']:
        with tempfile.TemporaryDirectory(prefix='Kimi347-invoker-') as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();plan=plan_copy();plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            intent=root/'results'/('Kimi-long-pair-r347-'+plan['run']+'-intent.json');watch=root/'results'/('Kimi-long-pair-r347-'+plan['run']+'-watch.json')
            if scenario=='existing_empty_intent':intent.write_bytes(b'')
            if scenario=='existing_empty_watch':watch.write_bytes(b'')
            calls=[]
            def popen(args,**kw):
                assert kw['start_new_session'] and '--gres=none' in args and '-t' in args
                assert args[args.index('-t')+1]=='09:00:00' and '--jobid=910407' in args
                assert Path(args[-1]).is_absolute() and args[-2]=='--plan'
                assert len(kw['env']['KIMI_LONG_SESSION_R347'])==64
                calls.append(args);return SimpleNamespace(pid=555)
            with mock.patch.object(invoke_once,'ROOT',root),mock.patch.object(invoke_once,'DIRECTORY',HERE),mock.patch.object(invoke_once,'verify_sources',return_value={}),mock.patch.object(slurm_backend.SlurmBackend,'validate_resources',return_value=None),mock.patch.object(subprocess,'Popen',side_effect=popen),mock.patch.object(sys,'argv',['invoke_once.py','--plan',str(plan_path)]),contextlib.redirect_stdout(io.StringIO()):
                if scenario=='normal_then_duplicate':
                    invoke_once.main();rejected(invoke_once.main);assert len(calls)==1
                    record=json.loads(intent.read_bytes());assert record['CPU_keeper_only'] and not record['target_GPU_submitted']
                else:rejected(invoke_once.main);assert not calls
            total+=len(calls)
    return dict(actual_invoker=True,recording_keeper_Popen=total,duplicate_intent_suppressed=True,
                empty_watch_suppressed=True,source_verifier_mocked_before_manifest_generation=True)


def test_passive_proc_listener():
    if os.name!='posix' or not Path('/proc/net/tcp').exists():return dict(skipped='Real /proc passive listener requires Linux CPU container')
    worker='/refactor/incoming/Kimi-long-draft-worker-r334b/worker.py';output='/refactor/results/recording-only-Kimi347-selector.json'
    body="import socket,json,sys; s=socket.socket();s.bind(('127.0.0.1',0));s.listen(1);print(json.dumps({'port':s.getsockname()[1]}),flush=True);sys.stdin.readline();s.close()"
    child=subprocess.Popen([sys.executable,'-u','-c',body,worker,output],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        # Child only listens; the real selector must inspect proc without connecting.
        line=child.stdout.readline();port=json.loads(line)['port']
        with mock.patch.object(socket.socket,'connect',side_effect=AssertionError('Active connection forbidden')),mock.patch.object(socket.socket,'connect_ex',side_effect=AssertionError('Active connection forbidden')):
            assert resource_probe.listener_owner(port,output)==[child.pid]
            assert resource_probe.listener_owner(port,output+'-wrong')==[]
        child.stdin.write('close\n');child.stdin.flush();child.wait(timeout=10);assert child.returncode==0
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
    return dict(recording_worker_like_child=True,actual_r334_worker=False,TCP_connections_made=0,actual_proc_selector=True)


def main():
    scenarios=['normal','partial','bad_nonce','bad_node','bad_epoch_type','bad_JSON','partial_never','no_marker','native_deadline','cancel','preflight_failure','target_spawn_failure','target_failed','target_early_complete','srun_nonzero_early','target_exit_nonzero','draft_exit_nonzero','draft_failed','listener_failure','exit_lag','account_lag','exit_never','srun0_account_unknown','srun0_account_running','evidence_failure','cleanup_failure','post_failure']
    for name in scenarios:case('actual_Controller_'+name,lambda n=name:run_controller_scenario(n))
    case('existing_markers_and_exclusive_firstwatch',test_existing_markers)
    case('actual_SlurmBackend_argv_original_deadline',test_real_backend_argv)
    case('actual_SlurmBackend_account_and_foreign_boundaries',test_actual_account_and_foreign)
    case('actual_SlurmBackend_fourleader_quality_normal_evidence',test_actual_collect_success)
    case('actual_invoker_exclusive_intent_no_replay',test_actual_invoker_exclusive_intent)
    case('actual_SlurmBackend_owned_group_cleanup',test_backend_owned_cleanup)
    case('actual_passive_proc_listener_no_connect',test_passive_proc_listener)
    passed=all(x['passed'] for x in CASES)
    report=dict(completed=True,passed=passed,CPU_only=True,GPU=0,Native_dlopen=False,
                actual_Context=False,actual_Native=False,actual_Slurm=False,actual_model_load=False,
                scope='Recording Slurm plus actual Controller, real tempfile markers, actual backend argv/cleanup and optional Linux proc selector; not GPU execution authorization',
                cases=CASES,source_SHA256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['controller.py','slurm_backend.py','contract.py','resource_probe.py','invoke_once.py','execution-plan.template.json','check_controller_CPU.py']})
    print(json.dumps(report,ensure_ascii=False));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
