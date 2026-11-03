"""One resident NoV16 Model, six paired W2/W4 full-history requests."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
S=R/'incoming/GLM-NoV-16card-W4-pair-r414'
LABEL='GLM-NoV-16card-W4-pair-r414';RUN='pair-20261103-a';NODES=['a02r4n12','a02r4n13']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def publish(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def verify(root,source,digest):
    from verify_source import verify_source
    m=verify_source(root,source,digest)
    assert m['GPU_GO']is True and m['Model_GO']is True
    p=root/'candidates/GLM-NoV-16card-W4-pair-r414-CPU/CPU-result.json';assert sha(p)==m['CPU_receipt_SHA256'];q=json.loads(p.read_bytes())
    assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']and q['owner_manifest_SHA256']==m['owner_manifest_SHA256']and q['source_manifest_SHA256']==m['source_manifest_SHA256']
    return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publication-commit',required=True);parser.add_argument('--manifest-sha',required=True);args=parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.publication_commit)
    assert os.environ['SLURM_JOB_ID']=='910407' and socket.gethostname().split('.')[0]=='a14r2n20'
    m=verify(R,S,args.manifest_sha);cleanup_deadline=time.monotonic()+5980
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_')) or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    W=R/'results'/(LABEL+'-'+RUN+'-watch.json');cancel=R/'control'/(LABEL+'-'+RUN+'-cancel.json');go=R/'control'/(LABEL+'-'+RUN+'-GO.json')
    watch=dict(completed=False,passed=False,run=RUN,publication_commit=args.publication_commit,GPU_manifest_SHA256=args.manifest_sha,
               CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,Model_loaded=None,checkpoint_IO=None,checkpoint_IO_planned=True,compile=False,
               GPU_max_s=5400,CPU_keeper_max_s=6000,retry=False,reload=False,forced_cleanup=False)
    assert not any(p.exists() for p in [W,cancel,go]);publish(W,watch);proc=None;actual=None;row=None
    def save():
        watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();temp=W.with_suffix('.tmp');temp.write_text(json.dumps(watch,indent=2)+'\n');temp.replace(W)
    def command(command,timeout=20):return subprocess.check_output(command,env=env,text=True,timeout=min(timeout,max(1,cleanup_deadline-time.monotonic())))
    def expand(nodes):return command(['scontrol','show','hostnames',nodes]).splitlines()
    def rows():return [s.split('|') for s in command(['sacct','-S','2026-10-31','-j','910376','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines() if s]
    def exact():
        found=[r for r in rows() if r[1]==LABEL+'-'+RUN];assert len(found)<=1
        if not found:return None
        r=found[0];assert re.fullmatch(r'910376[.]\d+',r[0]) and set(expand(r[4]))==set(NODES)
        if actual is not None:assert r[0]==actual
        return r
    def dedup(ignored=()):
        for r in rows():
            if re.fullmatch(r'910376[.]\d+',r[0]) and r[0]not in ignored and r[2].split()[0].rstrip('+')not in TERMINAL:
                assert not set(expand(r[4]))&set(NODES),'Activeforeignstep onclaimednodes'
    def idle(phase):
        cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(S/'controller/idle.py')]
        p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=min(115,max(1,cleanup_deadline-time.monotonic())))
        value=[json.loads(s) for s in p.stdout.splitlines() if s.startswith('{')];assert p.returncode==0 and {v['node'] for v in value}==set(NODES) and len(value)==2
        assert all(v['CPU_only'] and not v['GPU_initialized'] and v['KFD_available'] and not v['kfd_pids'] and not v['matching_component_processes'] and v['ports_bindable']=={'43843':True} for v in value)
        assert len({v['actual_step'] for v in value})==1
        assert len({(v['node'],p)for v in value for p in v['physical_PCI']})==16
        return value
    try:
        raw=command(['scontrol','show','job','910376','-o']);assert 'JobState=RUNNING'in raw and 'UserId=aclsylqidf('in raw
        assert set(NODES)<=set(expand(re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)))
        assert exact()is None;dedup();watch['fresh_before']=idle('before');dedup([r['actual_step'] for r in watch['fresh_before']]);save()
        epoch=time.time();deadline=epoch+5400;watch.update(original_GPU_start_unix=epoch,GPU_deadline_unix=deadline,Native_deadline_unix=epoch+5100,service_deadline_unix=epoch+4500)
        publish(go,dict(run=RUN,GPU_GO=True,Model_GO=True,operation='single_resident_C1PP2_noV_six_W2_W4_identical_tool_requests',source_manifest_SHA256=m['source_manifest_SHA256'],owner_manifest_SHA256=m['owner_manifest_SHA256'],Native_SHA256=m['Native_SHA256'],original_GPU_start_unix=epoch,publication_commit=args.publication_commit))
        env.update(WAVE_INSIDE_R2323='GLM-NoV-16card-W4-pair-r414/owner/GPU_inside_r290.sh',GLM_R290_GPUSTART_UNIX=str(epoch),GLM_SOURCE_MANIFEST_SHA=m['owner_manifest_SHA256'],GLM_RUN_R5040=RUN,GLM_R404_CANCEL_FILE='/refactor/control/'+cancel.name,
                   GLM_PP_SIZE_R3603='2',GLM_CACHE_ROWS_R3627='1',GLM_CACHE_CAP_R3627='32768',GLM_NETWORK_TRIAL='shca',GLM_BOUNDED_ROPE_R4279='1',GLM_SHARED_CONTROL_R4279='1',GLM_PIPELINE_TAIL_NODE='a02r4n13',GLM_EXACT_BANKS_R4188='1',GLM_CHANNELS_R4363='0',GLM_VARIANT_R4363='layer77-5783',INFINILM_OWNER_CPU_WIDTH_R2392='4',USE_WEIGHT_CACHE_R3604='0',GLM_SERVICE_HOST_R143='127.0.0.1',GLM_SERVICE_PORT_R143='43843',GLM_REGULAR_FIXTURE_RECEIPT_R180='/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json',GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180='ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf')
        cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=224000M','-t','01:30:00','--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),'bash',str(S/'controller/resource_wave_host_alias_outer.sh'),'probe']
        with(R/'logs'/(LABEL+'-'+RUN+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        watch.update(GPU_submitted=True,srun_PID=proc.pid);save()
        while time.time()<deadline:
            # Observation transport loss is not proof the owned healthy step
            # failed. Keep its original epoch/label/handle; never resubmit.
            try:row=exact()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as observation:
                watch.setdefault('transient_observation_errors',[]).append(dict(
                    at_unix=time.time(),error_type=type(observation).__name__,
                    error=str(observation)[:1000],actual_GPU=actual,label=LABEL+'-'+RUN))
                if proc.poll() not in (None,0):raise RuntimeError('Owned srun nonzero during observation loss')
                save();time.sleep(min(15,max(0,deadline-time.time())));continue
            if row:
                actual=row[0];watch.update(actual_GPU=actual,last_Slurm=row);save()
                if row[2].split()[0].rstrip('+')in TERMINAL:break
            if proc.poll() not in (None,0):raise RuntimeError('Owned srun nonzero')
            time.sleep(min(15,max(0,deadline-time.time())))
        else:raise TimeoutError('Original5400s deadline;neverrestart')
        watch['srun_exit_code']=proc.wait(timeout=min(25,max(1,cleanup_deadline-time.monotonic())))
        assert row[2]=='COMPLETED' and row[3]=='0:0' and watch['srun_exit_code']==0,'Originalfailedstep retained'
        from collect import collect
        summary=collect(R,RUN,actual,m['owner_manifest_SHA256'],min(time.time()+90,epoch+5880))
        watch.update(summary,Model_loaded=True,checkpoint_IO=True)
        env['GLM_R404_EXPECTED_PIDS']=json.dumps(summary['expected_node_PIDs'])
        env['GLM_R404_POST_DEADLINE_UNIX']=str(min(time.time()+90,epoch+5880))
        watch['post']=idle('post')
        logs=sorted((R/'logs').glob(LABEL+'-'+RUN+'-'+actual+'-*.log'));assert len(logs)==2
        assert sorted(marker for p in logs for marker in re.findall(r'GLM_REGULAR_SERVICE_R204_NORMAL ([01])',p.read_text(errors='replace')))==['0','1']
        watch.update(completed=True,passed=True,mp_spawn_join_markers=True,log_refs=[dict(path=str(p),SHA256=sha(p))for p in logs],state='C1_PP2_NO_V_SIX_REQUEST_W2_W4_SCOPE_PASS_PERFORMANCE_PENDING_AUDIT')

    except BaseException as error:
        watch.update(error=repr(error),error_frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]])
        if proc is not None:
            watch['cleanup_attempted']=True
            if not cancel.exists():publish(cancel,dict(run=RUN,cancelled=True,reason=type(error).__name__))
            try:
                row=exact()
                if row and row[2].split()[0].rstrip('+')not in TERMINAL:
                    watch['forced_cleanup']=True;subprocess.run(['scancel','--signal=TERM',row[0]],env=env,check=True,timeout=15)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    watch['forced_cleanup']=True;os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            except BaseException as close:watch['owned_cleanup_error']=repr(close)
            env['GLM_R404_POST_DEADLINE_UNIX']=str(min(time.time()+90,watch.get('original_GPU_start_unix',time.time())+5880))
            try:watch['post_after_failure']=idle('failure-post')
            except BaseException as close:watch['post_error']=repr(close)
        watch.update(completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY')
    finally:save()
    print(json.dumps({k:watch.get(k) for k in ['completed','passed','actual_GPU','CPU_keeper','state']}))
    return 0 if watch['passed']else 1

if __name__=='__main__':raise SystemExit(main())
