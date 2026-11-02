"""Bounded386 lifecycle specialized to one16card unloaded constructor group."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
S=R/'incoming/GLM-C1-PP2-constructor-r395'
LABEL='GLM-C1-PP2-constructor-r395';RUN='constructor-20261102-a';NODES=['a02r4n12','a02r4n13']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def publish(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def verify(root,source,digest):
    mpath=source/'GPU_MANIFEST.json';assert sha(mpath)==digest;m=json.loads(mpath.read_bytes())
    assert m['GPU_GO'] is True and m['Model_GO'] is False and m['GPU_max_s']==300
    assert m['source_manifest_SHA256']==sha(source/'SOURCE_MANIFEST.json')
    original=json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
    for name,h in original['files_SHA256'].items():assert sha(source/name)==h,name
    for name,h in m['controller_files_SHA256'].items():assert sha(source/'controller'/name)==h,name
    receipt=root/'candidates/GLM-C1-PP2-constructor-r395/CPU-result.json';assert sha(receipt)==m['CPU_receipt_SHA256']
    q=json.loads(receipt.read_bytes());assert q['completed'] and q['passed'] and q['GPU']==0 and q['Native_dlopen']is False
    assert q['source_manifest_SHA256']==m['source_manifest_SHA256'] and q['config_SHA256']==m['config_SHA256']
    assert sha(root/'incoming/resource_wave_local_r3097_outer.sh')==m['outer_wrapper_SHA256']
    return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publication-commit',required=True);parser.add_argument('--manifest-sha',required=True);args=parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.publication_commit)
    assert os.environ['SLURM_JOB_ID']=='910407' and socket.gethostname().split('.')[0]=='a14r2n20'
    m=verify(R,S,args.manifest_sha);cleanup_deadline=time.monotonic()+580
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_')) or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    W=R/'results'/(LABEL+'-'+RUN+'-watch.json');cancel=R/'control'/(LABEL+'-'+RUN+'-cancel.json');go=R/'control'/(LABEL+'-'+RUN+'-GO.json')
    watch=dict(completed=False,passed=False,run=RUN,publication_commit=args.publication_commit,GPU_manifest_SHA256=args.manifest_sha,
               CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,Model_loaded=False,checkpoint_IO=False,compile=False,
               GPU_max_s=300,CPU_keeper_max_s=600,retry=False,reload=False,forced_cleanup=False)
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
        cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(S/'controller/idle.py')]
        p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=min(90,max(1,cleanup_deadline-time.monotonic())))
        value=[json.loads(s) for s in p.stdout.splitlines() if s.startswith('{')];assert p.returncode==0 and {v['node'] for v in value}==set(NODES) and len(value)==2
        assert all(v['CPU_only'] and not v['GPU_initialized'] and v['KFD_available'] and not v['kfd_pids'] and not v['matching_component_processes'] and v['ports_bindable']=={'43843':True} for v in value)
        assert len({v['actual_step'] for v in value})==1
        return value
    try:
        raw=command(['scontrol','show','job','910376','-o']);assert 'JobState=RUNNING'in raw and 'UserId=aclsylqidf('in raw
        assert set(NODES)<=set(expand(re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)))
        assert exact()is None;dedup();watch['fresh_before']=idle('before');dedup([r['actual_step'] for r in watch['fresh_before']]);save()
        epoch=time.time();deadline=epoch+300;watch.update(original_GPU_start_unix=epoch,GPU_deadline_unix=deadline)
        publish(go,dict(run=RUN,GPU_GO=True,Model_GO=False,operation='constructor_only_no_weights_finish_forward',source_manifest_SHA256=m['source_manifest_SHA256'],CPU_receipt_SHA256=m['CPU_receipt_SHA256'],Native_SHA256=m['Native_SHA256'],config_SHA256=m['config_SHA256'],TP=8,PP=2,rows=1,capacity=32768,GPU_max_s=300,original_GPU_start_unix=epoch,publication_commit=args.publication_commit))
        inside='GLM-C1-PP2-constructor-r395/GPU_inside.sh';env.update(WAVE_INSIDE_R2323=inside,GLM_R395_GO='/refactor/control/'+go.name,GLM_R395_DEADLINE_UNIX=str(deadline),GLM_R395_CANCEL_FILE='/refactor/control/'+cancel.name)
        cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','00:05:00','--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
        with(R/'logs'/(LABEL+'-'+RUN+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        watch.update(GPU_submitted=True,srun_PID=proc.pid);save()
        while time.time()<deadline:
            row=exact()
            if row:
                actual=row[0];watch.update(actual_GPU=actual,last_Slurm=row);save()
                if row[2].split()[0].rstrip('+')in TERMINAL:break
            time.sleep(min(5,max(0,deadline-time.time())))
        else:raise TimeoutError('Original300s deadline;neverrestart')
        watch['srun_exit_code']=proc.wait(timeout=min(25,max(1,cleanup_deadline-time.monotonic())))
        assert row[2]=='COMPLETED' and row[3]=='0:0' and watch['srun_exit_code']==0,'Originalfailedstep retained'
        watch['post']=idle('post')
        receipts=[];until=min(time.monotonic()+90,cleanup_deadline-5)
        for stage in range(2):
            masks=[]
            for rank in range(8):
                path=R/'results'/('GLM-C1-PP2-constructor-r395-'+RUN+'-stage%d-rank%d.json'%(stage,rank))
                while True:
                    try:q=json.loads(path.read_bytes())
                    except (FileNotFoundError,json.JSONDecodeError):q=None
                    if q and q.get('completed'):break
                    if time.monotonic()>=until:raise TimeoutError('Originalreportvisibility;no rerun')
                    time.sleep(.5)
                assert q['normal_destroy'] and q['normal_shutdown'] and q['actual_step']==actual and q['source_manifest_SHA256']==m['source_manifest_SHA256']
                assert q['Native_SHA256']==m['Native_SHA256'] and q['rows']==1 and q['capacity']==32768 and q['PP']==2 and q['TP']==8
                assert q['stage']==stage and q['rank']==rank and q['world16_identity_sum']==16 and not q['weights_IO'] and q['finish_calls']==q['forward_calls']==0
                assert q['Model_qualified']is False and q['loaded_service_admission_passed']is False
                masks+=q['owner_affinity']['CPUs'];receipts.append(dict(stage=stage,rank=rank,path=str(path),SHA256=sha(path),normal=True))
            assert len(masks)==len(set(masks))==32
        logs=sorted((R/'logs').glob(LABEL+'-'+RUN+'-'+actual+'-*.log'));assert len(logs)==2
        assert sorted(marker for p in logs for marker in re.findall(r'GLM_C1_PP2_R395_NORMAL ([01])',p.read_text(errors='replace')))==['0','1']
        watch.update(completed=True,passed=True,normal16owners=True,mp_spawn_join_markers=True,reports=receipts,
                     log_refs=[dict(path=str(p),SHA256=sha(p)) for p in logs],state='CONSTRUCTOR_ONLY_NORMAL_NOT_LOADED_SERVICE_FIT')
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
            try:watch['post_after_failure']=idle('failure-post')
            except BaseException as close:watch['post_error']=repr(close)
        watch.update(completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY')
    finally:save()
    print(json.dumps({k:watch.get(k) for k in ['completed','passed','actual_GPU','CPU_keeper','state']}))
    return 0 if watch['passed']else 1

if __name__=='__main__':raise SystemExit(main())
