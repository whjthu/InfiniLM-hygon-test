"""Same unique constructor handle through terminal/audit/unconditional post. No replay."""
import datetime,hashlib,json,pathlib,re,subprocess,sys,time,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
out=r/'results/GLM-PP3-constructor-watch-r8781.json';assert not out.exists()
launch=json.loads((r/'GLM-PP3-constructor-r8781-launch.json').read_text());run=launch['run']
manifest=r/'incoming/GLM-PP3-constructor-manifest-r8781.json'
state={'state':'watching','run':run,'manifest_SHA256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'restart':False,'GPU_submission':False}
nodes=['a02r4n12','a02r4n13','a02r4n14']
def save():
    state['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=out.with_suffix('.tmp')
    p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out)
def terminal(job):
    state['actual_step']=job;save()
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            paths=list((r/'results').glob(f'GLM-PP3-constructor-r8781-{run}-stage*-rank*.json'))
            if len(paths)==24:break
            time.sleep(1)
        audit=subprocess.run([sys.executable,str(r/'incoming/audit_GLM_PP3_constructor_r8781.py')],text=True,capture_output=True,timeout=60)
        state['audit_returncode']=audit.returncode;state['audit_stderr']=audit.stderr[-12000:]
        if audit.returncode==0:
            data=json.loads(audit.stdout);target=r/'results/GLM-PP3-constructor-audit-r8781.json'
            with target.open('x') as f:json.dump(data,f,indent=2)
            state['audit_SHA256']=hashlib.sha256(target.read_bytes()).hexdigest()
            state['all24_normal']=data['all24_normal'];state['capacity_screen']=data['all24_estimated_resident_plus6GiB_capacity_screen']
        logs=[]
        for stage in range(3):
            p=r/'logs'/f'GLM-PP3-constructor-r8781-{job[0]}-{stage}.log';text=p.read_text(errors='replace') if p.exists() else ''
            logs.append({'file':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,
                'normal_markers':len(re.findall(r'^GLM_PP3_CONSTRUCTOR_NORMAL ',text,re.M)),
                'errors':[line for line in text.splitlines() if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|NCCL WARN|HIP error|CUDA error',line)],
                'actual_IBext_v8':bool(re.search(r'Using network IBext_v8\b',text)),
                'Socket_fallback':bool(re.search(r'Using network Socket\b',text))})
        state['logs']=logs
    except BaseException as exc:state.update(audit_error=repr(exc),audit_traceback=traceback.format_exc())
    finally:
        try:
            args=['srun','--jobid=910376','--overlap','--exact','-N3','-n3','--ntasks-per-node=1','-c1','-w',','.join(nodes),'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-PP3-constructor-post-r8781','python3',str(r/'incoming/verify_gpu_idle_r5819.py')]
            p=subprocess.run(args,text=True,capture_output=True,timeout=150)
            state['post_returncode']=p.returncode;state['post_stdout']=p.stdout;state['post_stderr']=p.stderr
            rows=[json.loads(v) for v in p.stdout.splitlines() if v.startswith('{')]
            state['post_passed']=p.returncode==0 and sorted(v['node'] for v in rows)==nodes and all(v['kfd_pids']==[] for v in rows)
        except BaseException as exc:state.update(post_passed=False,post_error=repr(exc))
        finally:save()
    state['qualified_constructor_only']=bool(state.get('all24_normal') and state.get('post_passed') and job[2:4]==['COMPLETED','0:0'] and all(v['normal_markers']==8 and not v['errors'] and v['actual_IBext_v8'] and not v['Socket_fallback'] for v in state.get('logs',[])))
    state['state']='terminal';save()
save();start=time.monotonic()
while time.monotonic()-start<900:
    try:
        rows=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
        jobs=[v.split('|') for v in rows.splitlines() if v.split('|')[1]=='GLM-PP3-constructor-r8781' and v.split('|')[0].split('.')[-1].isdigit()]
        assert len(jobs)<=1,('duplicate unique constructor',jobs)
        if jobs:
            if state.get('actual_handle') is None:state['actual_handle']=jobs[0][0]
            assert state['actual_handle']==jobs[0][0];state['actual_step']=jobs[0];save()
            if jobs[0][2].split('+')[0].split()[0] in {'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','DEADLINE','BOOT_FAIL','REVOKED'}:
                terminal(jobs[0]);break
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:state['accounting_transient_error']=repr(exc);save()
    time.sleep(5)
else:state['state']='UNKNOWN_bound_expired_no_replay';save()
