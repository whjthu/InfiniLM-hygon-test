"""Single original900 owner: passive admission, three held native processes, cleanup, post."""
import json,os,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
from common import D,P,need,obj,save,left,pinned,verify_source,native_inputs,own,sha
from held_child import cleanup_held
from owned_processes import snapshot,refresh,cleanup
from qualified_supervisor import verify_model,sample_group_rss
from gpu_admission import snapshot as gpu_snapshot,query
from resource_guard import acquire_lease,host_memory,runtime_foreign_processes

def main(c):
    verify_source();need(P['root_GO'] is P['GPU_GO'] is P['READ_GO'] is True,'ROOT_ADOPTED_ONLY');need(c['deadline']==c['epoch']+900,'ONE_ORIGIN');need(time.time()<c['epoch']+100,'ADMISSION_BUDGET');scope=Path(P['namespace']);need(scope==D.parent and scope.stat().st_uid==os.getuid(),'FIXED_NEW_SOURCE_SCOPE')
    required=set(P['CPU_affinity']);need(required<=os.sched_getaffinity(0),'FIXED_FOUR_CPUS');os.sched_setaffinity(0,required);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));owner=own();save(scope/'controller-owner.json',dict(c,owner=owner));child=None;anchor=None;registry={};lease=None;results=[];phase_windows=[];GPU_samples=[];failure=None;post=None;cleanup_results=[];started=time.monotonic();phase='admission';resources=dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_bytes=256*1024**3,soft_limit_exceeded=False,hard_limit=False,last_sample=None)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('ORIGINAL810_SUPERVISOR')));signal.setitimer(signal.ITIMER_REAL,left(c,810))
    try:
        lease,leaseproof=acquire_lease(P,owner,c['request_id']);save(scope/'lease.json',leaseproof);need(host_memory()['MemAvailable_bytes']>=P['host_MemAvailable_admission_bytes'],'HOST_AVAILABLE32G')
        before=gpu_snapshot(P['expected_GPU'],P['master_port'],c['epoch']+100,owner);save(scope/'before.json',before)
        fm=obj(D/'FRAMEWORK_BINDING.json')['files'];need(len(fm)==404,'EXACT404')
        for n,p in fm.items():left(c,100);pinned(Path(P['framework'])/n,p)
        for rec in P['providers']:left(c,100);pinned(rec['canonical_path'],rec)
        import importlib.metadata as md
        need({k:md.version(k) for k in P['versions']}==P['versions'],'EXACT_QUALIFIED_ENV')
        binding,native,data=native_inputs(c);model=verify_model(P,lambda _:left(c,100));save(scope/'assets-admission.json',model)
        need(shutil.disk_usage(scope).free>=P['private_disk_required_bytes'],'PRIVATE_DISK40G');save(scope/'input-admission.json',dict(databinding=P['databinding'],native_identities=P['native_identities'],rows=72,train_file=binding['files']['train-swift.jsonl'],train_images=12,dev_blind_opened=False,max_native_length=max(r['full_length'] for r in native)))
        for item in P['phases']:
            if results:save(scope/('before-'+item['name']+'.json'),gpu_snapshot(P['expected_GPU'],P['master_port'],c['epoch']+790,owner))
            phase=item['name'];need(left(c,790)>60,'NO_PHASE_WITHOUT_NATIVE_RESERVE');root=scope/phase;root.mkdir(mode=0o700);config=dict(c,phase=phase,phase_root=str(root),train_output=str(root/'train'),continuous_root=str(scope/'continuous'),checkpoint20=str(scope/'continuous/train/checkpoint-20'),supervisor=owner);path=root/'config.json';save(path,config)
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=P['expected_GPU']['UUID'],PYTHONPATH=str(D)+':'+P['framework'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            launch_epoch=time.time()
            with (root/'stdout.private').open('xb') as out,(root/'stderr.private').open('xb') as err:
                child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(D/'native_entry.py'),'--config',str(path)],env=env,stdout=out,stderr=err,start_new_session=True);anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner,request_id=c['request_id']);save(root/'held-anchor.json',dict(anchor,launch_epoch=launch_epoch,planned=item));row=snapshot().get(child.pid)
                if row:registry[child.pid]={k:row[k] for k in ['PID','PPID','PGID','startticks','state']}
                last=0.
                while child.poll() is None:
                    left(c,790);refresh(registry,scope);need(sample_group_rss(registry,resources),'QUALIFIED_RSS_PARSER');need(resources['sampled_group_peak_RSS_bytes']<=P['RSS_sampled_soft_bytes'],'SAMPLED64G_STOP')
                    now=time.monotonic()
                    if now-last>=5:
                        need(host_memory()['MemAvailable_bytes']>=P['host_MemAvailable_runtime_stop_bytes'],'HOST_AVAILABLE8G');foreign=runtime_foreign_processes(P,registry,c['epoch']+790);need(not foreign['foreign_selected_GPU_PIDs'],'FOREIGN_GPU_STOP_ONLY_OWN');sample=query('index,uuid,utilization.gpu,memory.used','gpu',c['epoch']+790);selected=[x for x in sample if len(x)==4 and x[1].strip()==P['expected_GPU']['UUID']];need(len(selected)==1,'SELECTED_GPU_SAMPLE');GPU_samples.append(dict(epoch=time.time(),phase=phase,index=int(selected[0][0]),GPU_util_percent=int(selected[0][2]),memory_used_MiB=int(selected[0][3])));save(scope/'GPU-samples.json',GPU_samples,True);save(scope/'progress.json',dict(phase=phase,completed_phases=len(results),planned_updates=32,planned_microbatches=128,resources=resources),True);last=now
                    try:child.wait(timeout=.25)
                    except subprocess.TimeoutExpired:pass
            held=cleanup_held(child,anchor,c['epoch']+825);tree=cleanup(registry,scope,grace=max(0,min(1,(c['epoch']+835-time.time())/2)));cleanup_results.append(dict(phase=phase,held=held,tree=tree));phase_windows.append(dict(phase=phase,launch_epoch=launch_epoch,reaped_epoch=time.time(),native_returncode=child.returncode,window_is_process_lifecycle_not_busy=True));need(child.returncode==0 and held['reaped'] and not held['termination_needed'] and tree['owned_processes_remaining']==0 and not tree['termination_needed'],'NORMAL_PHASE_TERMINAL')
            receipt=obj(root/'receipt.json');need(all(receipt[k]==c[k] for k in ['request_id','publication','epoch','deadline']) and receipt['phase']==phase and receipt['passed'] is True and receipt['updates']==item['updates'] and receipt['microbatches']==item['microbatches'],'FULL_PHASE_RECEIPT');results.append(receipt);save(scope/'phase-results.json',results,True);child=None;anchor=None;registry={}
        need(sum(x['updates'] for x in results)==32 and sum(x['microbatches'] for x in results)==128,'COMPLETE32_128')
    except BaseException as e:failure=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()),phase=phase)
    finally:
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));remaining=c['epoch']+845-time.time()
        if remaining<=0:os._exit(124)
        signal.setitimer(signal.ITIMER_REAL,remaining);signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        # Two independent attempts retain failures; no owned-PID failure removes the other.
        tail={}
        try:tail['held']=cleanup_held(child,anchor,c['epoch']+825)
        except BaseException as e:tail['held_error']=type(e).__name__
        try:tail['tree']=cleanup(registry,scope,grace=max(0,min(1,(c['epoch']+835-time.time())/2)))
        except BaseException as e:tail['tree_error']=type(e).__name__
        save(scope/'cleanup.json',dict(phases=cleanup_results,tail=tail))
        try:post=gpu_snapshot(P['expected_GPU'],P['master_port'],min(c['epoch']+845,c['epoch']+855),owner);save(scope/'post.json',post)
        except BaseException as e:
            if failure is None:failure=dict(type=type(e).__name__,phase='post')
        passed=bool(failure is None and len(results)==3 and post and post['scope_admission_satisfied'] is True and tail.get('tree',{}).get('owned_processes_remaining')==0 and tail.get('held',{}).get('reaped') is True and not any(k.endswith('_error') for k in tail))
        partial={}
        for item in P['phases']:
            pp=scope/item['name']/'progress.json'
            partial[item['name']]=obj(pp) if pp.is_file() and pp.stat().st_size<=262144 else dict(completed_updates=0,completed_microbatches=0,not_observed=True)
        save(scope/'phase-progress-final.json',partial)
        outcome=dict(c,passed=passed,owner=owner,phases=results,planned_updates=32,planned_microbatches=128,completed_updates=sum(x['completed_updates'] for x in partial.values()),completed_microbatches=sum(x['completed_microbatches'] for x in partial.values()),phase_progress=partial,phase_windows=phase_windows,GPU_samples=len(GPU_samples),sampled_GPU_busy_observed=any(x['GPU_util_percent']>0 for x in GPU_samples),failure=failure,cleanup=dict(phases=cleanup_results,tail=tail),post=post,resources=resources,new_RSS_limit_bytes=P['RSS_sampled_soft_bytes'],RSS_hard_limit=False,GPU_sampling_scope='Selected GPU sampled every>=5s; lifecycle windows separately reported; no allocated-as-busy or sustained640 claim',finish_epoch=time.time(),whole_seconds=time.monotonic()-started,Model_GO=False,Hygon_GPU_GO=False,formal17=False,old_domain_FAIL_preserved=True)
        save(scope/'outcome.json',outcome)
        if lease is not None:os.close(lease)
        signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(passed=passed,request_id=c['request_id'],completed_updates=outcome['completed_updates'],completed_microbatches=outcome['completed_microbatches'])));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main(obj(D.parent/'intent.json')))
