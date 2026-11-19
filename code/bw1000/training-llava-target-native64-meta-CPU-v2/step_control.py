"""Actual HOST task controller: dedicated step charged-memory budget and held tree."""
import json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from common import P,RUN,save,read,need,sha
from bounded import identity,device_fds
from held_child import cleanup_held
from step_memory import StepMemory,LIMIT
from step_tree import OwnedTree

def publish(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<=65536 and not path.exists(),'UNIQUE_BOUNDED_RELEASE')
    tmp=path.with_suffix('.pending')
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def run(argv,env,outdir,epoch,work,stop,held_deadline,registry_deadline):
    need(epoch<=time.time()<work<stop<held_deadline<registry_deadline<=epoch+282,'ORIGINAL_STEP_CONTROL_BUDGET')
    outdir.mkdir(mode=0o700);owner=identity();save(outdir/'owner.json',owner)
    need(device_fds(owner['PID'])==[],'HOST_CONTROLLER_DEVICE_FDS')
    child=None;anchor=None;tree=None;memory=None;failure=None;held=None;post=None;errors={};released=False
    phase='qualify_numeric_step_memory';mono=time.monotonic()+stop-time.time()
    records=[];record_count=0;privileged=set()
    resources=dict(metric='STEP_CGROUP_CHARGED_MEMORY',samples=0,peak_charged_memory_bytes=None,
                   soft_limit_bytes=LIMIT,soft_limit_exceeded=False,hard_limit=False,last_sample=None,
                   RSS_group_bytes=None,RSS_coverage='UNKNOWN_NOT_READ',charged_memory_is_sum_RSS=False,
                   charged_samples=[])
    def persist(row):
        nonlocal record_count
        record_count+=1;records.append(row)
        if len(records)>128:del records[:-128]
        if row.get('UID_classification')=='ANCHORED_359_PRIVILEGED_INTERMEDIARY':
            privileged.add((row['PID'],row['before']['startticks']))
        save(outdir/'identity-observations.json',dict(owner=owner,total_records=record_count,
             retained_records=records,truncated_records=max(0,record_count-len(records)),
             privileged_identities=[dict(PID=p,startticks=t,RSS_bytes=None) for p,t in sorted(privileged)],
             RSS_coverage='UNKNOWN_NOT_READ',global_proc_scan=False),replace=True)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    def interrupted(*_):raise TimeoutError('ORIGINAL_STEP_CONTROL_STOP')
    for sig in handlers:signal.signal(sig,interrupted)
    def account(sample):
        need(len(resources['charged_samples'])<1200,'CHARGED_MEMORY_SAMPLE_BOUND')
        resources['samples']+=1
        resources['peak_charged_memory_bytes']=max(resources['peak_charged_memory_bytes'] or 0,sample['charged_memory_bytes'])
        resources['soft_limit_exceeded']|=sample['soft_limit_exceeded'];resources['last_sample']=sample
        resources['charged_samples'].append({k:sample[k] for k in ('observed_monotonic','charged_memory_bytes',
            'observed_limit','events','memory_stat_SHA256','member_PIDs','soft_limit_exceeded')})
        save(outdir/'resources.json',resources,replace=True)
        need(not sample['soft_limit_exceeded'],'STEP_CHARGED_MEMORY8G_SOFT_STOP')
    try:
        memory=StepMemory(P['parent'],os.environ.get('SLURM_STEP_ID',''),stop,mono)
        initial=memory.sample()
        need(initial['member_PIDs']==[os.getpid()],'STEP_ALREADY_CONTAINS_UNRELATED_PROCESS_NO_NATIVE')
        account(initial)
        save(RUN/'cgroup-admission.json',dict(qualified=True,**memory.proof,epoch=epoch,deadline=epoch+300,
             initial_sample=initial,scope_membership_exclusive_to_controller_before_Popen=True))
        with (outdir/'stdout.private').open('xb') as out,(outdir/'stderr.private').open('xb') as err:
            phase='Popen_held_frontend'
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner)
            save(outdir/'held-child-anchor.json',anchor)
            tree=OwnedTree(child,memory,persist)
            while child.poll() is None:
                phase='whole_owned_tree_and_charged_memory'
                seen=tree.discover();sample=memory.sample()
                # A new own descendant can appear between tree and cgroup reads.
                # At most two extra lineage traversals, never admit an unexplained PID.
                for retry in range(2):
                    if set(sample['member_PIDs'])<=(seen|{os.getpid()}):break
                    seen=tree.discover();sample=memory.sample()
                need(set(sample['member_PIDs'])<=(seen|{os.getpid()}),'CGROUP_MEMBER_NOT_IN_HELD_TREE')
                # Validate every live registered identity, including any reparented/hidden member.
                tree_post=tree.post()
                need(not tree_post['unknown_owned_liveness'],'KNOWN_TREE_LIVENESS_UNKNOWN')
                need(not any(r['PID'] in sample['member_PIDs'] and r['state']=='ORIGINAL_ABSENT_PID_REUSED'
                             for r in tree_post['records']),'CGROUP_MEMBER_PID_REUSED')
                need(set(tree_post['remaining_known_owned'])<=set(sample['member_PIDs']),
                     'KNOWN_LIVE_OWNED_PROCESS_ESCAPED_MEMORY_SCOPE')
                account(sample)
                save(outdir/'owned-registry.json',tree.registry,replace=True)
                if not (outdir/'child-owner.json').exists():save(outdir/'child-owner.json',tree.registry[child.pid])
                if not released and (RUN/'native-ready.json').exists():
                    ready=read(RUN/'native-ready.json');w=ready['owner'];saved=tree.registry.get(w['PID'])
                    need(saved is not None and all(w[k]==saved[k] for k in ('PID','PPID','PGID','startticks')),
                         'READY_WORKER_NOT_IN_EXACT_HELD_TREE')
                    need(w['UID']==owner['UID'] and w['host']==owner['host'] and w['boot_id']==owner['boot_id']
                         and ready['epoch']==epoch and ready['deadline']==epoch+300 and ready['work_deadline']==work
                         and ready['native_imports']==0 and ready['Torch_imported'] is False,'READY_CONTEXT_CHANGED')
                    need(w['PID'] in sample['member_PIDs'] and time.time()<work and time.monotonic()<mono,'READY_MEMORY_OR_DEADLINE')
                    phase='persist_native_release'
                    release=dict(owner=w,epoch=epoch,deadline=epoch+300,work_deadline=work,
                        native_release_authorized=True,memory_scope_qualified=True,
                        memory_scope=memory.proof,sample_monotonic=sample['observed_monotonic'],
                        request_id=read(RUN/'intent.json')['request_id'])
                    publish(RUN/'native-release.json',release);released=True
                need(all((outdir/n).stat().st_size<=16*1024**2 for n in ('stdout.private','stderr.private')),'BOUNDED_CONTAINER_LOG')
                left=min(stop-time.time(),mono-time.monotonic());need(left>0,'ORIGINAL_MEMORY_CONTROLLER_DEADLINE')
                try:child.wait(timeout=min(.25,left))
                except subprocess.TimeoutExpired:pass
            phase='native_child_returned'
    except BaseException as e:
        frames=traceback.extract_tb(e.__traceback__)
        failure=dict(error_type=type(e).__name__,reason=str(e)[:256],phase=phase,
                     source_linecode=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in frames[-4:]])
        if not (RUN/'cgroup-admission.json').exists():
            save(RUN/'cgroup-admission.json',dict(qualified=False,state='NOT_READY',epoch=epoch,deadline=epoch+300,
                 failure=failure,native_release_authorized=False,memory_scope=memory.proof if memory is not None else None))
    finally:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:held=cleanup_held(child,anchor,held_deadline)
        except BaseException as e:
            held=dict(reaped=child is None or child.returncode is not None,termination_needed=True,error_type=type(e).__name__)
            errors['held_cleanup']=type(e).__name__
        try:
            known=tree.post() if tree is not None else dict(remaining_known_owned=[],unknown_owned_liveness=[],records=[],privileged_members_included=True)
            # Use cleanup reserve; no fresh epoch. Cgroup must now contain only this task controller.
            if memory is not None:
                memory.deadline=registry_deadline;memory.end=mono+(registry_deadline-stop)
                final=memory.sample();members=final['member_PIDs']
            else:final=None;members=None
            complete=members==[os.getpid()] and not known['remaining_known_owned'] and not known['unknown_owned_liveness']
            post=dict(same_host=socket.gethostname()==owner['host'],same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'],
                held_child_reaped=held.get('reaped'),controller_device_FDs=device_fds(os.getpid()),
                owned_descendants_remaining=0 if complete else None,owned_device_FD_holders={} if complete else None,
                known_tree=known,cgroup_final=final,step_only_controller_remains=complete,
                remaining_count_covers_privileged_members=True)
        except BaseException as e:post=dict(state='UNKNOWN',error_type=type(e).__name__);errors['post']=type(e).__name__
        if memory is not None:memory.close()
        for sig,h in handlers.items():signal.signal(sig,h)
    rc=child.returncode if child else None
    passed=bool(released and rc==0 and failure is None and not errors and held.get('reaped')
                and not held.get('termination_needed') and post.get('step_only_controller_remains')
                and post.get('same_host') and post.get('same_boot') and post.get('controller_device_FDs')==[])
    remaining=post.get('owned_descendants_remaining')
    value=dict(passed=passed,returncode=rc,owner=owner,child=anchor,failure=failure,resources=resources,
        native_released=released,admission_state='RELEASED' if released else 'NOT_READY',
        held_cleanup=held,registry_cleanup=dict(owned_processes_remaining=remaining,
            termination_needed=held.get('termination_needed'),all_UID_known_tree_covered=post.get('step_only_controller_remains',False)),
        cleanup_errors=errors,post=post,stop=stop,held_deadline=held_deadline,registry_deadline=registry_deadline,
        RSS_unknown_not_zero=True,charged_memory_is_sum_RSS=False,implicit_library_device_queries='UNKNOWN')
    save(outdir/'resources.json',resources,replace=True);save(outdir/'outcome.json',value)
    return value
