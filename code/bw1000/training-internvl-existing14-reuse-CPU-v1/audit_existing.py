"""One new exact14 target SHA audit; no transfer, hardlink, tensor or old writes.

Root injects published PLAN/CLAIM/CONTEXT after its ordinary once/publication gate.
This draft cannot run with the source-prepared false GO flags.
"""
import hashlib,json,math,os,re,resource,signal,socket,stat,time
from pathlib import Path,PurePosixPath

def need(ok,why):
    if not ok:raise ValueError(why)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def canonical(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()

def stamp(value):
    return {k:getattr(value,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid','st_mode','st_nlink')}

def no_links(path):
    need(path.is_absolute() and '..' not in path.parts,'EXACT_ABSOLUTE_PATH')
    for p in [*reversed(path.parents),path]:need(not p.is_symlink(),'SYMLINK_NOT_AUTHORIZED')
    return path

def save(path,value,replace=False):
    raw=canonical(value);need(len(raw)<=256*1024,'BOUNDED_AUDIT_METADATA')
    target=path.with_suffix('.tmp') if replace else path
    with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if replace:os.replace(target,path)

def snapshot(path,expected=None):
    no_links(path)
    try:s=path.lstat()
    except FileNotFoundError:return dict(exists=False,qualified=False)
    need(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid(),'EXACT_TARGET_REGULAR_OWNER')
    return dict(exists=True,stat=stamp(s),expected_size_matches=expected is None or s.st_size==expected,
                whole_SHA_qualified=False)

def main(context,plan,claim):
    started=time.monotonic();epoch=context['epoch']
    need(claim['root_GO'] is claim['CPU_GO'] is claim['READ_GO'] is True and claim['max_invocations']==1,
         'ROOT_ADOPTED_ONCE_CPU_READ_REQUIRED')
    need(all(claim[k] is False for k in ('provision_GO','GPU_GO','Model_GO','training_GO')),'READONLY_STAGE_NO_PROVISION')
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<=time.time()-epoch<plan['work_seconds'],
         'SINGLE_ORIGINAL300_EPOCH')
    need(context['deadline']==epoch+300 and re.fullmatch('[0-9a-f]{32}',context['request_id']) and re.fullmatch('[0-9a-f]{40}',context['publication']),
         'NEW_REQUEST_PUBLICATION_CONTEXT')
    need(context['PLAN_canonical_SHA256']==sha(canonical(plan)) and context['CLAIM_canonical_SHA256']==sha(canonical(claim)),
         'PUBLISHED_CANONICAL_INJECTED_BINDING')
    need(plan['hash_file_count']==14 and plan['hash_bytes']==4922323586 and len(plan['files'])==26,
         'FROZEN26_AND14_DENOMINATORS')
    need(plan['original_receiver_exit']['original_process_terminal'] is True and plan['prior_outcome']['passed'] is False,
         'ORIGINAL_TERMINAL_FAILURE_NOT_UNKNOWN_REPLAY')
    root=no_links(Path(plan['new_namespace']));old=no_links(Path(plan['old_namespace']))
    need(root.name==plan['label']==claim['label'] and root.parent==old.parent and root!=old,
         'NEW_EXCLUSIVE_AUDIT_NAMESPACE')
    need(old.is_dir() and old.stat().st_uid==os.getuid()==plan['original_receiver_exit']['UID'],
         'CURRENT_TARGET_NAMESPACE_OWNER')
    need(root.is_dir() and root.stat().st_uid==os.getuid() and Path(__file__).resolve().parent==root/'source','EXACT_ALREADY_INSTALLED_SOURCE_NAMESPACE')
    need(os.environ.get('SLURM_JOB_ID')==plan['parent'] and os.environ.get('SLURM_STEP_ID','').isdigit()
         and os.uname().nodename==plan['node'] and os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)'), 'ACTUAL_HASH_CPU_STEP_CONTEXT')
    work=context['work_deadline'];step_end=context['step_end']
    need(time.time()<work<=min(epoch+270,step_end-10) and step_end<=epoch+270,'SHRINKING_ORIGINAL_STEP_WORK')
    root=root/'audit';root.mkdir(mode=0o700)
    mono=time.monotonic()+work-time.time()
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.001,min(step_end-2,epoch+plan['hard_stop_seconds'])-time.time()))
    def check():need(time.time()<work and time.monotonic()<mono,'ORIGINAL_AUDIT_WORK_DEADLINE')
    def stop(*_):raise TimeoutError('ORIGINAL_AUDIT_STOP')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    owner=dict(PID=os.getpid(),startticks=int(fields[19]),PPID=int(fields[1]),PGID=int(fields[2]),
               host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UID=os.getuid())
    save(root/'intent.json',dict(context,owner=owner,planned_inventory=26,planned_hash_files=14,
                               planned_hash_bytes=plan['hash_bytes'],old_namespace=str(old),payload_writes=0))
    resources={};inventory=[];completed=[];attempted=0;read_bytes=0;error=None;active=None;phase='resources'
    def progress():
        save(root/'progress.json',dict(context,owner=owner,phase=phase,active_member=active,
             attempted_hash_files=attempted,completed_hash_files=len(completed),planned_hash_files=14,
             actual_hash_bytes=read_bytes,planned_hash_bytes=plan['hash_bytes']),replace=True)
    try:
        cpus=sorted(os.sched_getaffinity(0)-{0,1});need(len(cpus)>=2,'TWO_CPU_AFFINITY');os.sched_setaffinity(0,cpus[:2])
        before= os.getpriority(os.PRIO_PROCESS,0);target=max(before,10);need(target<=19,'NICE_RANGE')
        if target>before:os.nice(target-before)
        actual=os.getpriority(os.PRIO_PROCESS,0);need(target<=actual<=19,'NICE_NOT_RAISED_PRIORITY')
        raw=Path('/proc/self/status').read_bytes();need(len(raw)<=65536,'SELF_STATUS_BOUND')
        mem={}
        for line in raw.splitlines():
            k,sep,v=line.partition(b':')
            if k in (b'VmSize',b'VmRSS'):
                parts=v.split();need(len(parts)==2 and parts[0].isdigit() and parts[1]==b'kB','SELF_MEMORY_SCHEMA')
                mem[k.decode()]=int(parts[0])*1024
        need(set(mem)=={'VmSize','VmRSS'},'SELF_MEMORY_BASELINE')
        address_limit=max(512*1024**2,mem['VmSize']+256*1024**2)
        need(address_limit<=plan['AS_max_bytes']==4*1024**3,'DYNAMIC_ADDRESS_SPACE_CAP')
        oldsoft,oldhard=resource.getrlimit(resource.RLIMIT_AS)
        need(oldhard==resource.RLIM_INFINITY or oldhard>=address_limit,'INHERITED_AS_HARD_LIMIT_TOO_LOW')
        resources=dict(CPU_affinity=cpus[:2],nice_before=before,nice_target=target,nice_actual=actual,
                       initial_process_memory=mem,AS_limit_bytes=address_limit,AS_is_not_RSS=True,allocated_CPU=4,worker_CPU=2,
                       remaining_hash_envelope_seconds=work-time.time(),necessary_mean_logical_bytes_per_second=plan['hash_bytes']/(work-time.time()))
        save(root/'resources.json',resources)
        resource.setrlimit(resource.RLIMIT_AS,(address_limit,address_limit))
        phase='exact26_final_and_one_partial_metadata'
        byname={}
        for item in plan['files']:
            check();name=item['name'];part=PurePosixPath(name)
            need(not part.is_absolute() and '..' not in part.parts and part.as_posix()==name and name not in byname,
                 'EXACT_MEMBER_PATH')
            byname[name]=item;observed=snapshot(old/name,item['bytes'])
            prior=plan['old_saved_stat_reference'].get(name)
            observed['old_four_fields_match']=(all(observed['stat'][k]==prior[k] for k in ('st_ino','st_size','st_mtime_ns','st_ctime_ns')) if prior and observed['exists'] else None)
            observed['old_four_fields_are_current_SHA_qualification']=False
            inventory.append(dict(name=name,**observed))
        partial=dict(name=plan['partial_name'],**snapshot(old/plan['partial_name']))
        save(root/'inventory.json',dict(context,owner=owner,final_files=inventory,partial=partial,
             cross_host_saved_st_dev_not_compared=True,old_receipts_reread=False))
        need(len(byname)==26 and len(plan['hash_names'])==len(set(plan['hash_names']))==14,'EXACT_UNIQUE14')
        need(sum(byname[n]['bytes'] for n in plan['hash_names'])==plan['hash_bytes'],'EXACT_READ_BYTE_BUDGET')
        inv={r['name']:r for r in inventory}
        need(all(inv[n]['exists'] and inv[n]['expected_size_matches'] for n in plan['hash_names']),
             'ORIGINAL14_NOT_CURRENTLY_COMPLETE_NO_HASH_FALLBACK')
        phase='once14_whole_SHA'
        for name in plan['hash_names']:
            check();active=name;attempted+=1;progress();pin=byname[name];path=no_links(old/name)
            before=path.lstat();need(stamp(before)==inv[name]['stat'],'TARGET_CHANGED_AFTER_INVENTORY')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                need(stamp(os.fstat(fd))==stamp(before),'OPENED_TARGET_IDENTITY')
                h=hashlib.sha256();count=0;last_progress=time.monotonic()
                while count<pin['bytes']:
                    check();block=os.read(fd,min(plan['chunk_bytes'],pin['bytes']-count))
                    need(bool(block),'TARGET_SHORT_READ');h.update(block);count+=len(block);read_bytes+=len(block)
                    need(read_bytes<=plan['hash_bytes'],'ONCE_READ_BUDGET')
                    if time.monotonic()-last_progress>=5:progress();last_progress=time.monotonic()
                check();extra=os.read(fd,1);read_bytes+=len(extra);need(not extra,'TARGET_TRAILING_BYTES')
                after=os.fstat(fd);need(stamp(before)==stamp(after)==stamp(path.lstat()),'TARGET_STABLE_FD_PATH_CHANGED')
                digest=h.hexdigest();need(digest==pin['SHA256'],'CURRENT_WHOLE_SHA_MISMATCH')
                completed.append(dict(name=name,bytes=count,SHA256=digest,current_target_stat=stamp(after),
                    observer_host=owner['host'],observer_boot_id=owner['boot_id'],whole_file_read_once=True,
                    later_reuse_requires_same_current_identity=True))
            finally:os.close(fd)
            progress()
        phase='final14_stat_stability'
        for row in completed:
            check();need(stamp(no_links(old/row['name']).lstat())==row['current_target_stat'],'QUALIFIED_MEMBER_CHANGED_BEFORE_FINAL_RECEIPT')
        phase='complete';check()
    except BaseException as e:
        error=dict(type=type(e).__name__,phase=phase,reason_SHA256=sha(str(e).encode()),errno=getattr(e,'errno',None))
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
    passed=error is None and len(completed)==14 and read_bytes==plan['hash_bytes']
    result=dict(context,owner=owner,step=plan['parent']+'.'+os.environ['SLURM_STEP_ID'],passed=passed,phase=phase,error=error,resources=resources,
        planned_files=26,planned_hash_files=14,attempted_hash_files=attempted,completed_hash_files=len(completed),
        missing_hash_files=14-len(completed),actual_hash_bytes=read_bytes,planned_hash_bytes=plan['hash_bytes'],
        verified_members=completed,remaining26_asset_qualification=False,all26_assets_qualified=False,
        old_payload_writes=0,hardlinks_created=0,transfer_bytes=0,A100_calls=0,Torch_SDK_GPU_calls=0,
        ordinary_read_atime_effect='NOT_ASSERTED_UNCHANGED',physical_IO='UNKNOWN_NOT_MEASURED',
        elapsed_monotonic_seconds=time.monotonic()-started,self_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU_GO=False,Model_GO=False,replay_allowed=False)
    save(root/'outcome.json',result)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(request_id=context['request_id'],passed=passed,completed_hash_files=len(completed),
                         planned_hash_files=14,actual_hash_bytes=read_bytes,GPU_GO=False),sort_keys=True))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main(CONTEXT,PLAN,CLAIM))
