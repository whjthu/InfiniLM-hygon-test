"""Single compute observer: hash exactly3 new shards, then copy exactly14 old qualified files."""
import hashlib,os,resource,signal,stat,time
from pathlib import Path
from common import D,P,DEP,need,sha,stable,no_links,save,load,own,budget,canonical,control
from dependency_gate import validate_snapshot

def syncdir(p):
    f=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(f)
    finally:os.close(f)

def original_marker():
    path=no_links(Path(P['original_common_root'])/'CONTEXT.json');need(path.is_file() and path.stat().st_size<=16384,'ORIGINAL_MARKER_REQUIRED')
    expected=dict(schema='INTERNVL_THREE_SHARD_ZLIB_24H_V1',label='training-internvl-three-shard-zlib-24h-v1',context=DEP['original_context'],first_receiver_file_id=2,common_root=P['original_common_root'],modelroot=P['modelroot'])
    a=path.stat();need(a.st_uid==154459,'ORIGINAL_MARKER_OWNER');raw=path.read_bytes();need(stable(a)==stable(path.stat()) and raw==canonical(expected),'ORIGINAL_MARKER_CONTEXT_AND_SAME_OBSERVER_STABILITY')
    return dict(bytes=len(raw),SHA256=sha(raw),stat=stable(a),owner_UID=a.st_uid,observer=own())

def main(context,snapshot):
    started=time.monotonic();epoch=context['epoch'];need(context['deadline']==epoch+900,'ONE_ORIGINAL900_EPOCH');validate_snapshot(snapshot);need(sha(canonical(snapshot))==context['dependency_snapshot_SHA256'],'EXACT_DEPENDENCY_SNAPSHOT')
    root=D.parent;need(str(root)==P['namespace'],'EXCLUSIVE_OWN_SOURCE_NAMESPACE');need(os.getuid()==154459 and os.uname().nodename==P['node'] and os.environ.get('SLURM_JOB_ID')==P['parent'] and os.environ.get('SLURM_STEP_ID','').isdigit() and os.environ.get('SLURM_CPUS_PER_TASK')=='4','ACTUAL_ALLOC4_FIXED_COMPUTE_UID')
    cpus=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(cpus)>=2,'WORKER2_AFFINITY');os.sched_setaffinity(0,cpus[:2]);resource.setrlimit(resource.RLIMIT_AS,(P['AS_limit_bytes'],P['AS_limit_bytes']));nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:os.nice(10-nice)
    os.environ['CUDA_VISIBLE_DEVICES']='';owner=own();work_end=min(epoch+P['budget']['body_stop'],context['step_end']-20);left=budget(epoch,work_end-epoch)
    def stop(*_):raise TimeoutError('ORIGINAL_ASSEMBLY_WORK_STOP')
    for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,left());scope=root/'worker';scope.mkdir(mode=0o700)
    identity=dict(context,owner=owner,step=P['parent']+'.'+os.environ['SLURM_STEP_ID'],CPU_affinity=cpus[:2],allocated_CPU=4,worker_CPU=2,AS_limit_bytes=4294967296,AS_is_not_RSS=True,nice=os.getpriority(os.PRIO_PROCESS,0),work_deadline=work_end)
    save(scope/'intent.json',identity);reads=[];copies=[];fds=[];targets=[];error=None;phase='preconditions';active=None;total=0;last_total=0;last_time=time.monotonic();marker=None
    counters=dict(read3_bytes=0,copy14_source_read_bytes=0,total_read_bytes=0,total_write_bytes=0,read_hash_bytes=0,output_hash_bytes=0,read_calls=0,write_calls=0)
    phases={};buffer=bytearray(P['chunk_bytes']);view=memoryview(buffer)
    def timed(name,fn):
        t=time.monotonic()
        try:return fn()
        finally:
            p=phases.setdefault(name,dict(seconds=0.,calls=0));p['seconds']+=time.monotonic()-t;p['calls']+=1
    def progress(force=False):
        nonlocal last_total,last_time
        if force or total-last_total>=P['progress_bytes'] or time.monotonic()-last_time>=P['progress_seconds']:
            timed('progress_JSON_write_fsync_rename',lambda:save(scope/'progress.json',dict(identity,phase=phase,active=active,completed_read3=len(reads),completed_copy14=len(copies),planned_files=17,counters=counters,phase_costs=phases,observed_epoch=time.time()),replace=True));last_total=total;last_time=time.monotonic()
    def open_input(row):
        path=no_links(row['source']);before=path.stat();need(stat.S_ISREG(before.st_mode) and before.st_uid==154459 and before.st_size==row['bytes'],'EXACT_OWNED_REGULAR_INPUT_SIZE')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:need(stable(before)==stable(os.fstat(fd)),'INPUT_OPEN_STABLE')
        except BaseException:os.close(fd);raise
        fds.append((fd,path,before));return fd,path,before
    def check_input(fd,path,before):need(stable(before)==stable(os.fstat(fd))==stable(path.stat()),'INPUT_SAME_OBSERVER_FULL_STAMP_UNCHANGED')
    try:
        # No path discovery/rglob/26 inventory. Even metadata paths are restricted to the exact17 manifest.
        marker=original_marker();model=no_links(Path(P['modelroot']));need(model.is_dir() and model.stat().st_uid==154459,'ORIGINAL_FIXED_MODELROOT_OWNER')
        for row in P['read3']:need(Path(row['source']).parent==model,'EXACT3_SOURCE_PATHS')
        for row in P['copy14']:
            dest=no_links(row['destination']);need(dest.parent==model and not dest.exists() and not dest.is_symlink() and not dest.with_name(dest.name+'.assembly-incomplete').exists(),'ALL14_NEW_EXCLUSIVE_DESTINATIONS')
        v=os.statvfs(model);need(v.f_bavail*v.f_frsize>P['copy14_bytes']+64*1024**2,'COPY14_FREE_BYTES_QUOTA_UNKNOWN')
        phase='single_observer_hash3_before_any_copy'
        for row in P['read3']:
            active=row['name'];fd,path,before=open_input(row);h=hashlib.sha256();offset=0
            while offset<row['bytes']:
                left();n=timed('new3_read',lambda:os.readv(fd,[view[:min(P['chunk_bytes'],row['bytes']-offset)]]));need(n>0,'NEW3_SHORT_EOF');b=view[:n];counters['read_calls']+=1;counters['read3_bytes']+=len(b);counters['total_read_bytes']+=len(b);counters['read_hash_bytes']+=len(b);timed('new3_whole_hash',lambda:h.update(b));offset+=len(b);total+=len(b);progress()
            need(os.readv(fd,[view[:1]])==0 and h.hexdigest()==row['SHA256'],'NEW3_ORIGINAL_WHOLE_SHA_AND_EOF');check_input(fd,path,before)
            reads.append(dict(name=row['name'],bytes=offset,SHA256=h.hexdigest(),source_stat=stable(before),stable_FD_path=True,observer=owner,whole_SHA_qualified=True))
        need(len(reads)==3 and counters['read3_bytes']==P['read3_bytes'],'THREE_OF_THREE_BEFORE_COPY_OR_WRITE')
        phase='copy_exact14_once'
        for row in P['copy14']:
            active=row['name'];source,path,before=open_input(row);destination=Path(row['destination']);partial=destination.with_name(destination.name+'.assembly-incomplete');out=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);targets.append(out);out_before=os.fstat(out);hs=hashlib.sha256();ht=hashlib.sha256();offset=0
            while offset<row['bytes']:
                left();n=timed('old14_source_read',lambda:os.readv(source,[view[:min(P['chunk_bytes'],row['bytes']-offset)]]));need(n>0,'COPY14_SHORT_EOF');b=view[:n];counters['read_calls']+=1;counters['copy14_source_read_bytes']+=len(b);counters['total_read_bytes']+=len(b);counters['read_hash_bytes']+=len(b);counters['output_hash_bytes']+=len(b);timed('old14_source_hash',lambda:hs.update(b));timed('new14_output_hash',lambda:ht.update(b));position=0
                while position<len(b):
                    left();n=timed('new14_file_write',lambda:os.write(out,memoryview(b)[position:]));need(n>0,'COPY14_SHORT_WRITE');position+=n;counters['write_calls']+=1;counters['total_write_bytes']+=n
                offset+=len(b);total+=len(b);progress()
            need(os.readv(source,[view[:1]])==0 and hs.hexdigest()==ht.hexdigest()==row['SHA256'],'COPY14_ORIGINAL_WHOLE_SHA_EOF');check_input(source,path,before)
            timed('new14_file_fsync',lambda:os.fsync(out));after=os.fstat(out);need(after.st_size==row['bytes'] and [out_before.st_dev,out_before.st_ino]==[after.st_dev,after.st_ino] and stable(after)==stable(partial.stat()),'NEW14_FILE_FSYNC_STABLE_FD_PATH')
            need((after.st_dev,after.st_ino)!=(before.st_dev,before.st_ino),'NEW14_NOT_ORIGINAL_INODE');timed('new14_pre_rename_dir_fsync',lambda:syncdir(model));need(stable(after)==stable(os.fstat(out))==stable(partial.stat()) and not destination.exists(),'NEW14_PRE_RENAME_STABLE')
            partial.rename(destination);timed('new14_post_rename_dir_fsync',lambda:syncdir(model));renamed=os.fstat(out);need(stable(after)[:4]==stable(renamed)[:4] and stable(renamed)==stable(destination.stat())==stable(os.fstat(out)),'NEW14_POST_RENAME_FD_PATH_FULL_STAMP_STABLE')
            copies.append(dict(name=row['name'],bytes=offset,SHA256=hs.hexdigest(),source_stat=stable(before),target_stat=stable(renamed),source_stable_FD_path=True,target_stable_FD_path=True,file_fsync=True,pre_rename_directory_fsync=True,directory_fsync=True,new_exclusive_inode=True,observer=owner));progress()
        phase='final_same_observer_17'
        for fd,path,before in fds:check_input(fd,path,before)
        for row,fd in zip(P['copy14'],targets):
            proof=next(x for x in copies if x['name']==row['name']);need(proof['target_stat']==stable(os.fstat(fd))==stable(Path(row['destination']).stat()),'FINAL14_SAME_OBSERVER_STABLE_FD_PATH')
        need(marker==original_marker(),'ORIGINAL_CONTEXT_MARKER_UNCHANGED');need(len(reads)==3 and len(copies)==14 and counters['total_read_bytes']==P['total_read_bytes'] and counters['total_write_bytes']==P['total_write_bytes'],'ALL17_FULL_BYTES_DENOMINATORS');progress(True)
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:256])
    finally:
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+P['budget']['worker_tail']-time.time()))
        for s in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(s,signal.SIG_IGN)
        for fd in targets+[v[0] for v in fds]:
            try:os.close(fd)
            except OSError as e:
                if error is None:error=dict(type=type(e).__name__,reason='OWN_FD_CLOSE_FAILED')
    passed=error is None and len(reads)==3 and len(copies)==14
    result=dict(identity,passed=passed,error=error,planned_read3=3,planned_copy14=14,planned_files=17,completed_read3=len(reads),completed_copy14=len(copies),verified_read3=reads,verified_copy14=copies,dependency_snapshot_SHA256=sha(canonical(snapshot)),original_marker=marker,counters=counters,phase_costs=phases,active=active,phase=phase,finished_epoch=time.time(),wall_seconds=time.monotonic()-started,self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,buffer_max_bytes=1048576,old_partial_accesses=0,initial8_accesses=0,reference_accesses=0,hardlinks=0,tensor_decodes=0,GPU_calls=0,all26_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False,physical_IO_bytes='UNKNOWN',throughput_guaranteed=False,output_body_rereads=0)
    save(scope/'outcome.json',result);control(dict(result,stage='ASSEMBLY_WORKER_TERMINAL'));signal.setitimer(signal.ITIMER_REAL,0);return 0 if passed else 1
