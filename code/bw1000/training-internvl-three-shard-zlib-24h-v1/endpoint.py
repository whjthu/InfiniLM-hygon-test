"""One immutable shard, bounded lossless zlib CPU endpoints; no old assets or tensor imports."""
import hashlib,os,resource,signal,stat,time,zlib
from pathlib import Path
from common import D,P,need,member,save,own,budget,no_links,stable,verify_marker,line_fd,exact_fd,control,fsync_dir
from protocol import pack,unpack,raw_length,decompress_bounded,versions

def main(mode,context):
    started=time.monotonic();row=member(context);root=D.parent;epoch=context['epoch'];deadline=min(epoch+P['budget']['payload_stop'],context.get('payload_deadline',epoch+P['budget']['payload_stop']));left=budget(epoch,deadline-epoch)
    need(mode in ('source','receiver') and str(root)==row['source_namespace' if mode=='source' else 'target_namespace'],'EXACT_NEW_FILE_ROLE_NAMESPACE')
    def interrupt(*_):raise TimeoutError('ORIGINAL86400_PAYLOAD_STOP_OR_SIGNAL')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(s,interrupt)
    signal.setitimer(signal.ITIMER_REAL,left())
    available=set(os.sched_getaffinity(0));cpus=row['source_CPU_affinity'] if mode=='source' else sorted(available-{0,1})[:2]
    need(len(cpus)==2 and not set(cpus)&{0,1} and set(cpus)<=available,'FIXED_TWO_AVAILABLE_CPU')
    if mode=='source':need(cpus=={2:[10,11],3:[12,13],4:[14,15]}[row['file_id']],'EXACT_SOURCE_MASK_NO_SUBSTITUTION')
    os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_AS,(P['source_AS_bytes'],P['source_AS_bytes']))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:os.nice(10-nice)
    os.environ['CUDA_VISIBLE_DEVICES']='';owner=own();need(owner['UID']==(1001 if mode=='source' else 154459),'EXPECTED_UID')
    identity=dict(context,mode=mode,owner=owner,CPU_affinity=cpus,allocated_CPU=2 if mode=='source' else 4,worker_CPU=2,AS_limit_bytes=4294967296,AS_is_not_RSS=True,nice=os.getpriority(os.PRIO_PROCESS,0),GPU_calls=0,zlib=versions(),payload_deadline=deadline)
    if mode=='receiver':
        need(os.environ.get('SLURM_JOB_ID')==row['parent'] and os.environ.get('SLURM_STEP_ID','').isdigit() and os.environ.get('SLURM_CPUS_PER_TASK')=='4' and owner['host']==row['node'],'ACTUAL_FIXED_COMPUTE_ALLOC4')
        identity['step']=row['parent']+'.'+os.environ['SLURM_STEP_ID'];identity['common_marker']=verify_marker(context)
    scope=root/('source-stream' if mode=='source' else 'receiver');scope.mkdir(mode=0o700);save(scope/'intent.json',identity)
    counters=dict(source_disk_read_bytes=0,source_whole_hash_bytes=0,raw_block_hash_bytes=0,compressed_block_hash_bytes=0,wire_hash_bytes=0,compressed_produced_bytes=0,decompress_input_bytes=0,decoded_raw_bytes=0,target_whole_hash_bytes=0,payload_pipe_read_bytes=0,payload_pipe_write_bytes=0,payload_file_write_bytes=0,GO_control_read_bytes=0)
    phases={};hraw=hashlib.sha256();hwire=hashlib.sha256();blocks=0;raw_total=0;wire_total=0;error=None;fd=None;before=None;path=None;destination=None;partial=None;file_proof=None;last_raw=0;last_time=time.monotonic();released=False
    def timed(name,fn):
        t=time.monotonic()
        try:return fn()
        finally:
            v=phases.setdefault(name,dict(seconds=0.,calls=0));v['seconds']+=time.monotonic()-t;v['calls']+=1
    def rd(f,n,source=False):
        left();b=timed('source_disk_read' if source else 'receiver_pipe_read_blocked',lambda:os.read(f,min(n,P['chunk_bytes'])));counters['source_disk_read_bytes' if source else 'payload_pipe_read_bytes']+=len(b);return b
    def write_all(f,b,key,phase):
        position=0
        while position<len(b):
            left();n=timed(phase,lambda:os.write(f,memoryview(b)[position:position+P['chunk_bytes']]));need(n>0,'SHORT_WRITE');counters[key]+=n;position+=n
    def progress(force=False):
        nonlocal last_raw,last_time
        if force or raw_total-last_raw>=P['progress_bytes'] or time.monotonic()-last_time>=P['progress_seconds']:
            timed('progress_JSON_write_fsync_rename',lambda:save(scope/'progress.json',dict(identity,blocks_completed=blocks,planned_blocks=row['blocks'],raw_processed_bytes=raw_total,wire_processed_bytes=wire_total,counters=counters,phase_costs=phases,observed_epoch=time.time()),replace=True));last_raw=raw_total;last_time=time.monotonic()
    try:
        if mode=='source':
            path=no_links(row['source']);s=path.stat();need(stat.S_ISREG(s.st_mode) and s.st_size==row['bytes'],'SOURCE_METADATA_ONLY_READY')
            identity['source_metadata_stat']=stable(s)
        else:
            destination=no_links(row['target_payload']);need(str(destination.parent)==P['modelroot'] and not destination.exists(),'EXCLUSIVE_NEW_SHARD_PATH');partial=destination.with_name(destination.name+'.incomplete')
            free=os.statvfs(destination.parent);need(free.f_bavail*free.f_frsize>P['raw_bytes']+64*1024**2,'AGGREGATE_THREE_FILE_FREE_BYTES_QUOTA_UNKNOWN')
            fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);before=os.fstat(fd);need(before.st_size==0 and stat.S_ISREG(before.st_mode),'NEW_EMPTY_COMPUTE_PARTIAL');identity['prepared_output_stat']=stable(before)
        control(dict(identity,stage='READY',planned_files=1,group_files=3,planned_blocks=row['blocks'],planned_raw_bytes=row['bytes'],max_wire_bytes=row['max_wire_bytes'],body_read_before_GO=False))
        go=line_fd(0,16);counters['GO_control_read_bytes']=len(go);need(go==b'GO\n','RELEASE_ONLY_AFTER_ALL_SIX_READY');released=True
        if mode=='source':
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'],'ORIGINAL_REGULAR_FD_SIZE')
        for index in range(row['blocks']):
            left();expected=raw_length(row,index)
            if mode=='source':
                raw=exact_fd(fd,expected,lambda f,n:rd(f,n,True));counters['source_whole_hash_bytes']+=len(raw);timed('source_whole_hash',lambda:hraw.update(raw));counters['raw_block_hash_bytes']+=len(raw);raw_digest=timed('raw_block_hash',lambda:hashlib.sha256(raw).digest())
                def compress():
                    obj=zlib.compressobj(1,zlib.DEFLATED,15,8,zlib.Z_DEFAULT_STRATEGY);return obj.compress(raw)+obj.flush(zlib.Z_FINISH)
                comp=timed('source_zlib_compress',compress);counters['compressed_produced_bytes']+=len(comp);need(0<len(comp)<=expected+65536,'COMPRESSED_CAP_NO_RAW_FALLBACK');counters['compressed_block_hash_bytes']+=len(comp);comp_digest=timed('compressed_block_hash',lambda:hashlib.sha256(comp).digest());header=pack(row,index,len(comp),raw_digest,comp_digest)
            else:
                header=exact_fd(0,128,rd);meta=unpack(header,row,index);comp=exact_fd(0,meta['compressed_bytes'],rd);counters['compressed_block_hash_bytes']+=len(comp);need(timed('compressed_block_hash',lambda:hashlib.sha256(comp).digest())==meta['compressed_digest'],'COMPRESSED_BLOCK_SHA_BEFORE_DECODE')
                counters['decompress_input_bytes']+=len(comp);raw=timed('target_bounded_decompress',lambda:decompress_bounded(comp,expected));counters['decoded_raw_bytes']+=len(raw);counters['raw_block_hash_bytes']+=len(raw);need(timed('raw_block_hash',lambda:hashlib.sha256(raw).digest())==meta['raw_digest'],'DECODED_RAW_BLOCK_SHA');counters['target_whole_hash_bytes']+=len(raw);timed('target_whole_hash',lambda:hraw.update(raw))
            for part in (header,comp):
                counters['wire_hash_bytes']+=len(part);timed('wire_hash',lambda:hwire.update(part))
                if mode=='source':write_all(1,part,'payload_pipe_write_bytes','source_pipe_write_blocked')
            if mode=='receiver':write_all(fd,raw,'payload_file_write_bytes','target_file_write')
            raw_total+=len(raw);wire_total+=len(header)+len(comp);blocks+=1;progress()
        need(raw_total==row['bytes'] and blocks==row['blocks'] and wire_total<=row['max_wire_bytes'] and hraw.hexdigest()==row['SHA256'],'ORIGINAL_FULL_FILE_SHA_AND_BLOCK_DENOMINATOR')
        if mode=='source':
            need(not os.read(fd,1) and stable(before)==stable(os.fstat(fd))==stable(path.stat()),'SOURCE_WHOLE_STABLE_FD_PATH_EOF');file_proof=dict(name=row['name'],bytes=raw_total,SHA256=hraw.hexdigest(),source_stat=stable(before),stable_FD_path=True)
        else:
            need(not rd(0,1),'TRAILING_BYTES_OR_MISSING_ROOT_EOF_COMMIT');timed('target_file_fsync',lambda:os.fsync(fd));after=os.fstat(fd);need(after.st_size==row['bytes'] and [before.st_dev,before.st_ino]==[after.st_dev,after.st_ino] and stable(after)==stable(partial.stat()),'TARGET_FILE_FSYNC_STABLE_FD_PATH');need(not destination.exists(),'FINAL_FILE_ALREADY_EXISTS')
            timed('target_pre_rename_directory_fsync',lambda:fsync_dir(destination.parent));need(stable(after)==stable(os.fstat(fd))==stable(partial.stat()),'PRE_RENAME_FSYNC_STABLE_FD_PATH');partial.rename(destination);timed('target_directory_fsync',lambda:fsync_dir(destination.parent))
            renamed=os.fstat(fd);need(stable(after)[:4]==stable(renamed)[:4],'RENAME_CARRIES_DEV_INODE_SIZE_MTIME')
            need(stable(renamed)==stable(destination.stat())==stable(os.fstat(fd)),'RENAMED_FD_DESTINATION_FULL_STAMP_STABLE');after=renamed
            file_proof=dict(name=row['name'],bytes=raw_total,SHA256=hraw.hexdigest(),target_stat=stable(after),stable_FD_path=True,file_fsync=True,pre_rename_directory_fsync=True,directory_fsync=True,new_exclusive_inode=True,observer_host=owner['host'],observer_boot_id=owner['boot_id'],observer_UID=owner['UID'])
            need(verify_marker(context)==identity['common_marker'],'COMMON_MARKER_UNCHANGED')
        progress(True)
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:256])
    finally:
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
        signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+P['budget']['host_stop']-time.time()))
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        if fd is not None:os.close(fd)
    passed=error is None and file_proof is not None and blocks==row['blocks']
    result=dict(identity,passed=passed,error=error,planned_files=1,completed_files=1 if passed else 0,group_files=3,planned_blocks=row['blocks'],blocks_completed=blocks,planned_raw_bytes=row['bytes'],raw_processed_bytes=raw_total,wire_processed_bytes=wire_total,raw_SHA256=hraw.hexdigest(),wire_SHA256=hwire.hexdigest(),file_proof=file_proof,counters=counters,phase_costs=phases,finished_epoch=time.time(),wall_seconds=time.monotonic()-started,self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,old14_accesses=0,old_partial_accesses=0,initial362_364_accesses=0,assembly_actions=0,hardlinks=0,tensor_decodes=0,GPU_GO=False,Model_GO=False,training_GO=False,raw_fallback=False,replay_allowed=False,phase_semantics='Per-process logical syscall returns and blocking time, not physical wire/disk bytes; concurrent costs not added as network-exclusive; post-write body rereads0')
    save(scope/'outcome.json',result)
    if mode=='receiver' or passed or not released:control(dict(result,stage='ENDPOINT_TERMINAL'))
    else:os.write(2,('SOURCE_FAILURE '+error['type']+' raw='+str(raw_total)+' wire='+str(wire_total)+'\n').encode())
    signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if passed else 1
