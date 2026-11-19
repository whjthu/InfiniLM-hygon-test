"""Bounded exact two-file binary endpoints; no tensor/model/device import."""
import base64,hashlib,json,os,resource,signal,stat,sys,time
from pathlib import Path,PurePosixPath
from common import D,P,need,sha,stable,no_links,save,own,budget,control

def line_fd(fd,cap=128):
    out=bytearray()
    while len(out)<cap:
        b=os.read(fd,1)
        if not b:return bytes(out)
        out.extend(b)
        if b==b'\n':return bytes(out)
    raise ValueError('CONTROL_LINE_BOUND')

def main(mode,context,reference_b64=''):
    started=time.monotonic();stop=min(context['epoch']+P['payload_stop_offset'],context.get('payload_deadline',context['epoch']+P['payload_stop_offset']));left=budget(context['epoch'],stop-context['epoch']);root=D.parent
    need(context['deadline']==context['epoch']+900 and mode in ('source','receiver'),'ORIGINAL900_CONTEXT')
    need(str(root)==P['source_namespace' if mode=='source' else 'target_namespace'],'EXACT_NEW_SCOPE')
    def expired(*_):raise TimeoutError('ORIGINAL_PAYLOAD_DEADLINE_OR_SIGNAL')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,expired)
    signal.setitimer(signal.ITIMER_REAL,left())
    cpus=sorted(set(os.sched_getaffinity(0))-{0,1});count=P['source_CPU'] if mode=='source' else P['target_worker_CPU'];need(len(cpus)>=count,'AVAILABLE_CPU_AFFINITY')
    os.sched_setaffinity(0,cpus[:count]);resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:os.nice(10-nice)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if mode=='receiver':
        need(os.environ.get('SLURM_JOB_ID')==P['target_parent'] and os.environ.get('SLURM_STEP_ID','').isdigit(),'REAL_ALLOCATED_COMPUTE_STEP')
        need(own()['host']==P['target_node'],'NO_LOGIN_PAYLOAD_WRITE')
    need(os.getuid()==(1001 if mode=='source' else 154459),'EXPECTED_ENDPOINT_UID')
    scope=root/('source-stream' if mode=='source' else 'receiver');scope.mkdir(mode=0o700)
    owner=own();identity=dict(context,mode=mode,owner=owner,CPU_affinity=cpus[:count],allocated_CPU=P['source_CPU'] if mode=='source' else P['target_CPU'],worker_CPU=count,AS_limit_bytes=4294967296,AS_is_not_RSS=True,nice=os.getpriority(os.PRIO_PROCESS,0),actual_GPU_allocation=0)
    if mode=='receiver':identity['step']=P['target_parent']+'.'+os.environ['SLURM_STEP_ID']
    save(scope/'intent.json',identity)
    counters=dict(source_disk_read_bytes=0,reference_memory_bytes=0,payload_pipe_read_bytes=0,payload_pipe_write_bytes=0,payload_file_write_bytes=0,hash_input_bytes=0,reference_validation_hash_bytes=0,reference_postread_bytes=0)
    phases={};completed=[];total=0;active=None;offset=0;last_bytes=0;last_time=time.monotonic();error=None
    def timed(name,fn):
        mark=time.monotonic()
        try:return fn()
        finally:
            v=phases.setdefault(name,dict(seconds=0.,calls=0));v['seconds']+=time.monotonic()-mark;v['calls']+=1
    def write_all(fd,b,key,phase):
        used=0
        while used<len(b):
            left();n=timed(phase,lambda:os.write(fd,b[used:]));need(n>0,'SHORT_WRITE');used+=n;counters[key]+=n
    def progress(force=False):
        nonlocal last_bytes,last_time
        if force or total-last_bytes>=P['progress_bytes'] or time.monotonic()-last_time>=P['progress_seconds']:
            timed('progress_receipt_write',lambda:save(scope/'progress.json',dict(identity,logical_payload_bytes=total,completed_files=len(completed),active_member=active,member_offset=offset,counters=counters,phase_costs=phases,whole_MiB_s=total/1048576/max(.000001,time.monotonic()-started),interval_MiB_s=(total-last_bytes)/1048576/max(.000001,time.monotonic()-last_time),observed_epoch=time.time()),replace=True))
            last_bytes=total;last_time=time.monotonic()
    def validate_ref(raw):
        pin=P['local_reference'];counters['reference_validation_hash_bytes']+=len(raw);need(len(raw)==pin['bytes']==25357 and sha(raw)==pin['SHA256'],'FULL_REFERENCE_PIN')
        value=json.loads(raw);need(value['schema']=='native_first_two_inputs_v1' and value['case']=='T03.sft' and value['world_size']==8 and value['accumulation']==4 and len(value['rows'])==64,'REFERENCE_DENOMINATOR')
    try:
        need([r['name'] for r in P['files']]==['reference/input-fingerprints.json','initial/rank-00/initial.pt'] and sum(r['bytes'] for r in P['files'])==174847256,'EXACT_TWO_FILE_SCOPE')
        reference=base64.b64decode(reference_b64,validate=True) if mode=='source' else None
        if mode=='source':
            validate_ref(reference)
            path=no_links(P['files'][1]['source']);s=path.stat();need(stat.S_ISREG(s.st_mode) and s.st_size==174821899,'ORIGINAL_SOURCE_STAT')
        else:
            free=os.statvfs(root);need(free.f_bavail*free.f_frsize>174847256+64*1024**2,'TARGET_AVAILABLE_BYTES_QUOTA_UNKNOWN')
        control(dict(identity,stage='READY',expected_files=2,expected_payload_bytes=174847256,reference_origin='PC exact small original, injected once' if mode=='source' else None))
        need(line_fd(0)==b'GO\n','ROOT_RELEASE_AFTER_BOTH_READY')
        for row in P['files']:
            left();active=row['name'];offset=0;h=hashlib.sha256();source_fd=None;destination_fd=None;before=None;source_path=None;destination=None;partial=None
            member=PurePosixPath(active);need(not member.is_absolute() and '..' not in member.parts and member.as_posix()==active,'SAFE_NEW_MEMBER')
            if mode=='source' and row['source_kind']=='A100_original_file':
                source_path=no_links(row['source']);source_fd=os.open(source_path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);before=os.fstat(source_fd)
                need(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'],'SOURCE_STABLE_FD_SIZE')
            if mode=='receiver':
                destination=root/'payload'/active;destination.parent.mkdir(parents=True,exist_ok=True);no_links(destination.parent)
                partial=destination.with_name(destination.name+'.incomplete');destination_fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);before=os.fstat(destination_fd)
            try:
                while offset<row['bytes']:
                    left();amount=min(P['chunk_bytes'],row['bytes']-offset)
                    if mode=='source':
                        if source_fd is None:
                            block=reference[offset:offset+amount];counters['reference_memory_bytes']+=len(block)
                        else:
                            block=timed('source_disk_read',lambda:os.read(source_fd,amount));counters['source_disk_read_bytes']+=len(block)
                    else:
                        block=timed('receiver_pipe_read_blocked',lambda:os.read(0,amount));counters['payload_pipe_read_bytes']+=len(block)
                    need(block,'UNEXPECTED_PAYLOAD_EOF')
                    timed('payload_hash',lambda:h.update(block));counters['hash_input_bytes']+=len(block)
                    if mode=='source':write_all(1,block,'payload_pipe_write_bytes','source_pipe_write_blocked')
                    else:write_all(destination_fd,block,'payload_file_write_bytes','receiver_file_write')
                    offset+=len(block);total+=len(block);progress()
                need(h.hexdigest()==row['SHA256'],'EXACT_WHOLE_FILE_SHA')
                if mode=='source' and source_fd is not None:
                    need(not os.read(source_fd,1) and stable(before)==stable(os.fstat(source_fd))==stable(source_path.stat()),'SOURCE_WHOLE_STABLE_FD_PATH')
                if mode=='receiver':
                    timed('receiver_file_fsync',lambda:os.fsync(destination_fd));closed=os.fstat(destination_fd)
                    need(closed.st_size==row['bytes'] and [before.st_dev,before.st_ino]==[closed.st_dev,closed.st_ino] and stable(closed)==stable(partial.stat()),'RECEIVER_WRITE_FSYNC_STABLE')
            finally:
                if source_fd is not None:os.close(source_fd)
                if destination_fd is not None:os.close(destination_fd)
            if mode=='receiver':
                need(not destination.exists() and not destination.is_symlink(),'FINAL_NEW_MEMBER_EXISTS');partial.rename(destination)
                parent_fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:timed('receiver_member_directory_fsync',lambda:os.fsync(parent_fd))
                finally:os.close(parent_fd)
            item=dict(name=active,bytes=offset,SHA256=h.hexdigest(),whole_SHA_qualified=True)
            if mode=='source' and source_fd is not None:item['source_stat']=stable(before)
            if mode=='receiver':item['target_stat']=stable(destination.stat());item['observer_host']=owner['host']
            completed.append(item);progress(True)
        need(total==174847256 and len(completed)==2,'FULL_TWO_FILES_TOTAL')
        if mode=='receiver':
            need(not os.read(0,1),'TRAILING_PAYLOAD');ref_after=(root/'payload/reference/input-fingerprints.json').read_bytes();counters['reference_postread_bytes']+=len(ref_after);validate_ref(ref_after)
            directory=os.open(root/'payload',os.O_RDONLY|os.O_DIRECTORY)
            try:timed('receiver_directory_fsync',lambda:os.fsync(directory))
            finally:os.close(directory)
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:256])
    signal.setitimer(signal.ITIMER_REAL,0)
    passed=error is None and total==174847256 and len(completed)==2
    result=dict(identity,passed=passed,completed_files=len(completed),planned_files=2,logical_payload_bytes=total,expected_payload_bytes=174847256,verified_members=completed,counters=counters,phase_costs=phases,error=error,active_member=active,member_offset=offset,finished_epoch=time.time(),self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-started,old14_accesses=0,old_partial_accesses=0,hardlinks=0,tensor_decodes=0,Torch_SDK_GPU_calls=0,other7rank_RNG_inherited=False,all26_assets_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,post_write_large_initial_body_rehashed=False,post_write_reference_body_rehashed=mode=='receiver',phase_semantics='Per-process blocking durations; concurrent phases not added as exclusive network cost; physical wire/IO unknown',byte_counter_scope='Logical payload and successful syscall returns; control/script/reference bootstrap overhead separately accounted by relay',replay_allowed=False)
    save(scope/'outcome.json',result)
    # A source failure must not inject a JSON trailer into an incomplete binary payload.
    if mode=='receiver' or passed:control(dict(result,stage='ENDPOINT_TERMINAL'))
    else:sys.stderr.write(json.dumps(dict(mode=mode,error_type=error['type'],payload_bytes=total))+'\n');sys.stderr.flush()
    return 0 if passed else 1
