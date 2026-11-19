"""Bounded original-byte delta endpoints; opaque bytes only, all-or-stop source precompute."""
import hashlib,json,os,resource,signal,stat,sys,time
from pathlib import PurePosixPath
from common import D,P,need,sha,stable,no_links,save,own,budget,control,canonical
from delta_codec import make_frame,parse_frame,validate_plan

def line_fd(fd,cap=128):
    out=bytearray()
    while len(out)<cap:
        b=os.read(fd,1)
        if not b:return bytes(out)
        out.extend(b)
        if b==b'\n':return bytes(out)
    raise ValueError('CONTROL_LINE_BOUND')

def main(mode,context):
    started=time.monotonic();stop=min(context['epoch']+P['payload_stop_offset'],context.get('payload_deadline',context['epoch']+P['payload_stop_offset']));left=budget(context['epoch'],stop-context['epoch']);root=D.parent
    need(context['deadline']==context['epoch']+900 and mode in ('source','receiver'),'ORIGINAL900_CONTEXT')
    need(str(root)==P['source_namespace' if mode=='source' else 'target_namespace'],'EXACT_NEW_SCOPE')
    def expired(*_):raise TimeoutError('ORIGINAL_PAYLOAD_DEADLINE_OR_SIGNAL')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,expired)
    signal.setitimer(signal.ITIMER_REAL,left())
    cpus=sorted(set(os.sched_getaffinity(0))-{0,1});count=P['source_CPU'] if mode=='source' else P['target_worker_CPU'];need(len(cpus)>=count,'AVAILABLE_CPU_AFFINITY')
    os.sched_setaffinity(0,cpus[:count]);resource.setrlimit(resource.RLIMIT_AS,(P['source_AS_bytes'],P['source_AS_bytes']))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<P['endpoint_nice']:os.nice(P['endpoint_nice']-nice)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if mode=='receiver':
        need(os.environ.get('SLURM_JOB_ID')==P['target_parent'] and os.environ.get('SLURM_STEP_ID','').isdigit(),'REAL_ALLOCATED_COMPUTE_STEP')
        need(own()['host']==P['target_node'],'NO_LOGIN_PAYLOAD_WRITE')
    need(os.getuid()==(1001 if mode=='source' else 154459),'EXPECTED_ENDPOINT_UID')
    scope=root/('source-stream' if mode=='source' else 'receiver');scope.mkdir(mode=0o700)
    owner=own();identity=dict(context,mode=mode,owner=owner,CPU_affinity=cpus[:count],allocated_CPU=P['source_CPU'] if mode=='source' else P['target_CPU'],worker_CPU=count,AS_limit_bytes=P['source_AS_bytes'],AS_is_not_RSS=True,nice=os.getpriority(os.PRIO_PROCESS,0),actual_GPU_allocation=0)
    if mode=='receiver':identity['step']=P['target_parent']+'.'+os.environ['SLURM_STEP_ID']
    save(scope/'intent.json',identity)
    counters=dict(source_disk_read_bytes=0,source_whole_hash_bytes=0,target_base_read_bytes=0,target_base_hash_bytes=0,payload_pipe_read_bytes=0,payload_pipe_write_bytes=0,payload_file_write_bytes=0,target_output_hash_bytes=0,frame_hash_bytes=0,changed_block_data_bytes=0,verified_changed_block_hash_bytes=0,compared_original_bytes=0,memory_base_copy_bytes=0)
    phases={};source_members=[];outputs=[];frames_done=[];patch_plan=None;total=0;active=None;offset=0;last_bytes=0;last_time=time.monotonic();error=None;base_fd=None;base_path=None;base_before=None;base_proof=None;binary_started=False
    def timed(name,fn):
        mark=time.monotonic()
        try:return fn()
        finally:
            v=phases.setdefault(name,dict(seconds=0.,calls=0));v['seconds']+=time.monotonic()-mark;v['calls']+=1
    def write_all(fd,b,key,phase):
        used=0
        while used<len(b):
            left();n=timed(phase,lambda:os.write(fd,memoryview(b)[used:]));need(n>0,'SHORT_WRITE');used+=n;counters[key]+=n
    def progress(force=False):
        nonlocal last_bytes,last_time
        if force or total-last_bytes>=P['progress_bytes'] or time.monotonic()-last_time>=P['progress_seconds']:
            timed('progress_receipt_write',lambda:save(scope/'progress.json',dict(identity,logical_payload_bytes=total,verified_source_files=len(source_members),completed_output_files=len(outputs),completed_frames=len(frames_done),active_member=active,member_offset=offset,counters=counters,phase_costs=phases,whole_MiB_s=total/1048576/max(.000001,time.monotonic()-started),observed_epoch=time.time()),replace=True))
            last_bytes=total;last_time=time.monotonic()
    def open_original(path,row):
        path=no_links(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'],'ORIGINAL_REGULAR_STABLE_FD_SIZE')
            return path,fd,before
        except BaseException:os.close(fd);raise
    def stable_original(path,fd,before):
        need(not os.read(fd,1) and stable(before)==stable(os.fstat(fd))==stable(path.stat()),'ORIGINAL_WHOLE_STABLE_FD_PATH')
    def read_base(path):
        nonlocal base_fd,base_path,base_before,base_proof,offset
        row=P['base'];base_path,base_fd,base_before=open_original(path,row);base=bytearray();h=hashlib.sha256();offset=0
        while offset<row['bytes']:
            left();b=timed('source_disk_read' if mode=='source' else 'target_base_disk_read',lambda:os.read(base_fd,min(P['chunk_bytes'],row['bytes']-offset)));need(b,'ORIGINAL_BASE_EOF')
            counters['source_disk_read_bytes' if mode=='source' else 'target_base_read_bytes']+=len(b)
            timed('source_whole_hash' if mode=='source' else 'target_base_hash',lambda:h.update(b));counters['source_whole_hash_bytes' if mode=='source' else 'target_base_hash_bytes']+=len(b)
            base.extend(b);offset+=len(b);progress()
        need(h.hexdigest()==row['SHA256'],'ORIGINAL_BASE_WHOLE_SHA');stable_original(base_path,base_fd,base_before)
        base_proof=dict(name=row['name'],rank=0,bytes=offset,SHA256=h.hexdigest(),source_stat=stable(base_before),stable_FD_path=True,whole_SHA_qualified=True)
        return base
    try:
        need(P['block_bytes']==65536 and P['frame_cap_bytes']==4194304 and [r['rank'] for r in P['files']]==list(range(1,8)),'FROZEN_SEVEN_RANK_BLOCK_SCOPE')
        if mode=='source':
            for row in P['source_files']:
                s=no_links(row['source']).stat();need(stat.S_ISREG(s.st_mode) and s.st_size==row['bytes'],'SOURCE_METADATA_ONLY_READY')
        else:
            s=no_links(P['base']['target']).stat();need(stat.S_ISREG(s.st_mode) and s.st_size==P['base']['bytes'],'EXISTING_362_BASE_METADATA_ONLY')
            free=os.statvfs(root);need(free.f_bavail*free.f_frsize>P['target_output_bytes']+64*1024**2,'TARGET_AVAILABLE_BYTES_QUOTA_UNKNOWN')
        control(dict(identity,stage='READY',expected_source_files=8,expected_output_files=7,max_patch_wire_bytes=P['wire_cap_bytes'],base_body_read_before_READY=False,base_metadata_path=P['base']['source' if mode=='source' else 'target']))
        need(line_fd(0)==b'GO\n','ROOT_RELEASE_AFTER_BOTH_READY')
        if mode=='source':
            active=P['base']['name'];base=read_base(P['base']['source']);source_members.append(base_proof);prepared=[];descriptors=[]
            for row in P['files']:
                active=row['name'];offset=0;h=hashlib.sha256();records=[];blocks=[];changed_bytes=0;index=0;path,fd,before=open_original(row['source'],row)
                try:
                    while offset<row['bytes']:
                        left();amount=min(65536,row['bytes']-offset);block=bytearray()
                        while len(block)<amount:
                            b=timed('source_disk_read',lambda:os.read(fd,amount-len(block)));need(b,'ORIGINAL_RANK_EOF');counters['source_disk_read_bytes']+=len(b);block.extend(b)
                        timed('source_whole_hash',lambda:h.update(block));counters['source_whole_hash_bytes']+=len(block)
                        same=timed('source_exact_byte_compare',lambda:block==base[offset:offset+amount]);counters['compared_original_bytes']+=amount
                        if not same:
                            rec=dict(index=index,bytes=amount,SHA256=timed('source_changed_block_hash',lambda:sha(block)));counters['verified_changed_block_hash_bytes']+=amount;records.append(rec);changed_bytes+=amount
                            # Exact cap includes all metadata. At most one 64KiB temporary block beyond cap is held; no raw fallback.
                            header=dict(schema=P['schema'],rank=row['rank'],name=row['name'],base_bytes=P['base']['bytes'],base_SHA256=P['base']['SHA256'],target_bytes=row['bytes'],target_SHA256=row['SHA256'],block_bytes=65536,blocks=records)
                            header_bytes=len(canonical(header));need(header_bytes<=P['header_cap_bytes'] and 12+header_bytes+changed_bytes<=P['frame_cap_bytes'],'COMPLETE_FRAME_CAP_NO_RAW_FALLBACK')
                            blocks.append(bytes(block))
                        offset+=amount;index+=1;progress()
                    need(h.hexdigest()==row['SHA256'],'ORIGINAL_RANK_WHOLE_SHA');stable_original(path,fd,before)
                    source_members.append(dict(name=row['name'],rank=row['rank'],bytes=offset,SHA256=h.hexdigest(),source_stat=stable(before),stable_FD_path=True,whole_SHA_qualified=True))
                finally:os.close(fd)
                frame=timed('source_frame_build',lambda:make_frame(row,records,blocks));digest=timed('source_frame_hash',lambda:sha(frame));counters['frame_hash_bytes']+=len(frame)
                descriptors.append(dict(rank=row['rank'],name=row['name'],bytes=len(frame),SHA256=digest,changed_blocks=len(records),changed_data_bytes=changed_bytes));prepared.append(frame);counters['changed_block_data_bytes']+=changed_bytes;progress(True)
            need(stable(base_before)==stable(os.fstat(base_fd))==stable(base_path.stat()),'SOURCE_BASE_STABLE_THROUGH_PRECOMPUTE')
            patch_plan=validate_plan(dict(stage='PATCH_PLAN',schema=P['schema'],frames=descriptors,total_wire_bytes=sum(map(len,prepared)),source_verified_members=source_members));save(scope/'patch-plan.json',patch_plan);control(patch_plan)
            binary_started=True
            for desc,frame in zip(descriptors,prepared):
                active=desc['name'];offset=0
                while offset<len(frame):
                    b=memoryview(frame)[offset:offset+P['chunk_bytes']];write_all(1,b,'payload_pipe_write_bytes','source_pipe_write_blocked');offset+=len(b);total+=len(b);progress()
                frames_done.append(desc);progress(True)
        else:
            patch_plan=validate_plan(json.loads(line_fd(0,65536)));save(scope/'patch-plan.json',patch_plan);source_members=patch_plan['source_verified_members']
            active=P['base']['name'];base=read_base(P['base']['target'])
            for row,desc in zip(P['files'],patch_plan['frames']):
                active=row['name'];offset=0;frame=bytearray()
                while len(frame)<desc['bytes']:
                    left();b=timed('receiver_pipe_read_blocked',lambda:os.read(0,min(P['chunk_bytes'],desc['bytes']-len(frame))));need(b,'PATCH_FRAME_EOF');frame.extend(b);counters['payload_pipe_read_bytes']+=len(b);total+=len(b);progress()
                counters['frame_hash_bytes']+=len(frame);changed=timed('receiver_frame_validate',lambda:parse_frame(frame,row,desc));counters['verified_changed_block_hash_bytes']+=desc['changed_data_bytes'];output=timed('target_memory_base_copy',lambda:bytearray(base));counters['memory_base_copy_bytes']+=len(output)
                def apply_exact_blocks():
                    for start,block in changed:output[start:start+len(block)]=block;counters['changed_block_data_bytes']+=len(block)
                timed('target_memory_apply_blocks',apply_exact_blocks)
                member=PurePosixPath(row['name']);need(not member.is_absolute() and '..' not in member.parts and member.as_posix()==row['name'],'SAFE_NEW_MEMBER')
                destination=root/'payload'/row['name'];destination.parent.mkdir(parents=True,exist_ok=True);no_links(destination.parent);need(not destination.exists() and not destination.is_symlink(),'EXCLUSIVE_NEW_OUTPUT')
                partial=destination.with_name(destination.name+'.incomplete');fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);before=os.fstat(fd);h=hashlib.sha256();offset=0
                try:
                    while offset<row['bytes']:
                        left();b=memoryview(output)[offset:offset+P['chunk_bytes']];timed('target_output_hash',lambda:h.update(b));counters['target_output_hash_bytes']+=len(b);write_all(fd,b,'payload_file_write_bytes','receiver_file_write');offset+=len(b);progress()
                    need(h.hexdigest()==row['SHA256'],'RECONSTRUCTED_ORIGINAL_RANK_WHOLE_SHA');timed('receiver_file_fsync',lambda:os.fsync(fd));after=os.fstat(fd)
                    need(after.st_size==row['bytes'] and [before.st_dev,before.st_ino]==[after.st_dev,after.st_ino] and stable(after)==stable(partial.stat()),'OUTPUT_FSYNC_STABLE_FD_PATH')
                    need([after.st_dev,after.st_ino]!=[base_before.st_dev,base_before.st_ino],'NEW_OUTPUT_INODE_NOT_BASE')
                finally:os.close(fd)
                partial.rename(destination);parent_fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:timed('receiver_member_directory_fsync',lambda:os.fsync(parent_fd))
                finally:os.close(parent_fd)
                final_stat=stable(destination.stat());need(final_stat==stable(after),'RENAMED_OUTPUT_STABLE')
                outputs.append(dict(name=row['name'],rank=row['rank'],bytes=offset,SHA256=h.hexdigest(),whole_SHA_qualified=True,target_stat=final_stat,observer_host=owner['host'],file_fsync=True,directory_fsync=True,stable_FD_path=True,new_inode_not_base=True));frames_done.append(desc);del output,frame,changed;progress(True)
            need(not os.read(0,1),'TRAILING_PATCH_BYTES')
            directory=os.open(root/'payload',os.O_RDONLY|os.O_DIRECTORY)
            try:timed('receiver_payload_directory_fsync',lambda:os.fsync(directory))
            finally:os.close(directory)
        need(patch_plan is not None and total==patch_plan['total_wire_bytes'] and len(frames_done)==7,'FULL_SEVEN_FRAME_DENOMINATOR')
        need(stable(base_before)==stable(os.fstat(base_fd))==stable(base_path.stat()),'BASE_UNCHANGED_THROUGH_ALL_OUTPUTS')
        need(len(source_members)==8 and (mode=='source' or len(outputs)==7),'SOURCE8_TARGET7_DENOMINATORS')
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:256])
    finally:
        if base_fd is not None:os.close(base_fd)
    signal.setitimer(signal.ITIMER_REAL,0)
    passed=error is None and patch_plan is not None and total==patch_plan['total_wire_bytes'] and len(frames_done)==7 and (mode=='source' or len(outputs)==7)
    result=dict(identity,passed=passed,source_verified_count=len(source_members),source_verified_members=source_members,output_completed_count=len(outputs),verified_members=outputs,verified_frames=frames_done,base_proof=base_proof,base_unchanged_through_scope=error is None,logical_payload_bytes=total,expected_payload_bytes=patch_plan['total_wire_bytes'] if patch_plan else None,planned_source_files=8,planned_output_files=7,counters=counters,phase_costs=phases,error=error,active_member=active,member_offset=offset,finished_epoch=time.time(),self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-started,old14_accesses=0,old_partial_accesses=0,hardlinks=0,tensor_decodes=0,Torch_SDK_GPU_calls=0,rank00_RNG_inherited=False,all26_assets_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,post_write_initial_body_rehashed=False,raw_fallback=False,control_bytes=dict(GO_read_bytes=3,patch_plan_bytes=len(canonical(patch_plan)) if patch_plan else 0,patch_plan_direction='stdout' if mode=='source' else 'stdin'),phase_semantics='Per-process blocking durations; concurrent phases must not be added as exclusive network cost; physical wire/IO unknown',replay_allowed=False)
    save(scope/'outcome.json',result)
    if mode=='receiver' or passed or not binary_started:control(dict(result,stage='ENDPOINT_TERMINAL'))
    else:sys.stderr.write(json.dumps(dict(mode=mode,error_type=error['type'],payload_bytes=total))+'\n');sys.stderr.flush()
    return 0 if passed else 1
