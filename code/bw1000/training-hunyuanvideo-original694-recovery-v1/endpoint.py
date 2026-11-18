"""Injected pinned modules plus PLAN/CLAIM/APPROVAL/CONTEXT/MODE; no model imports."""
import hashlib,json,os,re,resource,signal,socket,struct,sys,time
from pathlib import Path
import reuse
from reuse import require,canonical,sha,no_links,save,sync_directory,stable,read_pinned,pack,unpack,tick
from nice_policy import apply_nice
from atomic_files import existing,install
from permission_gate import validate

def exact(stream,count):
    out=bytearray()
    while len(out)<count:
        tick();part=stream.read(min(65536,count-len(out)))
        if not part:raise EOFError('TRUNCATED_NEW_RECOVERY_STREAM')
        out.extend(part)
    return bytes(out)

def arm():
    validate(CLAIM,APPROVAL,PLAN)
    require(CONTEXT['original_state']=='UNKNOWN' and CONTEXT['original_request_id']==PLAN['original_request_id']
        and re.fullmatch('[0-9a-f]{32}',CONTEXT['request_id'])
        and CONTEXT['request_id']!=PLAN['original_request_id'],'NEW_REQUEST_KEEPS_OLD_UNKNOWN')
    require(CONTEXT['deadline']==CONTEXT['epoch']+300 and CONTEXT['epoch']<=time.time()<CONTEXT['deadline'],'ONE300_EPOCH')
    reuse.DEADLINE=CONTEXT['deadline'];reuse.MONO_END=time.monotonic()+CONTEXT['deadline']-time.time()
    def hard(*_):os._exit(124)
    def stop(*_):
        signal.signal(signal.SIGALRM,hard)
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(reuse.DEADLINE-time.time(),reuse.MONO_END-time.monotonic())))
        raise TimeoutError('ORIGINAL260_WORK_LIMIT')
    for kind in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT,signal.SIGALRM):signal.signal(kind,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,CONTEXT['epoch']+260-time.time()))
    return None

def admit_resources():
    resource.setrlimit(resource.RLIMIT_AS,(PLAN['AS_bytes'],PLAN['AS_bytes']))
    allowed=sorted(os.sched_getaffinity(0));require(len(allowed)>=2,'TWO_CPU_AVAILABLE')
    os.sched_setaffinity(0,allowed[:2]);require(len(os.sched_getaffinity(0))==2,'ACTUAL_TWO_CPU')
    return apply_nice(lambda:os.getpriority(os.PRIO_PROCESS,0),os.nice)

def create_control(priority):
    if MODE=='source':root=no_links(Path(PLAN['source_control']))
    else:
        target=no_links(Path(PLAN['target']))
        require(target.parent.is_dir() and target.parent.stat().st_uid==os.getuid()==154459,'OWN_PROJECT_RESULTS')
        try:target.mkdir(mode=0o700)
        except FileExistsError:pass
        require(target.is_dir() and target.stat().st_uid==os.getuid(),'ORIGINAL_TARGET_TYPE_OWNER')
        root=target/PLAN['target_control_name']
    no_links(root);require(root.parent.is_dir() and root.parent.stat().st_uid==os.getuid(),'OWN_CONTROL_PARENT')
    root.mkdir(mode=0o700)  # Fixed new scope; an unknown/previous control can never be replayed.
    state=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    identity=dict(CONTEXT,mode=MODE,host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        PID=os.getpid(),startticks=int(state[19]),uid=os.getuid(),control_root=str(root),nice_policy=priority,
        affinity=sorted(os.sched_getaffinity(0)),resource_admission='PENDING',
        source_execution='PINNED_STDIN_MODULE_BYTES',original694_state='UNKNOWN')
    save(root/'intent.json',identity);sync_directory(root)
    return root,identity

def emit(value):
    sys.stdout.buffer.write(canonical(value));sys.stdout.buffer.flush()

def main():
    started=time.monotonic();root=None;identity=None;passed=False;detail={};failure=None
    try:
        arm();root,identity=create_control(None)  # Own identity survives even a resource/nice admission failure.
        priority=admit_resources()
        identity.update(nice_policy=priority,affinity=sorted(os.sched_getaffinity(0)),resource_admission='PASS')
        save(root/'resource-ready.json',identity);sync_directory(root)
        if MODE=='receiver':
            target=Path(PLAN['target']);states={};missing=[]
            for name,pin in PLAN['members'].items():
                found=existing(target,name,pin,PLAN['max_file_bytes'])
                if found is None:missing.append(name)
                else:states[name]=found
            missing=sorted(missing)
            emit(dict(kind='RECOVERY_READY',identity=identity,missing=missing,existing=states))
            size=struct.unpack('<Q',exact(sys.stdin.buffer,8))[0]
            require(size<=PLAN['max_compressed_bytes'],'BOUNDED_COMPRESSED_STREAM')
            if missing:
                require(size>0,'MISSING_PAYLOAD_FOR_REQUESTED_MEMBERS')
                files=unpack(exact(sys.stdin.buffer,size),PLAN,dict(CONTEXT,requested_members=missing))
                require(set(files)==set(missing),'EXACT_REQUESTED_MEMBERS')
                for name in missing:
                    states[name]=install(target,name,files[name],PLAN['members'][name],CONTEXT['request_id'],PLAN['max_file_bytes'])
            else:require(size==0,'NO_SOURCE_READ_WHEN_ALL_REUSED')
            require(set(states)==set(PLAN['members']),'COMPLETE_TWO_MEMBER_DENOMINATOR')
            # Reused files were fully verified above; ensure the same identities still hold without rereading them.
            for name,state in states.items():
                require(stable((target/name).stat())==state['file_identity'],'FINAL_IDENTITY_CHANGED')
            detail=dict(member_states=states,member_denominator=2,original694_state='UNKNOWN',tensor_decodes=0)
            save(root/'validation.json',dict(identity,**detail));sync_directory(target)
        else:
            require(MODE=='source','EXACT_ENDPOINT_MODE')
            names=CONTEXT['requested_members']
            require(names==sorted(set(names)) and set(names)<=set(PLAN['members']) and names,'REQUESTED_TWO_MEMBER_SUBSET')
            emit(dict(kind='RECOVERY_READY',identity=identity,requested_members=names))
            files={};stamps={}
            for name in names:
                pin=PLAN['members'][name]
                require(pin['source']==str(Path(PLAN['source_capture'])/name),'EXACT_CAPTURE_PATH')
                files[name],stamps[name]=read_pinned(Path(pin['source']),pin,PLAN['max_file_bytes'])
            require(sum(map(len,files.values()))<=9002784,'EXACT_SOURCE_PAYLOAD_CAP')
            compressed=pack(files,PLAN,CONTEXT)
            # No source bundle body lands on disk; original capture files remain read-only.
            sys.stdout.buffer.write(compressed);sys.stdout.buffer.flush()
            detail=dict(source_file_identities=stamps,requested_members=names,payload_bytes_read=sum(map(len,files.values())),
                compressed_bytes=len(compressed),compressed_SHA256=sha(compressed),tensor_decodes=0)
        passed=True
    except BaseException as error:
        failure=dict(type=type(error).__name__,reason=str(error)[:180])
    finally:
        if root is not None:
            # Work alarm re-arms original hard deadline before raising; tail never starts a new epoch.
            for kind in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(kind,signal.SIG_IGN)
            signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
            signal.setitimer(signal.ITIMER_REAL,max(.001,min(reuse.DEADLINE-time.time(),reuse.MONO_END-time.monotonic())))
            usage=resource.getrusage(resource.RUSAGE_SELF)
            result=dict(identity=identity,passed=passed,detail=detail,error=failure,finished_epoch=time.time(),
                endpoint_wall_seconds=time.monotonic()-started,self_user_CPU_seconds=usage.ru_utime,
                self_system_CPU_seconds=usage.ru_stime,self_maxrss_bytes=usage.ru_maxrss*1024,
                physical_IO='UNKNOWN',original694_state='UNKNOWN',GPU_GO=False,replay_allowed=False)
            save(root/'outcome.json',result);sync_directory(root)
            if MODE=='receiver':emit(dict(kind='RECOVERY_RESULT',passed=passed,member_denominator=2,error=failure,
                outcome_SHA256=sha(canonical(result)),original694_state='UNKNOWN',GPU_GO=False))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
