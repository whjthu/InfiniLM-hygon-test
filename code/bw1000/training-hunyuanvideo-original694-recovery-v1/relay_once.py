"""One permission-gated finite relay; prepare source only until explicit user exception."""
import argparse,base64,hashlib,json,math,os,queue,re,runpy,shlex,signal,struct,subprocess,sys,threading,time,uuid
from pathlib import Path,PurePosixPath
D=Path(__file__).resolve().parent
G=D.parents[2]
if str(D) not in sys.path:sys.path.insert(0,str(D))
from permission_gate import validate

HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
BOOTSTRAP=("import sys\n"
 "def exact(n):\n b=bytearray()\n while len(b)<n:\n  x=sys.stdin.buffer.read(n-len(b))\n  if not x: raise EOFError('short source envelope')\n  b.extend(x)\n return bytes(b)\n"
 "n=int.from_bytes(exact(8),'little')\nif not 0<n<=131072: raise ValueError('source envelope limit')\n"
 "exec(compile(exact(n),'published-694-recovery-endpoint','exec'))\n")

def need(ok,reason):
    if not ok:raise ValueError(reason)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def encode(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()

def save(path,value):
    with path.open('xb') as stream:stream.write(encode(value));stream.flush();os.fsync(stream.fileno())

def transport_state(returncode,forced,errors,alive):
    return 'UNKNOWN' if forced or returncode==255 or errors or alive else 'RETURNED'

def remaining(epoch,mono,reserve=0):
    left=min(epoch+300-time.time(),mono+300-time.monotonic())-reserve
    need(math.isfinite(left) and left>0,'ORIGINAL300_EPOCH_EXHAUSTED')
    return left

def endpoint_code(mode,context,plan,claim,approval,blobs):
    code='import sys,types\n'
    for name in ('reuse','nice_policy','atomic_files','permission_gate'):
        body=blobs[name+'.py'].decode()
        code+=f"m=types.ModuleType({name!r});m.__file__={('<published-pinned-source>/'+name+'.py')!r};sys.modules[{name!r}]=m\n"
        code+=f"exec(compile({body!r},m.__file__,'exec'),m.__dict__)\n"
    code+='PLAN='+repr(plan)+'\nCLAIM='+repr(claim)+'\nAPPROVAL='+repr(approval)+'\nCONTEXT='+repr(context)+'\nMODE='+repr(mode)+'\n'
    code+="__name__='__main__'\n"+blobs['endpoint.py'].decode()
    raw=code.encode();need(len(raw)<=131072,'BOUNDED_STDIN_SCRIPT')
    return struct.pack('<Q',len(raw))+raw

def ready_identity(value,mode,context):
    need(value.get('kind')=='RECOVERY_READY','EXACT_ENDPOINT_READY')
    identity=value['identity']
    for key in ('request_id','publication','epoch','deadline','source_manifest_SHA256','exception_record_SHA256'):
        need(identity[key]==context[key],'READY_CONTEXT')
    need(identity['mode']==mode and identity['original694_state']=='UNKNOWN'
        and type(identity['PID']) is int and identity['PID']>0
        and type(identity['startticks']) is int and identity['startticks']>0
        and isinstance(identity['host'],str) and bool(identity['host'])
        and re.fullmatch('[0-9a-f-]{36}',identity['boot_id'])
        and type(identity['uid']) is int,'READY_OWNED_HOST_PID')
    return identity

def main():
    epoch,mono=time.time(),time.monotonic()  # Includes Git, permission, source installation and both endpoints.
    parser=argparse.ArgumentParser()
    parser.add_argument('--publication',required=True);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--exception-receipt',required=True);parser.add_argument('--exception-receipt-sha256',required=True)
    args=parser.parse_args();entries=[];expired=threading.Event();local=None;timers=[]
    def left(reserve=0):
        need(not expired.is_set(),'ORIGINAL_RELAY_WORK_LIMIT')
        return remaining(epoch,mono,reserve)
    def force(entry):
        if entry['proc'].poll() is None:
            entry['forced_local_SSH_kill']=True
            try:entry['proc'].kill()
            except ProcessLookupError:pass
            except OSError as error:entry['errors'].append(type(error).__name__)
    def expire():
        expired.set()
        for item in list(entries):force(item)
    timer=threading.Timer(max(.001,min(epoch+260-time.time(),mono+260-time.monotonic())),expire);timer.daemon=True;timer.start();timers.append(timer)
    hard=threading.Timer(max(.001,min(epoch+300-time.time(),mono+300-time.monotonic())),lambda:os._exit(124));hard.daemon=True;hard.start();timers.append(hard)
    def interrupted(*_):raise TimeoutError('OWN_LOCAL_CONTROLLER_SIGNAL')
    catchable=[getattr(signal,name) for name in ('SIGTERM','SIGINT','SIGHUP') if hasattr(signal,name)]
    for kind in catchable:signal.signal(kind,interrupted)
    outcome=None
    def git(*extra):
        return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),*extra],timeout=min(12,left(40)))
    def start(mode,argv,payload,cap):
        need(len(subprocess.list2cmdline(argv))<30000,'WINDOWS_COMMANDLINE_BOUND');left(45)
        save(local/(mode+'-exec-intent.json'),dict(request_id=outcome['request_id'],publication=args.publication,
            source_manifest_SHA256=args.source_manifest_sha256,epoch=epoch,deadline=epoch+300,
            endpoint_script_envelope_bytes=len(payload),endpoint_script_envelope_SHA256=sha(payload),argv_not_recorded=True))
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        entry=dict(mode=mode,proc=proc,forced_local_SSH_kill=False,errors=[],ready_queue=queue.Queue(maxsize=1),
            buffers={},threads=[],input_thread=None)
        entries.append(entry)  # Strong Popen anchor, registered before any save/pipe/proc observation.
        outcome[mode]='SUBMITTED_REMOTE_OUTCOME_UNKNOWN'
        left(45)  # Covers the timer firing between Popen and registration.
        save(local/(mode+'-owned-ssh.json'),dict(PID=proc.pid,owned_unreaped_Popen=True,epoch=epoch,deadline=epoch+300,
            argv_not_recorded=True,request_id=outcome['request_id']))
        def read_output():
            try:
                first=proc.stdout.readline(16385)
                need(first.endswith(b'\n') and len(first)<=16384,'BOUNDED_READY_LINE')
                value=json.loads(first);entry['ready_queue'].put(('ready',value));data=bytearray()
                while True:
                    block=proc.stdout.read(65536)
                    if not block:break
                    need(len(data)+len(block)<=cap,'BOUNDED_ENDPOINT_OUTPUT');data.extend(block)
                entry['buffers']['stdout']=bytes(data)
            except BaseException as error:
                entry['errors'].append(type(error).__name__)
                try:entry['ready_queue'].put_nowait(('error',type(error).__name__))
                except queue.Full:pass
                force(entry)
            finally:proc.stdout.close()
        def read_error():
            data=bytearray()
            try:
                while True:
                    block=proc.stderr.read(65536)
                    if not block:break
                    need(len(data)+len(block)<=262144,'BOUNDED_PRIVATE_STDERR');data.extend(block)
                entry['buffers']['stderr']=bytes(data)
            except BaseException as error:entry['errors'].append(type(error).__name__);force(entry)
            finally:proc.stderr.close()
        for fn in (read_output,read_error):
            t=threading.Thread(target=fn,daemon=True);entry['threads'].append(t);t.start()
        send(entry,payload,close=(mode=='source'))
        return entry
    def send(entry,data,*,close):
        previous=entry['input_thread']
        if previous is not None:
            previous.join(timeout=min(2,left(35)));need(not previous.is_alive(),'SCRIPT_INPUT_NOT_FINISHED')
        def writer():
            try:
                entry['proc'].stdin.write(data);entry['proc'].stdin.flush()
                if close:entry['proc'].stdin.close()
            except (OSError,ValueError) as error:entry['errors'].append(type(error).__name__)
        thread=threading.Thread(target=writer,daemon=True);entry['input_thread']=thread;thread.start()
    def ready(entry,context):
        kind,value=entry['ready_queue'].get(timeout=left(40))
        need(kind=='ready','ENDPOINT_EXIT_BEFORE_READY');identity=ready_identity(value,entry['mode'],context)
        expected=plan['source_control'] if entry['mode']=='source' else str(Path(plan['target'])/plan['target_control_name']).replace('\\','/')
        need(identity['control_root']==expected,'EXACT_NEW_CONTROL_ROOT')
        entry['ready']=value;save(local/(entry['mode']+'-ready.json'),value)
        return value
    def finish(entry):
        entry['proc'].wait(timeout=left(20))
        for thread in entry['threads']+[entry['input_thread']]:
            if thread is not None:thread.join(timeout=left(15))
        need(not any(t.is_alive() for t in entry['threads']+[entry['input_thread']] if t),'ENDPOINT_STREAM_UNFINISHED')
        return entry['buffers'].get('stdout',b'')
    try:
        need(re.fullmatch('[0-9a-f]{40}',args.publication),'EXACT_PUBLICATION')
        mfraw=(D/'SOURCE_FILES.json').read_bytes();need(sha(mfraw)==args.source_manifest_sha256,'SOURCE_MF_PIN')
        need(git('show',args.publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json')==mfraw,'PUBLISHED_SOURCE_MF')
        blobs={}
        for name,pin in json.loads(mfraw)['files'].items():
            need(Path(name).name==name,'BARE_SOURCE_MEMBER');raw=(D/name).read_bytes()
            need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'SOURCE_BYTES_SHA');blobs[name]=raw
        plan=json.loads(blobs['PLAN.json']);claim=json.loads(blobs['CLAIM.json'])
        # Check GO before even reading the exception path or running SSH helper/config code.
        need(claim['root_GO'] is claim['CPU_GO'] is claim['transfer_GO'] is True,'ROOT_TRANSFER_GO_REQUIRED')
        approval_path=Path(args.exception_receipt)
        declared=claim['exception_record_path']
        need(isinstance(declared,str) and not PurePosixPath(declared).is_absolute()
            and '..' not in PurePosixPath(declared).parts and '\\' not in declared and ':' not in declared,'BOUNDED_PROJECT_EXCEPTION_PATH')
        need(approval_path==G/declared
            and not approval_path.is_symlink(),'EXACT_EXCEPTION_RECORD_PATH')
        need(approval_path.is_file() and 0<approval_path.stat().st_size<=65536,'BOUNDED_EXCEPTION_RECORD')
        raw=approval_path.read_bytes();need(0<len(raw)<=65536 and sha(raw)==args.exception_receipt_sha256==claim['exception_record_SHA256'],'EXPLICIT_EXCEPTION_SHA')
        need(git('show',args.publication+':'+approval_path.relative_to(G).as_posix())==raw,'PUBLISHED_USER_EXCEPTION')
        approval=json.loads(raw);validate(claim,approval,plan)
        need(git('diff','--cached','--name-only').strip()==b'','INDEX_NOT_EMPTY')
        need(git('status','--porcelain','--',D.relative_to(G).as_posix()).strip()==b'','OWN_SOURCE_NOT_CLEAN')
        need(sha(HELPER.read_bytes())==HELPER_SHA,'QUALIFIED_EXISTING_HYGON_HELPER')
        request=uuid.uuid4().hex
        context=dict(request_id=request,publication=args.publication,source_manifest_SHA256=args.source_manifest_sha256,
            epoch=epoch,deadline=epoch+300,exception_record_SHA256=sha(raw),original_request_id=plan['original_request_id'],original_state='UNKNOWN')
        local=G/'local-only'/plan['label'];local.mkdir(mode=0o700)
        save(local/'intent.json',dict(context,replay_allowed=False))
        outcome=dict(context,passed=False,original694_state='UNKNOWN',source='NOT_DISPATCHED',receiver='NOT_DISPATCHED',GPU_GO=False,replay_allowed=False)
        target_ssh=runpy.run_path(str(HELPER))['ssh']
        target_command='/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(BOOTSTRAP)
        receiver=start('receiver',[*target_ssh,target_command],endpoint_code('receiver',context,plan,claim,approval,blobs),65536)
        rready=ready(receiver,context);missing=rready['missing']
        need(missing==sorted(set(missing)) and set(missing)<=set(plan['members']),'EXACT_MISSING_MEMBERS')
        if missing:
            source_context=dict(context,requested_members=missing)
            source_command='/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -c '+shlex.quote(BOOTSTRAP)
            source_argv=['ssh','-T','-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config',
                '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','a100-27',source_command]
            source=start('source',source_argv,endpoint_code('source',source_context,plan,claim,approval,blobs),plan['max_compressed_bytes'])
            ready(source,source_context);payload=finish(source)
            need(source['proc'].returncode==0 and not source['errors'] and not source['forced_local_SSH_kill']
                and 0<len(payload)<=plan['max_compressed_bytes'],'SOURCE_NOT_RETURNED_VALID')
            outcome['source']='RETURNED0'
        else:payload=b'';outcome['source']='NOT_NEEDED_ALL_EXISTING_CORRECT'
        send(receiver,struct.pack('<Q',len(payload))+payload,close=True)
        result_raw=finish(receiver);result=json.loads(result_raw)
        need(receiver['proc'].returncode==0 and not receiver['errors'] and not receiver['forced_local_SSH_kill']
            and result['kind']=='RECOVERY_RESULT' and result['passed'] is True and result['member_denominator']==2
            and result['original694_state']=='UNKNOWN','RECEIVER_NOT_RETURNED_TWO_PIN_PASS')
        outcome.update(receiver='RETURNED0',passed=True,producer_result=result,independent_terminal_verified=False)
    except BaseException as error:
        if outcome is not None:outcome['error']=dict(type=type(error).__name__,reason=str(error)[:180])
        else:raise
    finally:
        timer.cancel()  # Keep original300 hard self-exit armed through cleanup and receipt fsync.
        for kind in catchable:signal.signal(kind,signal.SIG_IGN)
        # A killed SSH does not demonstrate remote exit. Persist the forced flag even after poll() changes.
        for entry in entries:
            try:
                proc=entry['proc']
                if proc.poll() is None:
                    # Never block acquiring stdin's lock while an owned writer is blocked in a full pipe.
                    if entry['input_thread'] is not None and entry['input_thread'].is_alive():force(entry)
                    else:
                        try:
                            if proc.stdin and not proc.stdin.closed:proc.stdin.close()
                        except OSError:pass
                    try:proc.wait(timeout=max(.001,min(3,remaining(epoch,mono,20))))
                    except subprocess.TimeoutExpired:
                        entry['forced_local_SSH_kill']=True
                        try:proc.terminate()
                        except ProcessLookupError:pass
                        try:proc.wait(timeout=max(.001,min(10,remaining(epoch,mono,10))))
                        except subprocess.TimeoutExpired:force(entry);proc.wait(timeout=max(.001,min(5,remaining(epoch,mono,2))))
                stderr=entry['buffers'].get('stderr',b'')
                state=transport_state(proc.returncode,entry['forced_local_SSH_kill'],entry['errors'],proc.poll() is None)
                record=dict(returncode=proc.returncode,remote_outcome=state,forced_local_SSH_kill=entry['forced_local_SSH_kill'],
                    reaped=proc.poll() is not None,stream_errors=entry['errors'],stderr_bytes=len(stderr),stderr_SHA256=sha(stderr),replay_allowed=False)
                save(local/(entry['mode']+'-transport.json'),record)
                (local/(entry['mode']+'-stderr.private')).write_bytes(stderr)
                if state=='UNKNOWN' and outcome is not None:outcome.update(passed=False,**{entry['mode']:'UNKNOWN'})
                elif outcome is not None and outcome[entry['mode']]=='SUBMITTED_REMOTE_OUTCOME_UNKNOWN':
                    outcome[entry['mode']]='RETURNED_EXIT'+str(proc.returncode)
            except BaseException as cleanup_error:
                # Isolate cleanup failure per child, always continue to the other strong owned anchor.
                force(entry)
                try:entry['proc'].wait(timeout=max(.001,min(2,remaining(epoch,mono))))
                except BaseException:pass
                if outcome is not None:
                    outcome.update(passed=False,**{entry['mode']:'UNKNOWN'})
                    outcome.setdefault('cleanup_errors',[]).append(type(cleanup_error).__name__)
        if outcome is not None:
            outcome['finished_epoch']=time.time();save(local/'outcome.json',outcome)
        hard.cancel()
    if outcome is not None:
        print(json.dumps({k:outcome[k] for k in ('request_id','passed','source','receiver','original694_state','GPU_GO')}))
        return 0 if outcome['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
