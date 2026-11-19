"""Fresh metadata observation of one attempted original request. Never launch/replay."""
import hashlib,json,math,os,re,signal,socket,stat,subprocess,time
from pathlib import Path

W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native-first-boundary-pair-v1')
LABEL=W.name
ORIGINAL_REQUEST='5843a07ed8dc4fe9aefd3189ddaec079'
ORIGINAL_PUBLICATION='89e6548a32328352670e8546ae48dfc58e3157d6'
ORIGINAL_MF='00c58f192b2c654542a5a730907c5b3503dd5f3c15ac2bf3662bf9efd09bf1b5'
PARENT='1087656'
LABELS=[LABEL+s for s in ('','-pre','-post','-audit')]
FILES={'intent.json':65536,'keeper-owner.json':16384,'controller-ready.json':16384,'outcome.json':1048576}
COLUMNS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocCPUS','AllocTRES','User']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def need(ok,code):
    if not ok:raise ValueError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def stamp(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_mode]
def unique(pairs):
    out={}
    for k,v in pairs:
        need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
    return out
def decode(raw):
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
def read_file(path,limit):
    before=path.lstat();need(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=limit,'REGULAR_FILE_BOUND')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);need(stamp(before)==stamp(opened),'OPENED_IDENTITY')
        parts=[];size=0
        while True:
            left();b=os.read(fd,min(65536,limit+1-size))
            if not b:break
            parts.append(b);size+=len(b);need(size<=limit,'READ_BOUND')
        raw=b''.join(parts)
        need(len(raw)==before.st_size and stamp(before)==stamp(os.fstat(fd))==stamp(path.lstat()),'STABLE_FD_PATH')
        return raw,dict(bytes=len(raw),SHA256=sha(raw),stat=stamp(before))
    finally:os.close(fd)
def boot():return sha(Path('/proc/sys/kernel/random/boot_id').read_bytes())
def left():
    value=min(ORIGIN+42-time.time(),START_MONO+42-ELAPSED_AT_START-time.monotonic())
    if not math.isfinite(value) or value<=0:raise TimeoutError('ORIGINAL_ORIGIN_EXHAUSTED')
    return value
def stop(signum,frame):raise TimeoutError('ORIGINAL_ORIGIN_SIGNAL')

def command(argv,limit=1048576):
    """Exactly one held child; timeout/error always reaps that child, never old owners."""
    child=None;out=b'';err=b'';error=None;term=False;kill=False;reaped=False
    try:
        left();child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        out,err=child.communicate(timeout=min(12,left()))
        need(len(out)<=limit and len(err)<=65536,'ACCOUNTING_OUTPUT_BOUND')
    except BaseException as e:error=type(e).__name__
    finally:
        if child is not None:
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM);term=True
                except ProcessLookupError:pass
                try:out,err=child.communicate(timeout=.5)
                except BaseException:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL);kill=True
                        except ProcessLookupError:pass
                    try:out,err=child.communicate(timeout=1)
                    except BaseException as e:error=error or type(e).__name__
            reaped=child.poll() is not None
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    row=dict(returncode=child.returncode if child else None,error_type=error,held_reaped=reaped,TERM=term,KILL=kill,
        stdout_bytes=len(out),stdout_SHA256=sha(out),stderr_bytes=len(err),stderr_SHA256=sha(err))
    row['complete']=bool(child and child.returncode==0 and error is None and reaped and not term and not kill)
    return (out if row['complete'] else b''),row

def owner_status(owner,intent):
    result=dict(state='UNKNOWN',original_process_terminal=None,proc_reads=0)
    need(owner.get('request_id')==ORIGINAL_REQUEST and owner.get('claim_commit')==ORIGINAL_PUBLICATION
         and owner.get('epoch')==intent['epoch'] and owner.get('deadline')==intent['deadline'],'OWNER_ORIGINAL_CONTEXT')
    pid=owner.get('PID');ticks=owner.get('startticks');pgid=owner.get('PGID')
    need(type(pid)is int and pid>1 and type(ticks)is int and ticks>0 and type(pgid)is int and pgid>1,'ORIGINAL_OWNER_TYPED')
    same_host=socket.gethostname()==owner.get('host');same_boot=boot()==owner.get('boot_id_SHA256')
    result.update(PID=pid,expected_startticks=ticks,same_host=same_host,same_boot=same_boot)
    if not same_host or not same_boot:return result
    path=Path('/proc')/str(pid)/'stat'
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    except FileNotFoundError:
        result.update(state='ORIGINAL_PID_ABSENT',original_process_terminal=True);return result
    try:
        before=os.fstat(fd);need(before.st_uid==os.getuid(),'ORIGINAL_OWNER_UID')
        a=os.read(fd,8193);need(len(a)<=8192,'PROC_STAT_BOUND');os.lseek(fd,0,0);b=os.read(fd,8193)
        result['proc_reads']=2
        def identity(raw):
            values=raw.rsplit(b') ',1)[1].split();return (int(raw.split(b' ',1)[0]),values[0].decode(),int(values[1]),int(values[2]),int(values[19]))
        x,y=identity(a),identity(b)
        need(x[0]==y[0]==pid and x[2:]==y[2:] and before.st_ino==os.fstat(fd).st_ino==path.stat().st_ino,'PROC_IDENTITY_STABLE')
        need(socket.gethostname()==owner['host'] and boot()==owner['boot_id_SHA256'],'OWNER_HOST_BOOT_POST')
        result.update(observed_startticks=y[4],observed_PGID=y[3],state_code=y[1],stat_SHA256=sha(b))
        if y[4]!=ticks:result.update(state='ORIGINAL_PID_REUSED',original_process_terminal=True)
        elif y[3]!=pgid:result['state']='ORIGINAL_PGID_MISMATCH_UNKNOWN'
        elif y[1]=='Z':result.update(state='ORIGINAL_PID_ZOMBIE',original_process_terminal=True)
        else:result.update(state='ORIGINAL_PID_LIVE',original_process_terminal=False)
        return result
    finally:os.close(fd)

def main():
    global START_MONO,ELAPSED_AT_START
    START_MONO=time.monotonic();ELAPSED_AT_START=time.time()-ORIGIN
    need(CLAIM['READ_GO'] is True and CLAIM['root_GO'] is True and CLAIM['max_invocations']==1,'ROOT_READ_GO')
    need(math.isfinite(ORIGIN) and 0<=ELAPSED_AT_START<42 and re.fullmatch('[0-9a-f]{32}',REQUEST)
         and re.fullmatch('[0-9a-f]{40}',PUBLICATION),'NEW_OBSERVATION_CONTEXT')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,left())
    available=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(available)>=2,'TWO_AVAILABLE_CPU');os.sched_setaffinity(0,available[:2])
    before=boot();host=socket.gethostname();values={};errors=[]
    result=dict(schema='MINICPM_ORIGINAL5843_FRESH_RO_V1',request_id=REQUEST,publication=PUBLICATION,epoch=ORIGIN,deadline=ORIGIN+45,
        original_request=ORIGINAL_REQUEST,original_publication=ORIGINAL_PUBLICATION,original_namespace=str(W),parent=PARENT,
        observer_host=host,observer_boot_SHA256=before,CPU_affinity=sorted(os.sched_getaffinity(0)),files={},errors=errors,
        original_execution_state='UNKNOWN',replay_allowed=False,release_allowed=False,Model_GO=False,new_model_runs=0,new_Slurm_steps=0)
    try:
        try:s=W.lstat();result['namespace']=dict(present=True,regular_directory=stat.S_ISDIR(s.st_mode),stat=stamp(s));need(stat.S_ISDIR(s.st_mode),'NAMESPACE_DIRECTORY')
        except FileNotFoundError:result['namespace']=dict(present=False)
        for name,cap in FILES.items():
            try:
                raw,pin=read_file(W/name,cap);result['files'][name]=dict(present=True,**pin)
                v=decode(raw);need(type(v)is dict,'JSON_OBJECT');values[name]=v
            except FileNotFoundError:result['files'][name]=dict(present=False)
            except Exception as e:
                errors.append(dict(phase='metadata',name=name,error_type=type(e).__name__))
                result['files'].setdefault(name,dict(present=None));result['files'][name]['coverage']='UNKNOWN'
        intent=values.get('intent.json');bound=False
        if intent:
            bound=(intent.get('request_id')==ORIGINAL_REQUEST and intent.get('claim_commit')==ORIGINAL_PUBLICATION
                and intent.get('source_manifest_SHA256')==ORIGINAL_MF and type(intent.get('epoch')) in (int,float)
                and math.isfinite(intent['epoch']) and intent.get('deadline')==intent['epoch']+1800)
        result['original_context_bound']=bound
        result['original_owner']=dict(state='UNKNOWN',original_process_terminal=None,proc_reads=0)
        if bound and 'keeper-owner.json' in values:
            try:result['original_owner']=owner_status(values['keeper-owner.json'],intent)
            except Exception as e:errors.append(dict(phase='owner',error_type=type(e).__name__))
        out=values.get('outcome.json')
        legacy_context=bool(bound and out and out.get('claim_commit')==ORIGINAL_PUBLICATION and out.get('epoch')==intent['epoch'] and out.get('deadline')==intent['deadline'])
        out_bound=bool(legacy_context and out.get('request_id')==ORIGINAL_REQUEST)
        result['outcome_request_field_present']=bool(out and 'request_id' in out)
        result['outcome_request_binding']='EXACT' if out_bound else 'UNKNOWN'
        result['outcome_legacy_epoch_publication_matches']=legacy_context
        result['outcome_context_bound']=out_bound
        if legacy_context:result['outcome_summary']={k:out.get(k) for k in ('keeper_passed','GPU_step_requested','GPU_launched','independent_model_qualified','formal_performance')}
        raw,receipt=command(['sacct','-n','-P','-j',PARENT,'--starttime','2026-11-11T00:00:00','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocCPUS,AllocTRES,User'])
        rows=[]
        for line in raw.decode('utf8',errors='strict').splitlines():
            cols=line.split('|')
            if len(cols)>=2 and cols[1] in LABELS:
                need(len(cols)==len(COLUMNS) and re.fullmatch(re.escape(PARENT)+r'\.\d+',cols[0]),'EXACT_ACCOUNTING_ROW');rows.append(cols)
        need(len(rows)<=4 and len({r[1] for r in rows})==len(rows),'MAX_ONE_ORIGINAL_HANDLE_PER_LABEL')
        result['accounting']=dict(columns=COLUMNS,rows=rows,query=receipt,exact_labels=LABELS)
        raw,receipt=command(['scontrol','show','step',PARENT,'-o'])
        active=[];unparsed=0
        for line in raw.decode('utf8',errors='strict').splitlines():
            fields=dict(re.findall(r'(\w+)=([^\s]+)',line));name=fields.get('Name',fields.get('StepName'))
            if name in LABELS:active.append({k:fields.get(k) for k in ('StepId','Name','StepName','State','Nodes','UserId','StartTime')})
            elif name is None:unparsed+=1
        result['active_steps']=dict(rows=active,query=receipt,unparsed_lines=unparsed,coverage='COMPLETE_PARENT_QUERY' if receipt['complete'] and not unparsed else 'UNKNOWN')
        terminal=bool(len(rows)==4 and all(r[2].split()[0].rstrip('+') in TERMINAL for r in rows))
        result['all_four_accounting_terminal']=terminal
        if (out_bound and terminal and receipt['complete'] and not unparsed and not active
                and result['original_owner'].get('original_process_terminal') is True):
            result['original_execution_state']='ORIGINAL_TERMINAL_METADATA_OBSERVED_NOT_NUMERICAL_QUALIFICATION'
    except BaseException as e:errors.append(dict(phase='reconcile',error_type=type(e).__name__))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result['same_observer_host_boot_after']=host==socket.gethostname() and before==boot()
        if errors or not result['same_observer_host_boot_after']:result['original_execution_state']='UNKNOWN'
        result['verified_fullfile_bytes']=sum(v.get('bytes',0) for v in result['files'].values())
        result['partial_or_rejected_read_bytes']='UNKNOWN' if errors else 0
        result['observation_monotonic_seconds']=time.monotonic()-START_MONO
        encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
        need(len(encoded.encode())<=131072,'BOUNDED_RESULT');print(encoded,flush=True)
    return 0 if not errors else 1

if __name__=='__main__':raise SystemExit(main())
