"""Injected original-bound once read-only reconciliation; no target mutation."""
import hashlib,json,math,os,re,resource,signal,socket,stat,subprocess,time
from pathlib import Path
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-history-value-only-common-AB-CPU-v1')
LABEL=W.name
EXPECTED=dict(request_id='094e92db7d214e318e2f08358dd06fd8',publication='6a1b000069379d8fe46f1086bb9c64d725a6d7e3',epoch=1794933214.2382555,source_manifest_SHA256='47aada58ed7e9c9068d5a5c5694a4f32512207618aeb4f71c54966691657eb14',claim_SHA256='5cf8581d5046fdcf50388042b77d2d5221c101e1cefb72cd0d551f10428fb9a4')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
COLUMNS='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    def bad(x):raise ValueError('NONFINITE_JSON')
    def real(x):
        value=float(x);need(math.isfinite(value),'NONFINITE_JSON');return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)
def main():
    started=time.monotonic();allowance=EPOCH+90-time.time();need(0<allowance<=90,'ORIGIN_INVALID')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));cores=sorted(os.sched_getaffinity(0));need(cores,'CPU_AFFINITY');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,allowance)
    q=dict(schema='GLM474-commonAB-original-reconcile-v1',reconcile_request_id=REQUEST,original=EXPECTED,epoch=EPOCH,deadline=EPOCH+90,state='UNKNOWN',replay_allowed=False,files={},errors=[],accounting=None,producer_passed=None,Model_GO=False,GPU_GO=False,Native_compile_GO=False,remote_writes=0,observer_host=socket.gethostname())
    docs={};total=0
    def left():
        n=min(EPOCH+90-time.time(),allowance-(time.monotonic()-started));need(n>2,'ORIGIN_EXHAUSTED');return n
    def read(name,cap):
        nonlocal total
        p=W/name
        try:
            left();s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'SMALL_FILE_BOUND')
            with p.open('rb') as f:
                a=os.fstat(f.fileno());raw=f.read(cap+1);b=os.fstat(f.fileno())
            ident=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
            need(len(raw)<=cap and ident(s)==ident(a)==ident(b)==ident(p.stat()),'STABLE_FD')
            total+=len(raw);need(total<=40*1024**2,'TOTAL_BYTES_BOUND')
            q['files'][name]=dict(presence='PRESENT',bytes=len(raw),SHA256=H(raw));docs[name]=strict(raw)
        except FileNotFoundError:q['files'][name]=dict(presence='ABSENT',execution_implication='UNKNOWN')
        except (ValueError,OSError,UnicodeError) as error:
            q['files'].setdefault(name,dict(presence='UNKNOWN'))['parse_or_read_error_type']=type(error).__name__
    try:
        # Fixed six originals; no directory listing or log/body transfer.
        for name,cap in [('install-intent.json',262144),('control-state.json',65536),('intent.json',262144),('outcome.json',16*1024**2),('owned-srun.json',65536),('build-result.json',16*1024**2)]:read(name,cap)
        install=docs.get('install-intent.json',{});control=docs.get('control-state.json',{})
        bound=lambda d:isinstance(d,dict) and all(d.get(k)==v for k,v in EXPECTED.items())
        q['install_scope_bound']=bound(install);q['control_scope_bound']=bound(control)
        q['exec_request_attempted']=control.get('exec_request_attempted') if bound(control) and type(control.get('exec_request_attempted')) is bool else None
        if q['install_scope_bound'] and q['control_scope_bound'] and q['exec_request_attempted'] is False:q['state']='NOT_SUBMITTED'
        intent=docs.get('intent.json',{});q['exec_intent_scope_bound']=bound(intent)
        if q['exec_intent_scope_bound']:
            q['intent_summary']={k:intent.get(k) for k in ('deadline','inside_deadline','work_deadline','node','Slurm_seconds','inside_seconds','GPU_cards','maximum_invocations')}
        outcome=docs.get('outcome.json')
        if isinstance(outcome,dict):
            q['outcome_control_hash_matches']=outcome.get('control_state_SHA256')==q['files'].get('control-state.json',{}).get('SHA256')
            q['outcome_summary']={k:outcome.get(k) for k in ('passed','rc','epoch','deadline','observed_epoch','client_timeout','current_step_outcome','build_SHA256','error_type')}
            q['producer_passed']=outcome.get('passed')
            if 'error_reason' in outcome:
                q['outcome_summary']['error_reason_SHA256']=H(str(outcome['error_reason']).encode());q['outcome_summary']['error_reason']=str(outcome['error_reason'])[:240]
        build=docs.get('build-result.json')
        if isinstance(build,dict):
            q['build_summary']={k:build.get(k) for k in ('passed','step','node','planned','actual','attempted','phase','elapsed_s','error_type')}
            q['build_summary']['error_SHA256']=H(str(build.get('error','')).encode());q['build_summary']['error_reason']=str(build.get('error',''))[:240]
            q['build_matches_outcome_SHA']=isinstance(outcome,dict) and outcome.get('build_SHA256')==q['files']['build-result.json'].get('SHA256')
        owned=docs.get('owned-srun.json');q['owned_process']=dict(state='UNKNOWN',alive=None)
        if isinstance(owned,dict):
            own=q['owned_process'];own.update({k:owned.get(k) for k in ('host','PID','PGID','startticks','boot_id','exited_before_identity_observation')})
            same_host=owned.get('host')==q['observer_host'];own['same_host']=same_host
            if same_host:
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();own['same_boot']=owned.get('boot_id')==boot
                if own['same_boot'] and type(owned.get('PID')) is int and owned['PID']>1 and type(owned.get('startticks')) is int:
                    try:
                        row=Path('/proc',str(owned['PID']),'stat').read_text().rsplit(') ',1)[1].split()
                        same_ticks=int(row[19])==owned['startticks'];live=same_ticks and row[0]!='Z';own.update(state='LIVE_SAME_IDENTITY' if live else 'ZOMBIE_ORIGINAL_NOT_LIVE' if same_ticks else 'PID_REUSED_ORIGINAL_ABSENT',alive=live,same_startticks=same_ticks)
                    except FileNotFoundError:own.update(state='ORIGINAL_PID_ABSENT',alive=False)
                    except (OSError,ValueError,IndexError) as error:own['read_error_type']=type(error).__name__
        # One original parent accounting query, no retry/poll/second host.
        q['accounting_query_count']=1
        try:
            done=subprocess.run(['sacct','-n','-P','-j','1103606','-o',COLUMNS],stdin=subprocess.DEVNULL,capture_output=True,timeout=min(15,left()-1),check=False)
            need(len(done.stdout)<=256*1024 and len(done.stderr)<=65536,'SACCT_OUTPUT_BOUND')
            q['accounting_receipt']=dict(rc=done.returncode,stdout_bytes=len(done.stdout),stdout_SHA256=H(done.stdout),stderr_bytes=len(done.stderr),stderr_SHA256=H(done.stderr),columns=COLUMNS)
            need(done.returncode==0,'SACCT_FAILED');rows=[s.split('|') for s in done.stdout.decode().splitlines() if s.strip()]
            need(all(len(r)==7 for r in rows),'SACCT_SCHEMA');matches=[r for r in rows if r[1]==LABEL];need(len(matches)<=1,'DUPLICATE_EXACT_LABEL')
            if matches:
                row=matches[0];need(re.fullmatch(r'1103606\.[0-9]+',row[0]) and row[5]=='f13r1n05','EXACT_HANDLE_NODE')
                q['accounting']=row;q['accounting_terminal']=row[2].split()[0] in TERMINAL
                if q['state']=='NOT_SUBMITTED':q['state']='UNKNOWN';q['errors'].append('CONTROL_ACCOUNTING_CONFLICT')
                elif q['control_scope_bound'] and q['install_scope_bound'] and q['exec_request_attempted'] is True:
                    q['state']='TERMINAL' if q['accounting_terminal'] else 'ACTIVE'
            else:q['accounting_terminal']=None
        except (ValueError,OSError,UnicodeError,subprocess.SubprocessError) as error:q['errors'].append('ACCOUNTING_'+type(error).__name__)
    except BaseException as error:q['errors'].append(type(error).__name__);q['state']='UNKNOWN'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['read_bytes']=total;q['elapsed_s']=time.monotonic()-started;q['actual_CPU_affinity']=sorted(os.sched_getaffinity(0));q['AS_hard_limit_bytes']=512*1024**2
        need(len(json.dumps(q).encode())<=65536,'SUMMARY_BOUND');print(json.dumps(q,allow_nan=False),flush=True)
if __name__=='__main__':main()
