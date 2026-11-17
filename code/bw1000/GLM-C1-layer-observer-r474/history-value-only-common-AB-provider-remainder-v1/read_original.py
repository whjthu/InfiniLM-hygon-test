"""Only39 pending provider hashes and seven saved ELF texts. No executable tools."""
import argparse,hashlib,json,math,os,re,resource,signal,stat,time
from pathlib import Path
D=Path(__file__).resolve().parent
OUT=Path('/refactor/results/GLM474-history-value-only-common-AB-provider-remainder-v1')
OLD=Path('/refactor/results/GLM474-history-value-only-common-AB-CPU-reader-v1')
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def strict(raw):
    def pairs(items):
        q={}
        for k,v in items:need(k not in q,'DUPLICATE_JSON_KEY');q[k]=v
        return q
    def bad(x):raise ValueError('NONFINITE_JSON')
    def real(x):
        v=float(x);need(math.isfinite(v),'NONFINITE_JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)
def ident(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def main(a):
    begin=time.monotonic();available=a.reader_epoch+180-time.time();need(0<available<=180,'ORIGIN_INVALID')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    cores=sorted(os.sched_getaffinity(0));need(cores,'CPU_EMPTY');os.sched_setaffinity(0,set(cores[:4]))
    def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,available)
    need(D==OUT/'src' and OUT.is_dir() and not OUT.is_symlink() and OUT.stat().st_uid==os.getuid(),'OWN_NAMESPACE')
    need(not any((OUT/n).exists() or (OUT/n).is_symlink() for n in ('reader-start.json','report-private.json')),'PRIOR_READER_OUTPUT')
    owns=False;q=dict(schema='GLM474-commonAB-provider-remainder-v1',request_id=a.request_id,publication=a.publication,original_epoch=a.epoch,reader_epoch=a.reader_epoch,passed=False,execution_state='UNKNOWN',CPU_ELF_qualified=False,GPU_GO=False,Model_GO=False,component_qualified=False,replay_allowed=False,files={},provider_preflight={},new_provider_count=0,new_provider_hashed_bytes=0,saved_output_count=0,reused_provider_count=32,reused_object_count=64,new_tool_executions=0,new_ldd=0,new_compile=0,new_Native_dlopen=0,new_device_compute=0,new_driver_queries=0,phase='admission',failures=[],apptainer_setup_implicit_SDK_driver='UNKNOWN',AS_hard_limit_bytes=3*1024**3,CPU_affinity=sorted(os.sched_getaffinity(0)))
    def left():
        n=min(a.reader_epoch+180-time.time(),available-(time.monotonic()-begin));need(n>2,'ORIGIN_EXHAUSTED');return n
    def small(p,cap=4*1024**2,expected=None,bytes_expected=None):
        left();p=Path(p);s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'SMALL_FILE_BOUND')
        with p.open('rb') as f:
            first=os.fstat(f.fileno());raw=f.read(cap+1);last=os.fstat(f.fileno())
        need(ident(s)==ident(first)==ident(last)==ident(p.stat()) and len(raw)<=cap,'SMALL_STABLE_FD')
        sha=H(raw);need(expected is None or sha==expected,'SMALL_FILE_PIN');need(bytes_expected is None or len(raw)==bytes_expected,'SMALL_FILE_BYTES')
        q['files'][str(p)]=dict(bytes=len(raw),SHA256=sha);return raw
    try:
        mf=strict(small(D/'SOURCE_FILES.json',65536,a.reader_manifest_sha256))
        for name,pin in mf.items():
            need(len(Path(name).parts)==1 and not Path(name).is_absolute(),'OWN_MEMBER')
            small(D/name,262144,pin['SHA256'],pin['bytes'])
        claim=strict(small(D/'CLAIM.json'));need(claim['ROOT_GO'] and claim['CPU_read_GO'] and not claim['GPU_GO'] and not claim['Native_compile_GO'],'NOT_APPROVED')
        ri=strict(small(OUT/'install-intent.json'))
        need(ri['original_request_id']==a.request_id and ri['original_publication']==a.publication and ri['original_epoch']==a.epoch and ri['reader_epoch']==a.reader_epoch and ri['reader_deadline']==a.reader_epoch+180 and ri['source_manifest_SHA256']==a.reader_manifest_sha256 and ri['claim_SHA256']==H((D/'CLAIM.json').read_bytes()) and ri['original_outcome_SHA256']==a.outcome_sha256,'INSTALL_CONTEXT')
        with (OUT/'reader-start.json').open('x') as f:owns=True;json.dump(ri,f,allow_nan=False)
        P=strict(small(D/'EVIDENCE_PINS.json'));parent=P['parent_report']
        need(parent['SHA256']=='582471741e4fd30ec318c634327753d393236e1bf46284c8c58d5a875ca2f06a' and parent['request_id']==a.request_id and parent['publication']==a.publication and parent['original_epoch']==a.epoch,'PRIOR_READER_CONTEXT')
        need(parent['passed'] is False and parent['failures']==[dict(stage=arm+'-ELF',error_type='ValueError',code='FILE_BOUND') for arm in ('A','B')],'PRIOR_FAILURE_PRESERVED')
        need(parent['independently_verified']==dict(freshM=64,HOST_compile=50,GPU_compile=14,objects=64,links=0),'PRIOR_DENOMINATORS')
        row=parent['accounting'];need(row[0]=='1103606.179' and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n05','PRIOR_TERMINAL')
        need(len(P['reused_providers'])==32 and len(P['remaining_providers'])==39 and not(set(P['reused_providers'])&set(P['remaining_providers'])),'PROVIDER_PARTITION')
        need(len(P['saved_outputs'])==7 and len(P['reused_libraries'])==2 and len(P['NEEDED'])==14,'ELF_PIN_COUNTS')
        q.update(execution_state='TERMINAL',original_reader_failed_unchanged=parent,prior_evidence='Published extraction from exact prior report SHA;64M/64objects/2DSOs/32providers are historical verified facts, not rehashed by this reader.',dynamic_resolution='Original producer observation; no new loader call.')
        q['phase']='parse_seven_saved_outputs';texts={}
        for path,pin in P['saved_outputs'].items():
            need(Path(path).parent==OLD and Path(path).name.endswith('.stdout'),'SAVED_OUTPUT_PATH')
            texts[Path(path).name]=small(path,2*1024**2,pin['SHA256'],pin['bytes']).decode();q['saved_output_count']+=1
        def strong(text):
            result={}
            for line in text.splitlines():
                r=line.split()
                if len(r)>=3 and r[-2] in ('T','D','B','R'):
                    need(r[-1] not in result,'DUPLICATE_STRONG');result[r[-1]]=r[-2]
            return result
        base=strong(texts['reference-independent-nm.stdout']);need(len(base)==280,'BASELINE280')
        q['ELF_static']={}
        for arm in ('A','B'):
            syms=strong(texts[arm+'-independent-nm.stdout']);need(all(syms.get(k)==v for k,v in base.items()),'BASELINE_LOST')
            delta=set(syms)-set(base)
            need(not delta if arm=='A' else len(delta)==1 and all('AttentionPrepare' in s and 'expand_cached_values' in s for s in delta),'STRONG_DELTA')
            dyn=texts[arm+'-independent-dynamic.stdout'];need('SYMBOLIC' in dyn and re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',dyn)==P['NEEDED'],'NEEDED14_SYMBOLIC')
            table=texts[arm+'-independent-tls.stdout']
            for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current'):
                rows=[line.split() for line in table.splitlines() if line.rstrip().endswith(' '+name)]
                need(len({(r[1],r[6]) for r in rows if len(r)>6 and r[3]=='TLS' and r[6]!='UND'})==1,'TLS_SINGLETON')
            q['ELF_static'][arm]=dict(reused_DSO=P['reused_libraries'][arm],strong=len(syms),added=sorted(delta),NEEDED=14,TLS=2)
        q['phase']='prestat_all39';before={};total=0
        for path,pin in P['remaining_providers'].items():
            left();need(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',pin['SHA256']),'PROVIDER_PIN')
            try:
                s=Path(path).stat();regular=stat.S_ISREG(s.st_mode);q['provider_preflight'][path]=dict(actual_bytes=s.st_size,regular=regular,within_8GiB=s.st_size<=8*1024**3);before[path]=s;total+=s.st_size
            except OSError as e:q['provider_preflight'][path]=dict(error_type=type(e).__name__,regular=None,actual_bytes=None)
        q['actual_pending_bytes']=total
        need(len(before)==39 and all(stat.S_ISREG(s.st_mode) and 0<=s.st_size<=8*1024**3 for s in before.values()),'PROVIDER_TYPE_OR_8GIB_BOUND')
        need(total<=32*1024**3,'PROVIDER_TOTAL32GIB_BOUND')
        q['phase']='hash39_once'
        for path,pin in P['remaining_providers'].items():
            left();p=Path(path);s=before[path];h=hashlib.sha256();count=0
            with p.open('rb') as f:
                first=os.fstat(f.fileno());need(ident(first)==ident(s),'PROVIDER_OPEN_CHANGED')
                for block in iter(lambda:f.read(1024**2),b''):
                    left();count+=len(block);q['new_provider_hashed_bytes']+=len(block)
                    need(count<=8*1024**3 and q['new_provider_hashed_bytes']<=32*1024**3,'STREAM_BOUND');h.update(block)
                last=os.fstat(f.fileno())
            need(count==s.st_size and ident(s)==ident(last)==ident(p.stat()),'PROVIDER_STABLE_FD')
            sha=h.hexdigest();q['files'][path]=dict(bytes=count,SHA256=sha);need(sha==pin['SHA256'],'PROVIDER_SHA_DIFFERS');q['new_provider_count']+=1
        need(q['new_provider_count']==39 and q['saved_output_count']==7,'REMAINDER_COMPLETE')
        q.update(passed=True,CPU_ELF_qualified=True,remaining_ELF_links_completed=2,combined_provider_identity_count=71,phase='complete')
    except BaseException as error:q['failures'].append(dict(phase=q['phase'],error_type=type(error).__name__,code=str(error)[:120] if isinstance(error,ValueError) else 'IO_OR_BUDGET_FAILURE'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['elapsed_s']=time.monotonic()-begin;q['reader_process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;q['memory_scope']='AS hard cap is address space; reader RSS is not container/job peak.'
        q['remaining_provider_missing']=39-q['new_provider_count'];raw=(json.dumps(q,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<=262144,'REPORT_BOUND')
        if owns:
            with (OUT/'report-private.json').open('xb') as f:f.write(raw)
        print(json.dumps(dict(passed=q['passed'],execution_state=q['execution_state'],report_bytes=len(raw) if owns else 0,report_SHA256=H(raw) if owns else None,new_provider_count=q['new_provider_count'],saved_output_count=q['saved_output_count'],GPU_GO=False,Model_GO=False)))
    return q
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request-id',required=True);p.add_argument('--publication',required=True);p.add_argument('--epoch',required=True,type=float);p.add_argument('--reader-epoch',required=True,type=float);p.add_argument('--reader-manifest-sha256',required=True);p.add_argument('--outcome-sha256',required=True);a=p.parse_args()
    q=main(a);raise SystemExit(0 if q['execution_state']=='TERMINAL' else 1)
