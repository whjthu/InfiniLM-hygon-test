"""One HOST readelf -Ws only; preserve prior180 and completion failures."""
import hashlib,json,math,os,re,resource,signal,stat,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
W=R/'GLM474-history-value-only-component-CPU-v1'
OLD=R/'GLM474-history-value-only-component-completion-reader-v1'
O=R/'GLM474-history-value-only-component-symbol-completion-v1'
REPORT_SHA='1336084635320e76d4aedb8b66e6b5a30e0dd3f55e7fa43a1b7c56b74cfb494a'
LIB_SHA='46bed77bf9dd0d9fca781e19491daeb4f4465a807afcffc0ab003773f0cf7140'
TOOL_SHA='09d6c369067fd11d80ea5ec5b4d58271f82b4a8e1fabce4b341637fa8402f54e'
NM_SHA='cdd6c245f63c3837d33faea80b330390018f3c06da16d9fe33ab0ea69280bd9c'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def main():
    need(globals().get('ROOT_APPROVED') is True,'ROOT_APPROVAL_REQUIRED');need(re.fullmatch('[0-9a-f]{32}',REQUEST) and re.fullmatch('[0-9a-f]{40}',PUBLICATION),'IDENTIFIERS')
    started=time.monotonic();available=EPOCH+60-time.time();need(math.isfinite(available) and 0<available<=60,'ORIGIN')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
    cores=sorted(c for c in os.sched_getaffinity(0) if c not in (0,1));need(len(cores)>=2,'AFFINITY');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,available)
    q=dict(schema='GLM474-original180-symbol-only-completion-v1',request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+60,passed=False,prior_completion_passed=False,original180_state='FAILED1:0',original_outcome_build_null=True,prior_completion_report_SHA256=REPORT_SHA,GPU_GO=False,Model_GO=False,Native_GO=False,replay_allowed=False,files={},failures=[],new_command_count=0,CPU_affinity=sorted(os.sched_getaffinity(0)),AS_hard_bytes=512*1024**2,total_read_bytes=0)
    owned=False;proc=None
    def left(reserve=1):
        n=min(EPOCH+60-time.time(),available-(time.monotonic()-started))-reserve;need(n>0,'ORIGIN_EXHAUSTED');return n
    def read(p,cap,pin=None,size=None):
        left();s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap and (size is None or s.st_size==size),'FILE_BOUND')
        with p.open('rb') as f:a=os.fstat(f.fileno());raw=f.read(cap+1);z=os.fstat(f.fileno())
        need(len(raw)==s.st_size and identity(s)==identity(a)==identity(z)==identity(p.stat()),'STABLE_FD');sha=H(raw);need(pin is None or sha==pin,'FILE_PIN');q['total_read_bytes']+=len(raw);need(q['total_read_bytes']<=8*1024**2,'TOTAL_BOUND');q['files'][str(p)]=dict(bytes=len(raw),SHA256=sha);return raw
    try:
        os.umask(0o077);need(not O.exists(),'ONCE_NAMESPACE_NO_REPLAY');O.mkdir(mode=0o700);owned=True
        (O/'intent.json').write_text(json.dumps(dict(request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+60))+'\n')
        old=json.loads(read(OLD/'report-private.json',131072,REPORT_SHA));need(old['request_id']=='105dc8b13a8a43baa73d03a4db39af0b' and old['passed'] is False and old['source_dependency_object_passed'] is True and old['failures']==[dict(error_type='ValueError',code='PASSIVE_COMMAND_FAILED')],'PRIOR_EXACT_FAILURE')
        commands=old['commands'];need([c['label'] for c in commands]==['nm-version','readelf-version','component-imports','component-dynamic','component-symbols'] and all(c['rc']==0 and c['owned_child_reaped'] for c in commands[:4]) and commands[-1]['rc']==1 and commands[-1]['owned_child_reaped'] and commands[-1]['argv'][1:3]==['-Ws','--demangle'],'PRIOR_ONLY_LAST_SYMBOL_FAILURE')
        need(old['current_DSO']==dict(SHA256=LIB_SHA,bytes=347760),'PRIOR_DSO_PIN')
        tool=Path('/usr/bin/readelf');need(str(tool.resolve(strict=True))=='/usr/bin/readelf' and old['tools'][str(tool)]['SHA256']==TOOL_SHA,'ACTUAL_HOST_TOOL');read(tool,1024**2,TOOL_SHA,643656)
        lib=W/'libhistory_value_component.so';read(lib,1024**2,LIB_SHA,347760)
        nm=read(W/'component-nm.stdout',65536,NM_SHA,45969).decode();expected={'glm_history_component_'+n:'T' for n in ('error','result','create','keys','load','finish','pair','guards','close')};expected.update({'__bss_start':'B','_edata':'D','_end':'B'});strong={};addresses={}
        for line in nm.splitlines():
            r=line.split()
            if len(r)>=3 and r[-2] in ('T','D','B','R'):need(r[-1] not in strong,'DUPLICATE_OLD_SYMBOL');strong[r[-1]]=r[-2];addresses[r[-1]]=int(r[0],16)
        need(strong==expected,'EXACT9_AND3')
        argv=[str(tool),'-Ws',str(lib)];q['command']=dict(argv=argv,owned_child_reaped=False);q['new_command_count']=1
        with (O/'symbols.stdout').open('xb') as fo,(O/'symbols.stderr').open('xb') as fe:
            try:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=fo,stderr=fe,start_new_session=True,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','LANG':'C','HOME':'/nonexistent'});q['command']['PID']=proc.pid
                try:q['command']['startticks']=int(Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])
                except FileNotFoundError:q['command']['startticks']=None
                q['command']['rc']=proc.wait(timeout=min(15,left(4)))
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        need(os.getpgid(proc.pid)==proc.pid,'OWN_TOOL_SESSION')
                        if q['command'].get('startticks') is not None:need(int(Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])==q['command']['startticks'],'OWN_TOOL_STARTTICKS')
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    proc.wait(timeout=3);q['command']['owned_child_reaped']=True
        raw=read(O/'symbols.stdout',2*1024**2);err=read(O/'symbols.stderr',65536);need(q['command']['rc']==0 and not err,'SYMBOL_COMMAND_FAILED');table=raw.decode();q['command'].update(stdout_SHA256=H(raw),stderr_SHA256=H(err))
        dyn=False;current={};all_tls={}
        for line in table.splitlines():
            if line.startswith('Symbol table '):dyn="'.dynsym'" in line
            r=line.split()
            if len(r)>=8 and r[0].endswith(':') and r[6]!='UND':
                name=r[7]
                if r[3]=='TLS':all_tls.setdefault(name,set()).add((r[1],r[6]))
                if dyn and r[4]=='GLOBAL' and r[3] in ('FUNC','OBJECT','NOTYPE'):need(name not in current,'DUPLICATE_DYNSYM');current[name]=(int(r[1],16),r[3])
        need(set(current)==set(strong) and all(current[n][0]==addresses[n] for n in strong) and all(current[n][1]=='FUNC' for n,t in strong.items() if t=='T'),'CURRENT_DYNSYM_EQUALS_ORIGINAL_NM')
        known=('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current','infinilm::models::glm_moe_dsa::v_on_demand_r400::completed_expand_calls');checks={}
        for name in known:
            mangled='_ZN'+''.join(str(len(part))+part for part in name.split('::'))+'E';defs=all_tls.get(mangled,set());need(len(defs)<=1,'DUPLICATE_KNOWN_TLS');checks[name]=dict(mangled=mangled,definitions=len(defs))
        need(all(len(defs)<=1 for defs in all_tls.values()),'DUPLICATE_TLS_OTHER')
        read(lib,1024**2,LIB_SHA,347760);need(str(tool.resolve(strict=True))=='/usr/bin/readelf','TOOL_CANONICAL_CHANGED');read(tool,1024**2,TOOL_SHA,643656)
        q.update(passed=True,CPU_probe_static_qualified=True,component_numerical_qualified=False,current_DSO_SHA256=LIB_SHA,strong_CABI=9,linker_markers=3,TLS=checks,all_TLS_symbol_count=len(all_tls),reused_scope='Original failedcompletion source/M/MD990/object/parentB,imports/NEEDED passed before its unsupportedoption; no rereadM/compile/providers/container.')
    except BaseException as e:q['failures'].append(dict(error_type=type(e).__name__,code=str(e)[:120] if isinstance(e,ValueError) else 'IO_TOOL_OR_BUDGET'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['elapsed_s']=time.monotonic()-started;raw=(json.dumps(q,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<=65536,'REPORT_BOUND')
        if owned:(O/'report-private.json').write_bytes(raw)
        print(json.dumps(dict(request_id=REQUEST,passed=q['passed'],report_SHA256=H(raw),report_bytes=len(raw),failures=q['failures'],original180_state='FAILED1:0',prior_completion_passed=False,GPU_GO=False),allow_nan=False),flush=True)
if __name__=='__main__':main()
