"""Original180 static completion only. Never compile, container, ldd or Native."""
import base64,hashlib,json,math,os,re,resource,shlex,signal,socket,stat,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/GLM474-history-value-only-component-CPU-v1'
P=R/'results/GLM474-history-value-only-common-AB-CPU-v1'
O=R/'results/GLM474-history-value-only-component-completion-reader-v1'
PINS_SHA='25ddb52679f8f16d252a789807b32e5e7854c3a7841b411946de26dadaf8c6ec'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def ident(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def strict(raw):
    def pairs(items):
        o={}
        for k,v in items:need(k not in o,'DUPLICATE_JSON');o[k]=v
        return o
    def bad(v):raise ValueError('NONFINITE_JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def main():
    need(globals().get('ROOT_APPROVED') is True,'ROOT_APPROVAL_REQUIRED')
    need(re.fullmatch('[0-9a-f]{32}',REQUEST) and re.fullmatch('[0-9a-f]{40}',PUBLICATION),'IDENTIFIERS')
    start=time.monotonic();available=EPOCH+90-time.time();need(math.isfinite(available) and 0<available<=90,'ORIGINAL_EPOCH')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
    cores=sorted(c for c in os.sched_getaffinity(0) if c not in (0,1));need(len(cores)>=2,'AFFINITY');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGINAL_EPOCH_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,available)
    q=dict(schema='GLM474-component-original180-static-completion-v1',request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+90,passed=False,original_step='1103606.180',original_step_state='FAILED',original_exit='1:0',original_outcome_build_null=True,original_passed=False,historical_DSO_SHA256=None,source_dependency_object_passed=False,static_ELF_passed=False,failures=[],files={},tools={},commands=[],total_read_bytes=0,compile_calls=0,container_calls=0,Native_calls=0,ldd_calls=0,new_provider_hashes=0,GPU_GO=False,Model_GO=False,replay_allowed=False,CPU_affinity=sorted(os.sched_getaffinity(0)),AS_hard_limit_bytes=512*1024**2)
    owned=False
    def left(reserve=1):
        value=min(EPOCH+90-time.time(),available-(time.monotonic()-start))-reserve;need(value>0,'ORIGINAL_EPOCH_EXHAUSTED');return value
    def read(path,cap,pin=None,size=None):
        left();p=Path(path);s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap and (size is None or s.st_size==size),'FILE_BOUND')
        with p.open('rb') as f:
            a=os.fstat(f.fileno());raw=f.read(cap+1);z=os.fstat(f.fileno())
        need(len(raw)==s.st_size and ident(s)==ident(a)==ident(z)==ident(p.stat()),'STABLE_FD');sha=H(raw);need(pin is None or sha==pin,'FILE_PIN')
        q['total_read_bytes']+=len(raw);need(q['total_read_bytes']<=128*1024**2,'TOTAL_IO_BOUND');q['files'][str(p)]=dict(bytes=len(raw),SHA256=sha);return raw
    def command(label,argv):
        left(4);rec=dict(label=label,argv=argv,owned_child_reaped=False);q['commands'].append(rec);proc=None
        out=O/(label+'.stdout');err=O/(label+'.stderr')
        with out.open('xb') as fo,err.open('xb') as fe:
            try:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=fo,stderr=fe,start_new_session=True,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','LANG':'C','HOME':'/nonexistent'})
                rec['PID']=proc.pid;rec['host']=socket.gethostname();rec['PGID']=proc.pid
                try:rec['startticks']=int(Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])
                except FileNotFoundError:rec['startticks']=None
                rec['rc']=proc.wait(timeout=min(15,left(4)))
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        need(os.getpgid(proc.pid)==proc.pid,'OWN_TOOL_SESSION')
                        if rec.get('startticks') is not None:
                            now=int(Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19]);need(now==rec['startticks'],'OWN_TOOL_STARTTICKS')
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    proc.wait(timeout=3);rec['owned_child_reaped']=True
        stdout=read(out,2*1024**2);stderr=read(err,65536);rec.update(stdout_SHA256=H(stdout),stderr_SHA256=H(stderr),stdout_bytes=len(stdout),stderr_bytes=len(stderr));need(rec['rc']==0 and not stderr,'PASSIVE_COMMAND_FAILED');return stdout.decode()
    def dep(raw):
        head,colon,tail=raw.decode().replace('\\\n',' ').partition(':');names=set(shlex.split(tail));need(colon and names and all(x.startswith('/') for x in names),'DEPENDENCY_FORMAT');return names
    try:
        need(len(PINS_B64)<=8192,'PIN_INPUT_CAP');rawpins=base64.b64decode(PINS_B64,validate=True);need(H(rawpins)==PINS_SHA,'READER_PIN_MANIFEST');pins=strict(rawpins)
        os.umask(0o077);need(not O.exists(),'ONCE_NAMESPACE_NO_REPLAY');O.mkdir(mode=0o700);owned=True
        (O/'intent.json').write_text(json.dumps(dict(request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+90,PINS_SHA256=PINS_SHA))+'\n')
        old={name:strict(read(W/name,65536,p['SHA256'],p['bytes'])) for name,p in pins['original_files'].items()}
        install=old['install-intent.json'];control=old['control-state.json'];out=old['outcome.json'];build=old['build-result.json']
        for key,value in [('request_id',pins['original_request']),('publication',pins['publication']),('epoch',pins['epoch']),('source_manifest_SHA256',pins['source_manifest_SHA256']),('claim_SHA256',pins['claim_SHA256'])]:
            need(install[key]==value and control[key]==value,'ORIGINAL_CONTROL_BINDING')
        need(control['exec_request_attempted'] is True and out['passed'] is False and out['build'] is None and out['accounting']['row'][:4]==['1103606.180',W.name,'FAILED','1:0'] and out['epoch']==pins['epoch'],'ORIGINAL_FAILED_TERMINAL')
        need(build['step']=='1103606.180' and build['passed'] is False and build['error']=='EXACT9_COMPONENT_CABI' and build['actual']==dict(freshM=1,HOST_compile=1,GPU_compile=0,objects=1,links=1),'EXACT_FAILURE_STAGE')
        mf=strict(read(W/'src/SOURCE_FILES.json',65536,pins['source_manifest_SHA256']));need(mf==pins['source_files'],'PUBLISHED_SOURCE_MAP')
        sourceclaim=None
        for name,pin in mf.items():
            need(Path(name).name==name,'SOURCE_FILE_NAME');raw=read(W/'src'/name,65536,pin['SHA256'],pin['bytes'])
            if name=='CLAIM.json':sourceclaim=strict(raw)
        need(sourceclaim['expected_CABI']==pins['expected_CABI'],'ORIGINAL9_CABI')
        commands={c['label']:c for c in build['commands']};need(set(commands)==set(pins['command_SHA256']) and len(commands)==4,'EXACT_ORIGINAL_COMMANDS')
        for label,sha in pins['command_SHA256'].items():
            c=commands[label];need(H(json.dumps(c['argv'],separators=(',',':')).encode())==sha and c['rc']==0 and c['owned_child_reaped'] is True,'ORIGINAL_COMMAND_PIN')
        before=read(W/'component.before.d',1024**2,build['fresh_M']['SHA256']);after=read(W/'component.actual.d',1024**2,build['object']['MD_SHA256']);names=dep(before);need(names==dep(after) and len(names)==build['fresh_M']['count']==990,'ACTUAL_MD_EQUALS_M')
        observed=strict(read(W/'component.dependency-pins.json',1024**2,build['fresh_M']['pins_SHA256']));need(set(observed)==names,'EXACT990_DEPENDENCY_PINS')
        plan=strict(read(P/'src/PLAN.json',2*1024**2,'d5356d5b2bc60825537d544ee9f1d100e659f0569eb2521177a466acc7a82fea'))
        parent=strict(read(P/'build-result.json',2*1024**2,'6b098aaf201f1a45f64f4b8d68c40c5b32ac1ed4cfc2e8ea72ec283b568e1560'))
        mirror='/refactor/results/'+P.name+'/B/project-mirror';allowed=dict(plan['system_pins'])
        for path,sha in plan['project_pins'].items():
            sha=plan['B_overrides'].get(path,plan['common_override'].get(path,sha));allowed[mirror+path[len('/refactor'):]]=sha
            allowed[mirror+os.path.normpath(path)[len('/refactor'):]]=sha
        cpp='/refactor/results/'+W.name+'/src/component.cpp';allowed[cpp]=sourceclaim['component_source_SHA256']
        need(cpp in names and all(allowed.get(p)==sha for p,sha in observed.items()),'DEPENDENCY_MANIFEST_BINDING')
        need(all(not p.startswith('/refactor/') or p==cpp or p.startswith(mirror+'/') for p in names),'COHERENT_B_MIRROR_ONLY')
        aliases=[R/(mirror+x[len('/refactor'):])[len('/refactor/'):] for x in plan['prepare_aliases']]
        need(len(aliases)==11 and len({(p.stat().st_dev,p.stat().st_ino) for p in aliases})==1,'ELEVEN_ALIAS_IDENTITY')
        obj=read(W/'component.o',1024**2,build['object']['SHA256'],build['object']['bytes'])
        bso=P/'B/libGLM_history_value_B.so';read(bso,2*1024**2,pins['parent_B_SHA256'],1934848)
        q['source_dependency_object_passed']=True;q['dependency_body_scope']='Original180 producer actualbyte checks with independent M/MD/pins-to-originalPLAN closure; no new990headerbody hash.'
        nm=read(W/'component-nm.stdout',65536,pins['original_nm']['SHA256'],pins['original_nm']['bytes']).decode();strong={};addresses={}
        for line in nm.splitlines():
            row=line.split()
            if len(row)>=3 and row[-2] in ('T','D','B','R'):
                need(row[-1] not in strong,'DUPLICATE_OLD_STRONG');strong[row[-1]]=row[-2];addresses[row[-1]]=int(row[0],16)
        expected={n:'T' for n in pins['expected_CABI']};expected.update(pins['linker_markers']);need(strong==expected and len(strong)==12,'NINE_T_PLUS_EXACT_THREE_LINKER_MARKERS')
        lib=W/'libhistory_value_component.so';libraw=read(lib,2*1024**2);libsha=H(libraw);q['current_DSO']=dict(SHA256=libsha,bytes=len(libraw));del libraw,obj
        for tool in ('/usr/bin/nm','/usr/bin/readelf'):
            path=Path(tool);canonical=path.resolve(strict=True);raw=read(canonical,32*1024**2);version=command(path.name+'-version',[str(path),'--version'])
            q['tools'][tool]=dict(canonical=str(canonical),SHA256=H(raw),bytes=len(raw),version=version[:1024],identity_scope='New actual HOST tool identity; not container tool pin.');del raw
        undefined=command('component-imports',['/usr/bin/nm','-D','--undefined-only',str(lib)]);imports={x.split()[-1].split('@')[0] for x in undefined.splitlines() if x.split()}
        added=parent['libraries']['B']['new_strong'];need(len(added)==1 and added[0] in imports,'NEW_CPP_METHOD_IMPORT')
        need(any('AttentionPrepare' in x and 'expand_cached' in x and 'expand_cached_values' not in x for x in imports) and any('CommunicationGroupC' in x for x in imports) and any('get_rank_info' in x for x in imports) and 'modern_glm_fp8_runtime_clear_r809' in imports,'REAL_FACTORY_RANK_ORIGINAL_CLEANUP_IMPORTS')
        dynamic=command('component-dynamic',['/usr/bin/readelf','-d',str(lib)]);needed=re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',dynamic);containerB='/refactor/results/'+P.name+'/B/libGLM_history_value_B.so'
        need('SYMBOLIC' in dynamic and len(needed)==len(set(needed)) and sum(x in (containerB,bso.name) for x in needed)==1 and set(needed)<=set(plan['original169']['NEEDED'])|{containerB,bso.name},'B_ONLY_NEEDED')
        table=command('component-symbols',['/usr/bin/readelf','-Ws','--demangle',str(lib)]);dyn=False;current={};tls={}
        for line in table.splitlines():
            if line.startswith('Symbol table '):dyn="'.dynsym'" in line
            row=line.split()
            if dyn and len(row)>=8 and row[4]=='GLOBAL' and row[6]!='UND' and row[3] in ('FUNC','OBJECT','NOTYPE'):
                name=' '.join(row[7:]);need(name not in current,'DUPLICATE_DYNSYM_GLOBAL');current[name]=(int(row[1],16),row[3])
        need(set(current)==set(strong) and all(current[n][0]==addresses[n] for n in strong) and all(current[n][1]=='FUNC' for n in pins['expected_CABI']),'CURRENT_DYNSYM_EQUALS_ORIGINAL_NM')
        for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current','infinilm::models::glm_moe_dsa::v_on_demand_r400::completed_expand_calls'):
            rows=[s.split() for s in table.splitlines() if s.rstrip().endswith(' '+name)];defs={(x[1],x[6]) for x in rows if len(x)>6 and x[3]=='TLS' and x[6]!='UND'};need(len(defs)<=1,'DUPLICATE_TLS_DEFINITION');tls[name]=len(defs)
        read(lib,2*1024**2,libsha)
        for tool,pin in q['tools'].items():
            need(str(Path(tool).resolve(strict=True))==pin['canonical'],'HOST_TOOL_CANONICAL_CHANGED');read(pin['canonical'],32*1024**2,pin['SHA256'],pin['bytes'])
        q.update(passed=True,static_ELF_passed=True,strong_CABI=pins['expected_CABI'],allowed_linker_markers=pins['linker_markers'],NEEDED=needed,TLS_definition_counts=tls,imports=sorted(imports),CPU_probe_static_qualified=True,component_numerical_qualified=False)
    except BaseException as error:q['failures'].append(dict(error_type=type(error).__name__,code=str(error)[:150] if isinstance(error,ValueError) else 'READ_TOOL_OR_BUDGET_FAILURE'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['elapsed_s']=time.monotonic()-start;q['reader_process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        raw=(json.dumps(q,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<=131072,'REPORT_BOUND')
        if owned:(O/'report-private.json').write_bytes(raw)
        print(json.dumps(dict(request_id=REQUEST,passed=q['passed'],source_dependency_object_passed=q['source_dependency_object_passed'],static_ELF_passed=q['static_ELF_passed'],original180_state='FAILED1:0',report_bytes=len(raw),report_SHA256=H(raw),failures=q['failures'],GPU_GO=False,Model_GO=False),allow_nan=False),flush=True)
if __name__=='__main__':main()
