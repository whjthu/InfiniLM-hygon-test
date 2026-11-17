"""New common A/B recipe; all64 M precede64 new objects. Never load Native."""
import hashlib,json,math,os,posixpath,re,resource,shlex,signal,socket,stat,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def need(ok,code):
    if not ok:raise ValueError(code)

def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def cleanup_owned_command(proc,record):
    # The unreaped Popen child is the ownership anchor; never touch another session.
    if proc is None:return
    try:
        if proc.poll() is None:
            pgid=os.getpgid(proc.pid)
            need(pgid==proc.pid==record['owned_pgrp'],'OWN_COMPILER_SESSION_DIFFERS')
            if record.get('startticks') is not None:
                current=Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()
                need(int(current[19])==record['startticks'],'OWN_COMPILER_STARTTICKS_DIFFERS')
            os.killpg(pgid,signal.SIGTERM)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Same unreaped child; recheck PGID before escalating its own session.
                need(os.getpgid(proc.pid)==proc.pid,'OWN_COMPILER_SESSION_DIFFERS')
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
        else:proc.wait(timeout=0)
    except ProcessLookupError:
        proc.wait(timeout=3)
    finally:
        record['owned_child_reaped']=proc.returncode is not None


def main():
    J=read(W/'intent.json');P=read(D/'PLAN.json');C=read(D/'CLAIM.json');anchor=time.monotonic();initial=J['work_deadline']-time.time()
    q=dict(passed=False,step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),node=socket.gethostname().split('.')[0],planned=dict(freshM=64,HOST_compile=50,GPU_compile=14,objects=64,links=2),actual=dict(freshM=0,HOST_compile=0,GPU_compile=0,objects=0,links=0),attempted=dict(freshM=0,HOST_compile=0,GPU_compile=0,links=0),objects={},fresh_M={},commands=[],phase='admission',new_recipe=True,old169_qualification_inherited=False,Native_dlopen=False,device_compute_calls=0,device_driver_queries=0,SDK_explicit_calls=0,implicit_ldd_initialization='UNKNOWN',GPU_GO=False,Model_GO=False)
    def save():
        q['elapsed_s']=time.monotonic()-anchor
        tmp=W/'build-result.tmp';tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');os.replace(tmp,W/'build-result.json')
    def left(reserve=0):
        value=min(initial-(time.monotonic()-anchor),J['work_deadline']-time.time(),J['deadline']-time.time())-reserve
        if not math.isfinite(value) or value<=0:raise TimeoutError('ORIGINAL_WORK_EPOCH_EXHAUSTED')
        return value
    def stable(path,expected):
        left(5);p=Path(path);a=p.stat();need(stat.S_ISREG(a.st_mode) and a.st_size<=16*1024**2,'SOURCE_FILE_TYPE_SIZE')
        with p.open('rb') as f:
            s=os.fstat(f.fileno());need(identity(a)==identity(s),'SOURCE_OPEN_IDENTITY');b=f.read(16*1024**2+1);t=os.fstat(f.fileno())
        need(identity(s)==identity(t)==identity(p.stat()) and H(b)==expected,'SOURCE_PIN_OR_STABILITY');return b
    def digest(path):
        left(5);p=Path(path);a=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            s=os.fstat(f.fileno());need(identity(a)==identity(s),'HASH_OPEN_IDENTITY')
            for b in iter(lambda:f.read(1024**2),b''):left(5);h.update(b)
            t=os.fstat(f.fileno())
        need(identity(s)==identity(t)==identity(p.stat()),'HASH_STABILITY');return h.hexdigest()
    def command(label,args,limit=150):
        left(10);out=W/(label+'.stdout');err=W/(label+'.stderr')
        proc=None;rec=dict(label=label,argv=args,PID=None,owned_child_reaped=False)
        q['commands'].append(rec)
        with out.open('xb') as fo,err.open('xb') as fe:
            try:
                proc=subprocess.Popen(args,stdout=fo,stderr=fe,start_new_session=True)
                rec.update(PID=proc.pid,owned_pgrp=proc.pid,identity_basis='unreaped_owned_Popen_child')
                try:
                    row=Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()
                    rec['startticks']=int(row[19]);need(int(row[2])==proc.pid,'OWN_COMPILER_SESSION_DIFFERS')
                except FileNotFoundError:
                    rec['startticks']=None
                rc=proc.wait(timeout=min(limit,left(10)))
            except BaseException as error:
                rec['error_type']=type(error).__name__
                rec['timed_out']=isinstance(error,(TimeoutError,subprocess.TimeoutExpired))
                raise
            finally:
                try:cleanup_owned_command(proc,rec)
                except BaseException as cleanup_error:
                    rec['cleanup_error_type']=type(cleanup_error).__name__
                    raise
        rec.update(rc=rc,stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,stdout_SHA256=digest(out),stderr_SHA256=digest(err));save()
        need(rc==0,'COMMAND_FAILED:'+label);need(out.stat().st_size<=16*1024**2 and err.stat().st_size<=16*1024**2,'COMMAND_OUTPUT_BOUND')
        return out.read_text(errors='strict')+'\n'+err.read_text(errors='strict')
    def deps(path):
        head,sep,body=Path(path).read_text().replace('\\\n',' ').partition(':');names=set(shlex.split(body))
        need(sep and names and all(p.startswith('/') for p in names),'DEPENDENCY_SYNTAX');return names
    def nm(path,label):
        text=command(label,['/usr/bin/nm','-D','--defined-only',str(path)],30)
        return {a[-1]:a[-2] for s in text.splitlines() if len(a:=s.split())>=3 and a[-2] in 'TDBR'}
    def elf(path,label):
        text=command(label,['/usr/bin/readelf','-d',str(path)],30);need('SYMBOLIC' in text,'SYMBOLIC_REQUIRED')
        return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',text)
    def providers(path,label):
        text=command(label,['/usr/bin/ldd','-r',str(path)],40);need('not found' not in text and 'undefined symbol' not in text,'PROVIDER_UNRESOLVED')
        paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M));return {p:digest(p) for p in sorted(paths)}
    try:
        need(str(W)==P['root'] and q['node']==C['node']=='f13r1n05' and re.fullmatch(r'1103606\.[0-9]+',q['step']),'OWN_CPU_SCOPE')
        need(C['ROOT_GO'] and C['CPU_GO'] and C['Native_compile_GO'] and not C['GPU_GO'] and not C['Model_GO'],'CPU_GO_REQUIRED')
        need(J['deadline']==J['epoch']+1800 and J['epoch']<time.time()<J['work_deadline']<=J['epoch']+1680,'SINGLE_EPOCH')
        need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),'GPU_HIDDEN')
        need(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'NO_PRELOAD')
        for n,p in read(D/'SOURCE_FILES.json').items():need(digest(D/n)==p['SHA256'] and (D/n).stat().st_size==p['bytes'],'OWN_SOURCE_PIN')
        need(digest(D/'CLAIM.json')==J['claim_SHA256'] and digest(D/'SOURCE_FILES.json')==J['source_manifest_SHA256'],'INTENT_SOURCE_PIN')
        tools=read(D/'TOOLCHAIN.json')
        for p,row in tools['tools'].items():need(str(Path(p).resolve())==row['canonical'] and digest(p)==row['sha256'],'TOOLCHAIN_PIN')
        for k,v in tools['required_environment'].items():need(os.environ.get(k)==v,'TOOLCHAIN_ENV')
        original=P['original169'];need(digest(original['parent_library'])==original['parent_library_SHA256'],'REFERENCE_DSO_PIN')
        baseline=nm(original['parent_library'],'reference-nm');need(len(baseline)==280,'REFERENCE_STRONG280')
        # Exact known metadata/source paths only; no treewalk or NFS inventory.
        q['phase']='preflight_source_bytes';raws={};total=0
        for path,h in P['project_pins'].items():
            effective=P['common_override'].get(path,h);blob=P['source_blobs'].get(effective)
            b=stable(D/blob,effective) if blob else stable(path,effective)
            total+=len(b);need(total<=256*1024**2,'PROJECT_STAGE_TOTAL_BOUND');raws[path]=b
        for path,h in P['system_pins'].items():need(digest(path)==h,'SYSTEM_HEADER_PIN')
        q.update(source_bytes_preflight_passed=True,project_registered=len(raws),project_read_bytes=total);save()
        selected={};prefixes={};mirrors={};effective_by_arm={};q['phase']='owned_mirrors'
        for arm in ('A','B'):
            root=W/arm;root.mkdir(mode=0o700);(root/'objects').mkdir();mirror=root/'project-mirror';mirror.mkdir();mirrors[arm]=mirror;effective={};shared={}
            for path,base in raws.items():
                h=P['B_overrides'].get(path) if arm=='B' else None
                b=stable(D/P['source_blobs'][h],h) if h else base
                relative=posixpath.normpath(path)[len('/refactor/'):];need(path.startswith('/refactor/') and '..' not in Path(relative).parts,'MIRROR_PATH')
                dest=mirror/relative;dest.parent.mkdir(parents=True,exist_ok=True);sha=H(b)
                if dest.exists():need(digest(dest)==sha,'NORMALIZED_ALIAS_CONFLICT')
                elif path.endswith(('.h','.hpp')) and sha in shared:os.link(shared[sha],dest)
                else:
                    with dest.open('xb') as f:f.write(b)
                    if path.endswith(('.h','.hpp')):shared[sha]=dest
                effective[str(dest)]=sha
                effective[str(mirror)+path[len('/refactor'):]]=sha
            headerpaths=[mirror/posixpath.normpath(p)[len('/refactor/'):] for p in P['prepare_aliases']]
            need(len(headerpaths)==11 and len({(p.stat().st_dev,p.stat().st_ino) for p in headerpaths})==1,'ALL_PREPARE_ALIASES_ONE_INODE')
            effective_by_arm[arm]=effective
            inc=['-I'+(str(mirror)+p[len('/refactor'):] if p.startswith('/refactor/') else p) for p in P['include_roots']]
            for u in P['units']:
                src=str(mirror)+posixpath.normpath(u['source'])[len('/refactor'):]
                compiler='/usr/bin/g++' if u['kind']=='HOST' else '/opt/dtk/cuda/cuda/bin/nvcc'
                prefixes[arm,u['name']]=([compiler]+P['new_common_flags'][u['kind']]+inc,src)
        # ALL64 preprocess commands must succeed and their entireactual graph be pinned.
        q['phase']='all64_fresh_M'
        for arm in ('A','B'):
            for u in P['units']:
                prefix,src=prefixes[arm,u['name']];dep=W/arm/(u['name']+'.before.d');obj=W/arm/'objects'/(u['name']+'.o')
                q['attempted']['freshM']+=1
                command(arm+'-'+u['name']+'-M',prefix+['-M','-MF',str(dep),'-MT',str(obj),src],60)
                names=deps(dep);allowed=dict(P['system_pins'],**effective_by_arm[arm]);need(src in names and names<=set(allowed),'FRESH_M_UNREGISTERED')
                need(not any(p.startswith('/refactor/') and not p.startswith(str(mirrors[arm])+'/') for p in names),'QUOTE_INCLUDE_ESCAPED_MIRROR')
                actual={p:digest(p) for p in names};need(all(actual[p]==allowed[p] for p in names),'FRESH_M_BYTES')
                selected[arm,u['name']]=actual;q['actual']['freshM']+=1
                dp=W/arm/(u['name']+'.dependency-pins.json');dp.write_text(json.dumps(actual,sort_keys=True)+'\n')
                q['fresh_M'][arm+'-'+u['name']]=dict(count=len(names),MD_SHA256=digest(dep),pins_file=str(dp),pins_SHA256=digest(dp))
        # A/B graphs can differonlycandidateNoV/helper; samephysicalrelativeheaders otherwise.
        changes=set(P['B_overrides'])|{P['helper_path']}
        for u in P['units']:
            def normalized(arm):return {('/refactor'+p[len(str(mirrors[arm])):] if p.startswith(str(mirrors[arm])) else p):h for p,h in selected[arm,u['name']].items()}
            aa,bb=normalized('A'),normalized('B')
            need(set(aa)^set(bb)<=changes,'AB_UNEXPECTED_DEPENDENCY_DELTA')
            need(all(aa[p]==bb[p] or p in changes for p in set(aa)&set(bb)),'AB_UNEXPECTED_HEADER_DELTA')
        need(q['actual']['freshM']==64,'ALL64_M_BEFORE_COMPILE');q['all64_M_before_compile']=True;save()
        q['phase']='compile_all64_new_objects'
        for arm in ('A','B'):
            for u in P['units']:
                prefix,src=prefixes[arm,u['name']];expected=selected[arm,u['name']]
                need(all(digest(p)==h for p,h in expected.items()),'PRECOMPILE_DEP_BYTES')
                dep=W/arm/(u['name']+'.actual.d');obj=W/arm/'objects'/(u['name']+'.o')
                q['attempted'][u['kind']+'_compile']+=1
                command(arm+'-'+u['name']+'-compile',prefix+['-MD','-MF',str(dep),'-MT',str(obj),'-c',src,'-o',str(obj)],150)
                q['actual'][u['kind']+'_compile']+=1;q['actual']['objects']+=1
                need(deps(dep)==set(expected) and all(digest(p)==h for p,h in expected.items()),'ACTUAL_MD_M_BYTES_DIFFER')
                q['objects'][arm+'-'+u['name']]=dict(path=str(obj),SHA256=digest(obj),bytes=obj.stat().st_size,MD_SHA256=digest(dep),MD_equal_M=True)
        q['phase']='link_and_passive_ELF';libraries={};reference_needed=elf(original['parent_library'],'reference-elf');need(reference_needed==original['NEEDED'],'REFERENCE_NEEDED')
        need(providers(original['parent_library'],'reference-ldd')==original['providers'],'REFERENCE_PROVIDER71')
        for arm in ('A','B'):
            replacement={u['baseline169_object']:str(W/arm/'objects'/(u['name']+'.o')) for u in P['units']}
            argv=[replacement.get(a,a) for a in original['link_argv']];library=W/arm/('libGLM_history_value_'+arm+'.so');argv[argv.index('-o')+1]=str(library)
            need([a for a in argv if a.endswith('.o')]==[replacement[p] for p in original['object_order']],'NEW32_LINK_ORDER')
            q['attempted']['links']+=1
            command(arm+'-link',argv,90);q['actual']['links']+=1
            syms=nm(library,arm+'-nm');need(all(syms.get(k)==v for k,v in baseline.items()),'BASELINE_STRONG_LOST')
            delta=set(syms)-set(baseline)
            if arm=='A':need(not delta,'A_UNEXPECTED_STRONG_ADDITION')
            else:
                need(len(delta)==1,'B_EXACT_ONE_CPP_METHOD_ADDITION');symbol=next(iter(delta));need('expand_cached_values' in symbol and 'AttentionPrepare' in symbol,'B_WRONG_STRONG_ADDITION')
            need(elf(library,arm+'-elf')==reference_needed,'CANDIDATE_NEEDED')
            need(providers(library,arm+'-ldd')==original['providers'],'CANDIDATE_PROVIDER71')
            table=command(arm+'-tls',['/usr/bin/readelf','-Ws','--demangle',str(library)],30)
            for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current'):
                matches=[s.split() for s in table.splitlines() if s.rstrip().endswith(' '+name)]
                need(len({(r[1],r[6]) for r in matches if len(r)>6 and r[3]=='TLS' and r[6]!='UND'})==1,'TLS_SINGLE_DEFINITION')
            libraries[arm]=dict(path=str(library),SHA256=digest(library),bytes=library.stat().st_size,strong_exports=len(syms),new_strong=sorted(delta),providers=71)
        need(q['actual']==dict(freshM=64,HOST_compile=50,GPU_compile=14,objects=64,links=2),'COMPLETE_DENOMINATORS')
        q.update(passed=True,libraries=libraries,all_MD_equal_freshM=True,component_qualified=False,builder_process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,compiler_or_job_peak_RSS_bytes=None,RSS_scope='builder_process_only_not_compiler_or_cgroup_aggregate')
    except BaseException as e:q.update(error_type=type(e).__name__,error=str(e)[:700])
    finally:save()
    return 0 if q['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
