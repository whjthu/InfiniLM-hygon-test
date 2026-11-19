"""One depth-aware library: all 32 fresh M, 8 new HOST and 24 verified objects."""
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
    q=dict(passed=False,step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),node=socket.gethostname().split('.')[0],planned=P['expected_totals'],actual=dict(freshM=0,HOST_compile=0,GPU_compile=0,reused_objects=0,objects=0,links=0),attempted=dict(freshM=0,HOST_compile=0,GPU_compile=0,links=0),objects={},fresh_M={},commands=[],phase='admission',new_recipe=True,old169_qualification_inherited=False,Native_dlopen=False,device_compute_calls=0,device_driver_queries=0,SDK_explicit_calls=0,implicit_ldd_initialization='UNKNOWN',GPU_GO=False,Model_GO=False)
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
        original=P['original169'];binding=read(D/'BASELINE_BINDING.json')
        baseline=P['baseline_B']
        need(binding['library']==baseline and binding['build_SHA256']==P['baseline_B_build_SHA256']
             and binding['step']=='1103606.179','ORIGINAL_B_BINDING')
        old_root=P['baseline_B_base'];old_mirror=old_root+'/B/project-mirror'
        need(digest(old_root+'/build-result.json')==binding['build_SHA256'],'ORIGINAL_B_BUILD_RESULT_PIN')
        need(digest(baseline['path'])==baseline['SHA256'],'ORIGINAL_B_DSO_PIN')
        baseline_exports=nm(baseline['path'],'reference-B-nm')
        need(len(baseline_exports)==baseline['strong_exports']==281,'REFERENCE_B_STRONG281')
        # Keep the original selected-history B files; read only registered paths.
        q['phase']='preflight_source_bytes';raws={};total=0
        for path,h in P['project_pins'].items():
            blob=P['source_blobs'].get(h)
            b=stable(D/blob,h) if blob else stable(path,h)
            total+=len(b);need(total<=256*1024**2,'PROJECT_STAGE_TOTAL_BOUND');raws[path]=b
        for path,h in P['system_pins'].items():need(digest(path)==h,'SYSTEM_HEADER_PIN')
        q.update(source_bytes_preflight_passed=True,project_registered=len(raws),project_read_bytes=total);save()
        q['phase']='owned_mirror'
        root=W/'B';root.mkdir(mode=0o700);(root/'objects').mkdir()
        mirror=root/'project-mirror';mirror.mkdir()
        effective={};shared={}
        for path,b in raws.items():
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
        inc=['-I'+(str(mirror)+p[len('/refactor'):] if p.startswith('/refactor/') else p) for p in P['include_roots']]
        allowed=dict(P['system_pins'],**effective)
        prefixes={}
        for u in P['units']:
            src=str(mirror)+posixpath.normpath(u['source'])[len('/refactor'):]
            compiler='/usr/bin/g++' if u['kind']=='HOST' else '/opt/dtk/cuda/cuda/bin/nvcc'
            flags=P['new_common_flags'][u['kind']]
            if u['name'] not in P['affected_units']:
                flags=[flag for flag in flags if flag!='-DGLM_R474_DEPTH_STRATIFIED_DETAILS=1']
            prefixes[u['name']]=([compiler]+flags+inc,src)

        def normalized(deps,mirror_prefix):
            result={}
            for p,h in deps.items():
                visible='/refactor'+p[len(mirror_prefix):] if p.startswith(mirror_prefix+'/') else p
                key=posixpath.normpath(visible)
                need(result.setdefault(key,h)==h,'INCOHERENT_NORMALIZED_DEPENDENCY_ALIAS')
            return result

        changed={posixpath.normpath(p):h for p,h in P['old_B_project_pins'].items()}
        updated={p:P['project_pins'][p] for p in P['old_B_project_pins']}
        updated={posixpath.normpath(p):h for p,h in updated.items()}
        registered_depth={posixpath.normpath(p) for p in P['project_pins']
                          if p.endswith('/depth_detail_plan.hpp')}
        system_cstddef={posixpath.normpath(p) for p in P['system_pins'] if p.endswith('/cstddef')}
        selected={};q['phase']='all32_fresh_M'
        for u in P['units']:
            name=u['name'];old=binding['units'][name]
            original_graph=json.loads(stable(old_root+'/B/'+name+'.dependency-pins.json',old['dependency_pins_SHA256']))
            baseline_graph=normalized(original_graph,old_mirror)
            used_delta={p:h for p,h in baseline_graph.items() if p in changed}
            recorded_delta={posixpath.normpath(p):h for p,h in old['affected'].items()}
            need(used_delta==recorded_delta and all(changed[p]==h for p,h in used_delta.items()),
                 'ORIGINAL_DEPENDENCY_DELTA:'+name)
            need((name in P['affected_units'])==bool(used_delta),'ORIGINAL_AFFECTED_SCOPE:'+name)
            prefix,src=prefixes[name];dep=root/(name+'.before.d');obj=root/'objects'/(name+'.o')
            q['attempted']['freshM']+=1
            command('B-'+name+'-M',prefix+['-M','-MF',str(dep),'-MT',str(obj),src],60)
            names=deps(dep);need(src in names and names<=set(allowed),'FRESH_M_UNREGISTERED:'+name)
            need(not any(p.startswith('/refactor/') and not p.startswith(str(mirror)+'/') for p in names),
                 'QUOTE_INCLUDE_ESCAPED_MIRROR:'+name)
            actual={p:digest(p) for p in names}
            need(all(actual[p]==allowed[p] for p in names),'FRESH_M_BYTES:'+name)
            graph=normalized(actual,str(mirror))
            expected=dict(baseline_graph)
            expected.update({p:updated[p] for p in used_delta})
            added=set(graph)-set(expected)
            need(set(expected)<=set(graph) and all(graph[p]==h for p,h in expected.items()),
                 'ORIGINAL_UNCHANGED_DEPENDENCIES_OR_EXPECTED_SOURCE_DELTA_DIFFERS:'+name)
            if used_delta:
                required=set(added)&registered_depth
                need(len(required)==1 and added<=registered_depth|system_cstddef,
                     'EXACT_DEPTH_HEADER_ADDITION:'+name)
            else:
                need(not added,'UNEXPECTED_UNAFFECTED_DEPENDENCY:'+name)
            selected[name]=actual;q['actual']['freshM']+=1
            dp=root/(name+'.dependency-pins.json');dp.write_text(json.dumps(actual,sort_keys=True)+'\n')
            q['fresh_M'][name]=dict(count=len(names),MD_SHA256=digest(dep),pins_file=str(dp),pins_SHA256=digest(dp),
                                      normalized_added=sorted(added),affected=bool(used_delta))
        need(q['actual']['freshM']==32,'ALL32_M_BEFORE_COMPILE')
        q['all32_M_before_compile']=True;q['phase']='compile_eight_reuse_twentyfour';save()
        replacement={}
        for u in P['units']:
            name=u['name'];old=binding['units'][name];expected=selected[name]
            need(all(digest(p)==h for p,h in expected.items()),'PREBUILD_DEP_BYTES:'+name)
            if name in P['affected_units']:
                need(u['kind']=='HOST','AFFECTED_HOST_ONLY')
                prefix,src=prefixes[name];dep=root/(name+'.actual.d');obj=root/'objects'/(name+'.o')
                q['attempted']['HOST_compile']+=1
                command('B-'+name+'-compile',prefix+['-MD','-MF',str(dep),'-MT',str(obj),'-c',src,'-o',str(obj)],150)
                need(deps(dep)==set(expected) and all(digest(p)==h for p,h in expected.items()),
                     'ACTUAL_MD_M_BYTES_DIFFER:'+name)
                q['actual']['HOST_compile']+=1
                item=dict(path=str(obj),SHA256=digest(obj),bytes=obj.stat().st_size,MD_SHA256=digest(dep),MD_equal_M=True,reused=False)
            else:
                obj=Path(old['original_object'])
                need(obj.stat().st_size==old['object_bytes'] and digest(obj)==old['object_SHA256'],
                     'REUSED_B_OBJECT_PIN:'+name)
                q['actual']['reused_objects']+=1
                item=dict(path=str(obj),SHA256=old['object_SHA256'],bytes=old['object_bytes'],reused=True,
                          original_B_dependency_pins_SHA256=old['dependency_pins_SHA256'])
            q['objects'][name]=item;q['actual']['objects']+=1
            replacement[u['baseline169_object']]=str(obj)
        q['phase']='link_and_passive_ELF';save()
        reference_needed=elf(baseline['path'],'reference-B-elf')
        need(reference_needed==original['NEEDED'],'REFERENCE_B_NEEDED')
        need(providers(baseline['path'],'reference-B-ldd')==original['providers'],'REFERENCE_B_PROVIDER71')
        for name,item in q['objects'].items():
            need(digest(item['path'])==item['SHA256'],'PRELINK_OBJECT_STABILITY:'+name)
        argv=[replacement.get(a,a) for a in original['link_argv']]
        library=root/'libGLM_depth_observer_B.so';argv[argv.index('-o')+1]=str(library)
        need([a for a in argv if a.endswith('.o')]==[replacement[p] for p in original['object_order']],
             'EXACT_ORDERED_32_OBJECTS')
        q['attempted']['links']+=1;command('B-link',argv,90);q['actual']['links']+=1
        need(nm(library,'B-nm')==baseline_exports,'ORIGINAL_B_STRONG_SYMBOL_MAP_CHANGED')
        need(elf(library,'B-elf')==reference_needed,'CANDIDATE_NEEDED')
        need(providers(library,'B-ldd')==original['providers'],'CANDIDATE_PROVIDER71')
        table=command('B-tls',['/usr/bin/readelf','-Ws','--demangle',str(library)],30)
        tls=('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current','glm_sparse_r131::depth_pair')
        for name in tls:
            matches=[s.split() for s in table.splitlines() if s.rstrip().endswith(' '+name)]
            need(len({(r[1],r[6]) for r in matches if len(r)>6 and r[3]=='TLS' and r[6]!='UND'})==1,
                 'TLS_SINGLE_DEFINITION:'+name)
        need(q['actual']==P['expected_totals'],'COMPLETE_DENOMINATORS')
        q.update(passed=True,library=dict(path=str(library),SHA256=digest(library),bytes=library.stat().st_size,
               strong_exports=len(baseline_exports),providers=71,TLS_single_definition=list(tls)),
               modified_MD_equal_freshM=True,reused_B_dependencies_exact=True,component_qualified=False,
               builder_process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except BaseException as e:q.update(error_type=type(e).__name__,error=str(e)[:700])
    finally:save()
    return 0 if q['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
