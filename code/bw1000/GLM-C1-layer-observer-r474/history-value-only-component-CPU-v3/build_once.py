"""One HOST component binding only; exact179 B mirror and parent, no Native load."""
import hashlib,json,math,os,re,shlex,signal,socket,stat,subprocess,time
from pathlib import Path
from probe_recipe import recipe
D=Path(__file__).resolve().parent;W=D.parent
PARENT=Path('/refactor/results/GLM474-history-value-only-common-AB-CPU-v1')
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def cleanup_owned_command(proc,record):
    if proc is None:return
    try:
        if proc.poll() is None:
            need(os.getpgid(proc.pid)==proc.pid==record['owned_pgrp'],'OWN_COMPILER_SESSION')
            if record.get('startticks') is not None:
                row=Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split();need(int(row[19])==record['startticks'],'OWN_COMPILER_STARTTICKS')
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                need(os.getpgid(proc.pid)==proc.pid,'OWN_COMPILER_SESSION');os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
        else:proc.wait(timeout=0)
    except ProcessLookupError:proc.wait(timeout=3)
    finally:record['owned_child_reaped']=proc.returncode is not None
def main():
    J=json.loads((W/'intent.json').read_bytes());C=json.loads((D/'CLAIM.json').read_bytes());start=time.monotonic();available=J['work_deadline']-time.time()
    q=dict(passed=False,planned=dict(freshM=1,HOST_compile=1,GPU_compile=0,objects=1,links=1),actual=dict(freshM=0,HOST_compile=0,GPU_compile=0,objects=0,links=0),attempted=dict(freshM=0,HOST_compile=0,GPU_compile=0,links=0),commands=[],phase='admission',step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),node=socket.gethostname().split('.')[0],Native_dlopen=False,GPU_GO=False,Model_GO=False,component_numerical_qualified=False,device_compute_calls=0,device_driver_queries=0,new_ldd_calls=0,apptainer_setup_implicit_SDK_driver='UNKNOWN',provider_scope='Reuse original179 producer71hash closure and actual rootstaticABI; no new71body hashing. Runtime provider mapping remains futurecomponent qualification.')
    def save():
        q['elapsed_s']=time.monotonic()-start;t=W/'build-result.tmp';t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');os.replace(t,W/'build-result.json')
    def left(reserve=0):
        t=min(J['work_deadline']-time.time(),available-(time.monotonic()-start),J['deadline']-time.time())-reserve
        need(math.isfinite(t) and t>0,'ORIGINAL_WORK_EPOCH_EXHAUSTED');return t
    def take(path,cap=16*1024**2,expected=None,raw=False):
        left(3);p=Path(path);s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'FILE_BOUND')
        h=hashlib.sha256();parts=[]
        with p.open('rb') as f:
            a=os.fstat(f.fileno());need(identity(s)==identity(a),'OPEN_IDENTITY')
            for b in iter(lambda:f.read(1024**2),b''):
                left(3);h.update(b)
                if raw:parts.append(b)
            z=os.fstat(f.fileno())
        need(identity(s)==identity(z)==identity(p.stat()),'STABLE_FD');sha=h.hexdigest();need(expected is None or sha==expected,'FILE_PIN');return b''.join(parts) if raw else sha
    def command(label,argv,limit):
        left(8);rec=dict(label=label,argv=argv,PID=None,owned_child_reaped=False);q['commands'].append(rec);proc=None
        o=W/(label+'.stdout');e=W/(label+'.stderr')
        with o.open('xb') as fo,e.open('xb') as fe:
            try:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=fo,stderr=fe,start_new_session=True);rec.update(PID=proc.pid,owned_pgrp=proc.pid)
                try:
                    row=Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split();rec['startticks']=int(row[19]);need(int(row[2])==proc.pid,'OWN_SESSION')
                except FileNotFoundError:rec['startticks']=None
                rc=proc.wait(timeout=min(limit,left(8)))
            except BaseException as error:rec['error_type']=type(error).__name__;raise
            finally:cleanup_owned_command(proc,rec)
        rec.update(rc=rc,stdout_bytes=o.stat().st_size,stderr_bytes=e.stat().st_size,stdout_SHA256=take(o),stderr_SHA256=take(e));save();need(rc==0,'COMMAND_FAILED:'+label)
        return take(o,raw=True).decode()+'\n'+take(e,raw=True).decode()
    def deps(path):
        text=take(path,raw=True).decode().replace('\\\n',' ');head,colon,tail=text.partition(':');rows=set(shlex.split(tail));need(colon and rows and all(s.startswith('/') for s in rows),'DEPENDENCY_SYNTAX');return rows
    try:
        need(str(W)=='/refactor/results/GLM474-history-value-only-component-CPU-v3' and q['node']=='f13r1n05' and re.fullmatch(r'1103606\.[0-9]+',q['step']),'EXACT_CPU_SCOPE')
        need(C['ROOT_GO'] and C['CPU_GO'] and C['Native_compile_GO'] and not C['GPU_GO'] and not C['Model_GO'],'CPU_NOT_APPROVED')
        need(J['deadline']==J['epoch']+300 and J['inside_deadline']==J['epoch']+270 and J['work_deadline']<=J['epoch']+265,'SINGLE_ORIGIN')
        need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')) and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'DEVICE_HIDDEN_NO_PRELOAD')
        for name,pin in json.loads((D/'SOURCE_FILES.json').read_bytes()).items():need(take(D/name,expected=pin['SHA256']) and (D/name).stat().st_size==pin['bytes'],'OWN_SOURCE_PIN')
        take(D/'SOURCE_FILES.json',expected=J['source_manifest_SHA256']);take(D/'CLAIM.json',expected=J['claim_SHA256'])
        tools=json.loads((D/'TOOLCHAIN.json').read_bytes())
        for path in ('/usr/bin/g++','/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus','/usr/lib/gcc/x86_64-linux-gnu/11/collect2','/usr/bin/as','/usr/bin/ld','/usr/bin/nm','/usr/bin/readelf','/usr/bin/timeout'):
            pin=tools['tools'][path];need(str(Path(path).resolve())==pin['canonical'],'TOOL_CANONICAL');take(path,256*1024**2,pin['sha256'])
        for key,val in tools['required_environment'].items():need(os.environ.get(key)==val,'ORIGINAL_ENV')
        q['phase']='parent_and_recipe'
        out=json.loads(take(PARENT/'outcome.json',2*1024**2,'72558b96b8c7b3ef39d61431c403ffd0dce06116e608504c1028b29fbd58bddf',True));parent=json.loads(take(PARENT/'build-result.json',2*1024**2,'6b098aaf201f1a45f64f4b8d68c40c5b32ac1ed4cfc2e8ea72ec283b568e1560',True))
        need(out['passed'] is True and out['build']==parent and out['accounting']['row'][0]=='1103606.179' and out['accounting']['row'][2:4]==['COMPLETED','0:0'] and parent['passed'] is True,'PARENT179_TERMINAL')
        planraw=take(PARENT/'src/PLAN.json',2*1024**2,'d5356d5b2bc60825537d544ee9f1d100e659f0569eb2521177a466acc7a82fea',True);plan=json.loads(planraw)
        need(len(plan['original169']['providers'])==71 and H(json.dumps(plan['original169']['providers'],sort_keys=True,separators=(',',':')).encode())=='b433fe02fc229c6a2bb3768509f49155a68b60f0860fb3b2325c01a2174c8d4a' and parent['libraries']['B']['providers']==71,'PARENT_PROVIDER_RECORD')
        pthread='/lib/x86_64-linux-gnu/libpthread.so.0';need(plan['original169']['providers'].get(pthread)=='16e807d47ff2441f0203fb2b8805d471e555d932d3d8d262ec6a58ea1d640fd8','ORIGINAL_PTHREAD_PROVIDER_PIN');q['pthread_provider_SHA256']=take(pthread,16*1024**2,'16e807d47ff2441f0203fb2b8805d471e555d932d3d8d262ec6a58ea1d640fd8')
        bso=PARENT/'B/libGLM_history_value_B.so';take(bso,16*1024**2,'02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d');need(bso.stat().st_size==1934848,'PARENT_B_BYTES')
        source=D/'component.cpp';source_sha=take(source);need(source_sha==C['component_source_SHA256'],'FROZEN_COMPONENT_SOURCE');r=recipe(planraw,str(source),source_sha,str(W))
        need(r['link'][1:4]==[str(W/'component.o'),str(bso),pthread] and not any('libGLM_history_value_A.so' in s for s in r['link']),'OBJECT_BEFORE_B_NO_A')
        mirror=str(PARENT/'B/project-mirror');aliases=[Path(mirror+p[len('/refactor'):]) for p in plan['prepare_aliases']]
        need(len(aliases)==11 and len({(p.stat().st_dev,p.stat().st_ino) for p in aliases})==1,'COHERENT11_ALIASES')
        q['parent_B_SHA256']=take(bso);q['planned_CABI']=C['expected_CABI'];q['phase']='fresh_M'
        q['attempted']['freshM']+=1;command('component-M',r['preprocess'],60);names=deps(W/'component.before.d');allowed=r['allowed_dependency_pins']
        need(str(source) in names and names<=set(allowed),'UNREGISTERED_NEW_DEPENDENCY')
        need(all(not p.startswith('/refactor/') or p==str(source) or p.startswith(mirror+'/') for p in names),'QUOTE_INCLUDE_ESCAPES_B_MIRROR')
        pins={p:take(p,expected=allowed[p]) for p in sorted(names)}
        prepare=[p for p in names if p.endswith('/glm_moe_dsa_attention_prepare.hpp')];need(prepare and all(pins[p]=='8027b00c3b88ed6fd6445ddf68076e6896c309a82d67b6e97cb872eaa587ea68' for p in prepare),'NEW_PREPARE_HEADER_REQUIRED')
        (W/'component.dependency-pins.json').write_text(json.dumps(pins,sort_keys=True)+'\n');q['actual']['freshM']=1;q['fresh_M']=dict(count=len(names),SHA256=take(W/'component.before.d'),pins_SHA256=take(W/'component.dependency-pins.json'));save()
        q['phase']='HOST_compile';need(q['actual']['freshM']==1,'M_BEFORE_COMPILE')
        for path,pin in pins.items():take(path,expected=pin)
        q['attempted']['HOST_compile']+=1;command('component-compile',r['compile'],120)
        need(deps(W/'component.actual.d')==names,'ACTUAL_MD_NOT_EQUAL_M')
        for path,pin in pins.items():take(path,expected=pin)
        q['actual']['HOST_compile']=1;q['actual']['objects']=1;q['object']=dict(SHA256=take(W/'component.o',64*1024**2),bytes=(W/'component.o').stat().st_size,MD_SHA256=take(W/'component.actual.d'),MD_equal_M=True);save()
        q['phase']='link_passive_ELF';q['attempted']['links']+=1;command('component-link',r['link'],45);q['actual']['links']=1
        lib=W/'libhistory_value_component.so';libsha=take(lib,64*1024**2)
        text=command('component-nm',['/usr/bin/nm','-D','--defined-only',str(lib)],15)
        strong={a[-1]:a[-2] for line in text.splitlines() if len(a:=line.split())>=3 and a[-2] in ('T','D','B','R')}
        expected_strong={n:'T' for n in C['expected_CABI']};expected_strong.update({'__bss_start':'B','_edata':'D','_end':'B'});need(strong==expected_strong,'EXACT9_CABI_PLUS3_LINKER_MARKERS')
        undefined=command('component-imports',['/usr/bin/nm','-D','--undefined-only',str(lib)],15);imports={line.split()[-1].split('@')[0] for line in undefined.splitlines() if line.split()}
        added=parent['libraries']['B']['new_strong'];need(len(added)==1 and added[0] in imports,'ACTUAL_NEW_METHOD_IMPORT')
        need(any('AttentionPrepare' in s and 'expand_cached' in s and 'expand_cached_values' not in s for s in imports) and any('CommunicationGroupC' in s for s in imports) and any('get_rank_info' in s for s in imports) and 'modern_glm_fp8_runtime_clear_r809' in imports,'ACTUAL_FACTORY_ORIGINAL_METHOD_CLEANUP_IMPORTS')
        dynamic=command('component-dynamic',['/usr/bin/readelf','-d',str(lib)],15);needed=re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',dynamic)
        need('SYMBOLIC' in dynamic and len(needed)==len(set(needed)) and sum(n in (str(bso),bso.name) for n in needed)==1 and set(needed)<=set(plan['original169']['NEEDED'])|{str(bso),bso.name,'libpthread.so.0'},'COMPONENT_NEEDED_B_ONLY')
        table=command('component-tls',['/usr/bin/readelf','-Ws',str(lib)],15);tls={}
        for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current','infinilm::models::glm_moe_dsa::v_on_demand_r400::completed_expand_calls'):
            mangled='_ZN'+''.join(str(len(part))+part for part in name.split('::'))+'E';rows=[s.split() for s in table.splitlines() if s.rstrip().endswith(' '+mangled)];definitions={(x[1],x[6]) for x in rows if len(x)>6 and x[3]=='TLS' and x[6]!='UND'};need(len(definitions)<=1,'DUPLICATE_TLS_DEFINITION');tls[name]=len(definitions)
        take(lib,64*1024**2,libsha);take(bso,16*1024**2,'02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d')
        q.update(passed=True,library=dict(SHA256=libsha,bytes=lib.stat().st_size,strong_CABI=sorted(C['expected_CABI']),linker_markers={'__bss_start':'B','_edata':'D','_end':'B'},thread_imports=sorted(x for x in imports if x.startswith('pthread_')),imports=sorted(imports),NEEDED=needed,TLS_definition_counts=tls),phase='complete')
    except BaseException as e:q.update(error_type=type(e).__name__,error=str(e)[:700])
    finally:save()
    return 0 if q['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
