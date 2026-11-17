"""Independent bounded archive audit. No compile, preprocessing, ldd or Native load."""
import argparse,hashlib,json,math,os,posixpath,re,resource,shlex,signal,stat,subprocess,time
from pathlib import Path

D=Path(__file__).resolve().parent
W=Path('/refactor/results/GLM474-history-value-only-common-AB-CPU-v1')
OUT=Path('/refactor/results/GLM474-history-value-only-common-AB-CPU-reader-v1')
MF='47aada58ed7e9c9068d5a5c5694a4f32512207618aeb4f71c54966691657eb14'
CLAIM='5cf8581d5046fdcf50388042b77d2d5221c101e1cefb72cd0d551f10428fb9a4'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
PLANNED=dict(freshM=64,HOST_compile=50,GPU_compile=14,objects=64,links=2)
H=lambda b:hashlib.sha256(b).hexdigest()

def need(ok,code):
    if not ok:raise ValueError(code)

def strict_json(raw):
    def pairs(rows):
        q={}
        for k,v in rows:
            need(k not in q,'DUPLICATE_JSON_KEY');q[k]=v
        return q
    def bad(value):raise ValueError('NONFINITE_JSON')
    def real(value):
        parsed=float(value);need(math.isfinite(parsed),'NONFINITE_JSON');return parsed
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)

def deps(raw):
    head,colon,body=raw.decode('utf8').replace('\\\n',' ').partition(':')
    rows=shlex.split(body)
    need(colon and rows and all(p.startswith('/') and '\x00' not in p for p in rows),'DEPENDENCY_SYNTAX')
    return set(rows)

def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def main(a):
    origin=a.reader_epoch;anchor=time.monotonic();allowance=origin+180-time.time()
    need(math.isfinite(origin) and 0<allowance<=180,'READER_ORIGIN')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    cores=sorted(os.sched_getaffinity(0));need(len(cores)>=1,'AFFINITY_EMPTY')
    os.sched_setaffinity(0,set(cores[:4]))
    def alarm(*unused):raise TimeoutError('READER_EPOCH_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,allowance)
    def left():
        n=min(origin+180-time.time(),allowance-(time.monotonic()-anchor))
        need(n>2,'READER_EPOCH_EXHAUSTED');return n
    need(D==OUT/'src' and OUT.is_dir() and not OUT.is_symlink() and OUT.stat().st_uid==os.getuid(),'READER_NAMESPACE_STATE')
    # Receiver owns installation; reuse it exactly once, not a second mkdir.
    labels=['reference-independent-nm']+[arm+'-independent-'+kind for arm in ('A','B') for kind in ('nm','dynamic','tls')]
    prior=['report-private.json','reader-start.json']+[label+'.'+ext for label in labels for ext in ('stdout','stderr')]
    need(not any((OUT/name).exists() or (OUT/name).is_symlink() for name in prior),'READER_OUTPUT_ALREADY_EXISTS')
    q=dict(schema='GLM474-commonAB-independent-archive-v1',request_id=a.request_id,publication=a.publication,
        original_epoch=a.epoch,reader_epoch=origin,reader_deadline=origin+180,passed=False,execution_state='UNKNOWN',
        CPU_ELF_qualified=False,component_qualified=False,GPU_GO=False,Model_GO=False,replay_allowed=False,
        planned=PLANNED,producer_actual=None,independently_verified=dict(freshM=0,HOST_compile=0,GPU_compile=0,objects=0,links=0),
        failures=[],units={},libraries={},files={},total_hashed_bytes=0,phase='source_and_control',
        reader_device_compute_calls=0,reader_driver_queries=0,reader_Native_dlopen=False,reader_ldd_calls=0,
        dynamic_resolution='Original producer ldd observation only; no new dynamic-loader verification.',
        ordering_scope='Original command records plus stable saved dependencies; no independent live compile ordering observation.',
        address_space_hard_limit_bytes=3*1024**3,CPU_affinity=sorted(os.sched_getaffinity(0)))
    cache={};owns_read=False
    def take(path,cap=16*1024**2,raw=False,expected=None):
        left();p=Path(path);s=p.stat();need(stat.S_ISREG(s.st_mode) and 0<=s.st_size<=cap,'FILE_BOUND')
        key=str(p)
        if not raw and key in cache and identity(s)==cache[key][0]:
            sha=cache[key][1];need(expected is None or sha==expected,'FILE_PIN');return sha
        with p.open('rb') as f:
            before=os.fstat(f.fileno());need(identity(s)==identity(before),'OPEN_IDENTITY')
            h=hashlib.sha256();chunks=[]
            for block in iter(lambda:f.read(1024**2),b''):
                left();h.update(block);q['total_hashed_bytes']+=len(block)
                need(q['total_hashed_bytes']<=16*1024**3,'TOTAL_HASH_BOUND')
                if raw:chunks.append(block)
            after=os.fstat(f.fileno())
        need(identity(s)==identity(after)==identity(p.stat()),'STABLE_FD')
        sha=h.hexdigest();need(expected is None or sha==expected,'FILE_PIN')
        cache[key]=(identity(s),sha);q['files'][key]=dict(bytes=s.st_size,SHA256=sha)
        return b''.join(chunks) if raw else sha
    def js(p,cap=16*1024**2,expected=None):return strict_json(take(p,cap,True,expected))
    def failure(stage,error):q['failures'].append(dict(stage=stage,error_type=type(error).__name__,code=str(error)[:160] if isinstance(error,ValueError) else 'IO_OR_BUDGET_FAILURE'))
    def passive(label,argv):
        left();fo=OUT/(label+'.stdout');fe=OUT/(label+'.stderr');proc=None
        with fo.open('xb') as o,fe.open('xb') as e:
            try:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)
                rc=proc.wait(timeout=min(25,left()-1));need(rc==0,'PASSIVE_TOOL_EXIT')
            finally:
                if proc is not None:
                    if proc.poll() is None:proc.kill()
                    proc.wait(timeout=3)
        return take(fo,16*1024**2,True).decode()+ '\n'+take(fe,1024**2,True).decode()
    def strong(p,label):
        text=passive(label,['/usr/bin/nm','-D','--defined-only',str(p)])
        out={}
        for line in text.splitlines():
            row=line.split()
            if len(row)>=3 and row[-2] in ('T','D','B','R'):
                need(row[-1] not in out,'DUPLICATE_STRONG_SYMBOL');out[row[-1]]=row[-2]
        return out
    try:
        selfmf=js(D/'SOURCE_FILES.json',expected=a.reader_manifest_sha256)
        for name,pin in selfmf.items():
            need(not Path(name).is_absolute() and '..' not in Path(name).parts,'READER_MEMBER')
            take(D/name,16*1024**2,expected=pin['SHA256']);need((D/name).stat().st_size==pin['bytes'],'READER_MEMBER_BYTES')
        own=js(D/'CLAIM.json');need(own['ROOT_GO'] and own['CPU_read_GO'] and not own['GPU_GO'],'READER_NOT_APPROVED')
        ri=js(OUT/'install-intent.json')
        need(ri['original_request_id']==a.request_id and ri['original_publication']==a.publication and ri['original_epoch']==a.epoch and ri['reader_epoch']==origin and ri['reader_deadline']==origin+180 and ri['source_manifest_SHA256']==a.reader_manifest_sha256 and ri['claim_SHA256']==take(D/'CLAIM.json') and ri['original_outcome_SHA256']==a.outcome_sha256,'READER_INSTALL_CONTEXT')
        with (OUT/'reader-start.json').open('x',encoding='utf8') as f:
            owns_read=True
            json.dump(dict(reader_epoch=origin,original_request_id=a.request_id,source_manifest_SHA256=a.reader_manifest_sha256,original_outcome_SHA256=a.outcome_sha256),f,allow_nan=False);f.write('\n')
        mf=js(W/'src/SOURCE_FILES.json',expected=MF)
        for name,pin in mf.items():
            need(not Path(name).is_absolute() and '..' not in Path(name).parts,'SOURCE_MEMBER')
            take(W/'src'/name,16*1024**2,expected=pin['SHA256']);need((W/'src'/name).stat().st_size==pin['bytes'],'SOURCE_BYTES')
        take(W/'src/CLAIM.json',expected=CLAIM)
        P=js(W/'src/PLAN.json');tools=js(W/'src/TOOLCHAIN.json')
        expected=dict(request_id=a.request_id,publication=a.publication,epoch=a.epoch,source_manifest_SHA256=MF,claim_SHA256=CLAIM)
        install=js(W/'install-intent.json');control=js(W/'control-state.json');outcome=js(W/'outcome.json',expected=a.outcome_sha256)
        for obj in (install,control):need(all(obj.get(k)==v for k,v in expected.items()),'CONTROL_OR_INSTALL_BINDING')
        need(outcome.get('control_state_SHA256')==take(W/'control-state.json'),'OUTCOME_CONTROL_SHA')
        q['exec_request_attempted']=control.get('exec_request_attempted')
        if control.get('exec_request_attempted') is False:
            need(outcome.get('current_step_outcome')=='NOT_SUBMITTED' and outcome.get('control_checkpoint_verified') is True,'NOT_SUBMITTED_EVIDENCE')
            q['execution_state']='NOT_SUBMITTED';q['failure_stage']='prelaunch';return q
        need(control.get('exec_request_attempted') is True,'CONTROL_BOOLEAN')
        J=js(W/'intent.json');need(all(J.get(k)==v for k,v in expected.items()),'EXEC_INTENT_BINDING')
        need(J['deadline']==a.epoch+1800 and J['inside_deadline']==a.epoch+1740 and J['work_deadline']<=a.epoch+1680,'ORIGINAL_EPOCH')
        need(outcome.get('epoch')==a.epoch and outcome.get('deadline')==J['deadline'],'OUTCOME_EPOCH')
        need(J['Slurm_seconds']%60==0 and J['Slurm_seconds']<=1740 and J['inside_seconds']+10<=J['Slurm_seconds'],'MINUTE_FLOOR')
        need(J['GPU_cards']==0 and J['maximum_invocations']==1 and J['node']=='f13r1n05','ORIGINAL_SCOPE')
        pre=js(W/'HOST-preflight.json');uid=re.fullmatch(r'(?:[^()\s]+\((\d+)\)|(\d+))',pre['job'].get('UserId',''))
        need(uid and int(uid.group(1) or uid.group(2))==os.getuid() and pre['epoch']==a.epoch and pre['parent_end']>=J['deadline']+30 and pre['node']==J['node'] and pre['active_step_conflicts'] is False,'ORIGINAL_HOST_PREFLIGHT')
        for flag in ('--jobid=1103606','--job-name='+P['label'],'--exact','--nodes=1','--ntasks=1','--nodelist=f13r1n05','--cpus-per-task=4','--mem=16384M','--gres=none','--immediate=5'):
            need(J['command'].count(flag)==1,'ORIGINAL_SRUN_RESOURCES')
        row=outcome.get('accounting',{}).get('row')
        need(isinstance(row,list) and len(row)==7 and re.fullmatch(r'1103606\.[0-9]+',row[0]) and row[1]==P['label'] and row[2].split()[0] in TERMINAL and row[5]=='f13r1n05','EXACT_TERMINAL_ACCOUNTING')
        need('gres/dcu=' not in row[6] and 'cpu=4' in row[6].split(','),'CPU_ACCOUNTING_RESOURCES')
        q.update(execution_state='TERMINAL',accounting=row,original_failed=row[2:4]!=['COMPLETED','0:0'] or outcome.get('rc')!=0)
        B=js(W/'build-result.json',expected=outcome.get('build_SHA256'));need(B['step']==row[0] and B==outcome.get('build'),'BUILD_ACCOUNTING_BINDING')
        need(B['planned']==PLANNED,'PLANNED_DENOMINATORS');q['producer_actual']=B['actual'];q['producer_attempted']=B['attempted'];q['producer_phase']=B['phase']
        q['producer_failure']=dict(error_type=B.get('error_type'),error_SHA256=H(str(B.get('error','')).encode()))
        q['owned_srun_receipt']=js(W/'owned-srun.json')
        q['owned_srun_liveness']='Not re-observed: original login-host PID namespace is not assumed to be this container. Exact archived step terminal is required.'
        take(W/'step.log',64*1024**2,expected=outcome['log_SHA256'])
        need(len(P['units'])==32 and len(P['prepare_aliases'])==11 and len(P['project_pins'])==1859 and len(P['system_pins'])==494,'PLAN_DENOMINATORS')
        commands={}
        for pos,c in enumerate(B['commands']):
            label=c['label'];need(re.fullmatch(r'(?:reference-(?:nm|elf|ldd)|[AB]-(?:u[0-9]{2}-(?:M|compile)|link|nm|elf|ldd|tls))',label) and label not in commands,'COMMAND_LABEL')
            commands[label]=(pos,c)
            for ext in ('stdout','stderr'):
                if ext+'_SHA256' in c:take(W/(label+'.'+ext),16*1024**2,expected=c[ext+'_SHA256'])
        normalized={};all_m=[];all_compile=[]
        q['phase']='dependency_object_archives'
        if B.get('source_bytes_preflight_passed'):
            for path,pin in P['system_pins'].items():take(path,16*1024**2,expected=pin)
        for arm in ('A','B'):
            mirror=W/arm/'project-mirror';allowed=dict(P['system_pins']);identities=[]
            for path,basehash in P['project_pins'].items():
                effective=P['common_override'].get(path,basehash)
                if arm=='B':effective=P['B_overrides'].get(path,effective)
                dest=mirror/posixpath.normpath(path)[len('/refactor/'):]
                allowed[str(dest)]=effective;allowed[str(mirror)+path[len('/refactor'):]]=effective
                if mirror.exists():take(dest,16*1024**2,expected=effective)
            if mirror.exists():
                for path in P['prepare_aliases']:
                    dest=mirror/posixpath.normpath(path)[len('/refactor/'):]
                    take(dest,expected=allowed[str(dest)]);s=dest.stat();identities.append((s.st_dev,s.st_ino))
                need(len(set(identities))==1,'PREPARE_ALIAS_INODE')
            for u in P['units']:
                key=arm+'-'+u['name'];result=dict(M=False,object=False,missing=True);q['units'][key]=result
                try:
                    fm=B['fresh_M'].get(key)
                    if fm is None:continue
                    dep=W/arm/(u['name']+'.before.d');dp=W/arm/(u['name']+'.dependency-pins.json')
                    need(fm['pins_file']==str(dp),'DEPENDENCY_PIN_PATH')
                    names=deps(take(dep,16*1024**2,True,fm['MD_SHA256']));pins=js(dp,expected=fm['pins_SHA256'])
                    src=str(mirror)+posixpath.normpath(u['source'])[len('/refactor'):]
                    need(src in names and names==set(pins) and len(names)==fm['count'] and names<=set(allowed),'M_GRAPH')
                    for path in names:
                        need(not path.startswith('/refactor/') or path.startswith(str(mirror)+'/'),'MIRROR_ESCAPE')
                        need(pins[path]==allowed[path],'DEPENDENCY_EXPECTED_PIN');take(path,16*1024**2,expected=allowed[path])
                    result.update(M=True,missing=False,dependencies=len(names));q['independently_verified']['freshM']+=1
                    normalized[key]={('/refactor'+p[len(str(mirror)):] if p.startswith(str(mirror)) else p):pins[p] for p in names}
                    mpos,mc=commands[key+'-M'];need(mc.get('rc')==0 and mc.get('owned_child_reaped') is True,'M_COMMAND_STATUS');all_m.append(mpos)
                    compiler='/usr/bin/g++' if u['kind']=='HOST' else '/opt/dtk/cuda/cuda/bin/nvcc'
                    inc=['-I'+(str(mirror)+path[len('/refactor'):] if path.startswith('/refactor/') else path) for path in P['include_roots']]
                    prefix=[compiler]+P['new_common_flags'][u['kind']]+inc
                    target=W/arm/'objects'/(u['name']+'.o')
                    need(mc['argv']==prefix+['-M','-MF',str(dep),'-MT',str(target),src],'FRESH_M_RECIPE')
                    obj=B['objects'].get(key)
                    if obj is None:continue
                    target=W/arm/'objects'/(u['name']+'.o');need(obj['path']==str(target),'OBJECT_PATH')
                    actual=W/arm/(u['name']+'.actual.d');need(deps(take(actual,16*1024**2,True,obj['MD_SHA256']))==names,'MD_NOT_EQUAL_M')
                    take(target,512*1024**2,expected=obj['SHA256']);need(target.stat().st_size==obj['bytes'],'OBJECT_BYTES')
                    cp,cc=commands[key+'-compile'];need(cc.get('rc')==0 and cc.get('owned_child_reaped') is True,'COMPILE_COMMAND_STATUS');all_compile.append(cp)
                    need(cc['argv']==prefix+['-MD','-MF',str(actual),'-MT',str(target),'-c',src,'-o',str(target)],'COMPILE_COMMON_RECIPE')
                    result['object']=True;q['independently_verified']['objects']+=1;q['independently_verified'][u['kind']+'_compile']+=1
                except (ValueError,OSError,KeyError) as e:failure(key,e)
        changes=set(P['B_overrides'])|{P['helper_path']}
        for u in P['units']:
            aa=normalized.get('A-'+u['name']);bb=normalized.get('B-'+u['name'])
            if aa is not None and bb is not None:
                need(set(aa)^set(bb)<=changes and all(aa[p]==bb[p] or p in changes for p in set(aa)&set(bb)),'AB_GRAPH_DELTA')
        if all_compile:need(len(all_m)==64 and max(all_m)<min(all_compile),'ARCHIVED_M_BEFORE_COMPILE')
        q['phase']='passive_ELF'
        for tool in ('/usr/bin/nm','/usr/bin/readelf'):
            pin=tools['tools'][tool];need(str(Path(tool).resolve())==pin['canonical'],'PASSIVE_TOOL_CANONICAL');take(tool,64*1024**2,expected=pin['sha256'])
        original=P['original169'];need(len(original['providers'])==71 and len(original['NEEDED'])==14,'PROVIDER_NEEDED_DENOMINATORS')
        # No old qualification is inherited: only the pinned reference ELF defines the strong-symbol baseline.
        baseline=None
        for arm in ('A','B'):
            lib=B.get('libraries',{}).get(arm)
            if lib is None:continue
            try:
                if baseline is None:
                    ref=Path(original['parent_library']);take(ref,1024**3,expected=original['parent_library_SHA256']);baseline=strong(ref,'reference-independent-nm');need(len(baseline)==280,'BASELINE280')
                p=W/arm/('libGLM_history_value_'+arm+'.so');need(lib['path']==str(p),'LIBRARY_PATH');before=p.stat()
                replacements={u['baseline169_object']:str(W/arm/'objects'/(u['name']+'.o')) for u in P['units']}
                link=[replacements.get(arg,arg) for arg in original['link_argv']];link[link.index('-o')+1]=str(p)
                lc=commands[arm+'-link'][1]
                need(lc.get('rc')==0 and lc.get('owned_child_reaped') is True and lc['argv']==link and [arg for arg in link if arg.endswith('.o')]==[replacements[o] for o in original['object_order']],'NEW32_LINK_RECIPE')
                sha=take(p,1024**3,expected=lib['SHA256']);need(before.st_size==lib['bytes'],'LIBRARY_BYTES')
                syms=strong(p,arm+'-independent-nm');need(all(syms.get(k)==v for k,v in baseline.items()),'BASELINE_SYMBOL_LOST')
                delta=set(syms)-set(baseline)
                need(not delta if arm=='A' else len(delta)==1 and all('AttentionPrepare' in s and 'expand_cached_values' in s for s in delta),'STRONG_DELTA')
                text=passive(arm+'-independent-dynamic',['/usr/bin/readelf','-d',str(p)])
                need('SYMBOLIC' in text and re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',text)==original['NEEDED'],'NEEDED14_SYMBOLIC')
                table=passive(arm+'-independent-tls',['/usr/bin/readelf','-Ws','--demangle',str(p)])
                for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current'):
                    rows=[s.split() for s in table.splitlines() if s.rstrip().endswith(' '+name)]
                    need(len({(r[1],r[6]) for r in rows if len(r)>6 and r[3]=='TLS' and r[6]!='UND'})==1,'TLS_SINGLETON')
                log='\n'.join(take(W/(arm+'-ldd.'+ext),16*1024**2,True).decode() for ext in ('stdout','stderr'))
                need('not found' not in log and 'undefined symbol' not in log,'ORIGINAL_LDD_UNRESOLVED')
                paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',log,re.M));need(paths==set(original['providers']),'ORIGINAL_PROVIDER_PATHS')
                for path,pin in original['providers'].items():take(path,1024**3,expected=pin)
                need(identity(before)==identity(p.stat()),'ELF_CHANGED_DURING_PARSE')
                q['libraries'][arm]=dict(SHA256=sha,strong=len(syms),added=sorted(delta),NEEDED=14,provider_file_SHA_verified=71,TLS_singletons=2)
                q['independently_verified']['links']+=1
            except (ValueError,OSError,KeyError,subprocess.SubprocessError) as e:failure(arm+'-ELF',e)
        q['CPU_ELF_qualified']=bool(not q['original_failed'] and not q['failures'] and q['independently_verified']==PLANNED and B.get('passed') is True)
        q['passed']=q['CPU_ELF_qualified'];q['phase']='complete'
    except BaseException as error:failure(q['phase'],error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        q['missing']={k:PLANNED[k]-q['independently_verified'][k] for k in PLANNED}
        q['elapsed_s']=time.monotonic()-anchor;q['reader_process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        q['RSS_scope']='reader process only, not passive-tool or cgroup peak; AS is address space, not RSS'
        raw=(json.dumps(q,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<=4*1024**2,'REPORT_BOUND')
        if owns_read:
            with (OUT/'report-private.json').open('xb') as f:f.write(raw)
        print(json.dumps(dict(passed=q['passed'],execution_state=q['execution_state'],CPU_ELF_qualified=q['CPU_ELF_qualified'],report_bytes=len(raw) if owns_read else 0,report_SHA256=H(raw) if owns_read else None,verified=q['independently_verified'],missing=q['missing'],failure_count=len(q['failures']),GPU_GO=False,Model_GO=False)))
    return q

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request-id',required=True);p.add_argument('--publication',required=True)
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--reader-epoch',type=float,required=True)
    p.add_argument('--reader-manifest-sha256',required=True);p.add_argument('--outcome-sha256',required=True);a=p.parse_args()
    need(re.fullmatch('[0-9a-f]{32}',a.request_id) and re.fullmatch('[0-9a-f]{40}',a.publication) and all(re.fullmatch('[0-9a-f]{64}',s) for s in (a.reader_manifest_sha256,a.outcome_sha256)),'CLI_PINS')
    result=main(a);raise SystemExit(0 if result['execution_state'] in ('TERMINAL','NOT_SUBMITTED') else 1)
