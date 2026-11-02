"""Replace only afc6 HC unit3, append the exact CPU700 HIP object, CPU-only closure."""
import pathlib,json,hashlib,subprocess,os,time,shlex,re,sys,copy,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,whole_Target_Model_loaded=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
started=time.perf_counter()
def run(args,label,timeout=180):
    q['commands'].append(args);z=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.returncode,z.stderr[-5000:]);return z.stdout
def symbols(path):return {line.split()[-1]for line in run(['nm','-D','--defined-only',str(path)],'symbols-'+path.stem).splitlines()if line.split()}
try:
    parent_path=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(parent_path)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5';parent=json.loads(parent_path.read_bytes());n=parent['families']['Deep']['Native'];oldlib=pathlib.Path(n['new']);assert sha(oldlib)==n['new_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
    sys.path.insert(0,str(R/'incoming/Deep-prefill-chunk-pair-r348/repair-r2/owner'))
    from native_binding_r206b import binding
    parent_binding,parent_receipt,parent_normal=binding();oldobjects=parent_normal['link']['full']['objects'];assert len(oldobjects)==48
    oldunit=R/'candidates/deep-combined-r7169/objects/unit-3.o';assert oldobjects[str(oldunit)]==sha(oldunit)=='3269f7e6aea58fcfd961ab26b0d85841be36ffe12bb2b6fa4525da80264ef558'
    original=R/'candidates/deep-combined-r7169/incoming/deepseek_v4_hyper_connection_r177.cpp';assert sha(original)=='4b178cd98cd9e8dd2e24614bf022c1297d6332f22b2c790530fbb06866c5d313'
    source=S/'deepseek_v4_hyper_connection_model_r366.cpp';stripped=re.sub(r'/\*HC366_BEGIN_([A-Z0-9_]+)\*/.*?/\*HC366_END_\1\*/','',source.read_text(),flags=re.S);assert stripped==original.read_text()
    proof=json.loads((S/'SOURCE_PROOF.json').read_bytes());assert sha(source)==proof['new_cpp_source_SHA256'];assert sha(proof['original_header_path'])==proof['original_header_SHA256']
    kernel=R/'candidates/Deep-HC-epilogue-r358-cpu-r2/epilogue.o';assert sha(kernel)=='514f3739f71bcd6165edb2d2cedf387e7835b5f8026310b907c0eba968506fda'
    for p,h in oldobjects.items():assert sha(p)==h
    a=n['link_argv'];oi=a.index('-o');flags=[v for i,v in enumerate(a)if i not in [0,oi,oi+1]and not v.endswith('.o')and v!='-shared'and not v.startswith(('-l','-L','-Wl,'))]
    flags=['-I'+str(S),'-I'+str(original.parent)]+flags
    obj=B/'HC-unit3.o';md=B/'HC-unit3.d';parse=lambda t:set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]))
    fresh=parse(run(['g++',*flags,'-M',str(source)],'fresh-M'));before={p:sha(p)for p in fresh}
    oldrecord=json.loads((R/'results/deep-combined-r7169-build.json').read_bytes());unit=next(v for v in oldrecord['compiled_units']if v['object']==str(oldunit));hist=unit['deps']
    for p,h in before.items():
        if pathlib.Path(p).is_relative_to(S):continue
        assert p in hist and h==hist[p],('original-HC-dependency-diff',p,h,hist.get(p))
    assert any(p.endswith('/Model/Deep/InfiniLM/csrc/execution/submit.hpp')for p in fresh)and any(p.endswith('/execution/owner_affinity_r2392.hpp')for p in fresh)
    run(['g++',*flags,'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'HC-unit3-compile');assert parse(md.read_text())==fresh and all(sha(p)==h for p,h in before.items())
    lib=B/'libDeep_HC_T1_Model_r366.so';link=[str(obj)if p==str(oldunit)else p for p in a];link[link.index('-o')+1]=str(lib);link[1:1]=[str(kernel)];link[link.index('-o'):link.index('-o')]=['-L/opt/dtk/lib','-lamdhip64']
    run(link,'link',120);oldnames=symbols(oldlib);newnames=symbols(lib);assert oldnames<=newnames;required=['deep_hc_model_epilogue_'+name+'_r366'for name in ['set','get','counters']]+['deep_hc_epilogue_r358'];assert set(required)<=newnames
    decoded=run(['nm','-D','-C','--defined-only',str(lib)],'demangled');assert sum(' T 'in line and'HyperConnection::pre('in line for line in decoded.splitlines())==1
    ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd
    needed=lambda p:[line.split('[')[1].split(']')[0]for line in run(['readelf','-d',str(p)],'dynamic-'+p.stem).splitlines()if'NEEDED'in line]
    oldneeded=needed(oldlib);newneeded=needed(lib);assert set(oldneeded)<=set(newneeded);assert set(newneeded)-set(oldneeded)<={'libgalaxyhip.so.5'}
    bindings=copy.deepcopy(parent_receipt['actual_DTK_bindings'])
    for p,v in bindings.items():assert sha(p)==v['SHA256']
    hip=pathlib.Path('/opt/dtk/lib/libgalaxyhip.so.5');hipsha=sha(hip)
    oldldd=run(['ldd',str(oldlib)],'parent-ldd');oldhip=next(line.split('=>',1)[1].strip().split()[0]for line in oldldd.splitlines()if'libgalaxyhip.so.5 =>'in line);newhip=next(line.split('=>',1)[1].strip().split()[0]for line in ldd.splitlines()if'libgalaxyhip.so.5 =>'in line);assert sha(oldhip)==sha(newhip)==hipsha
    counter=B/'CPU-counter-contract';run(['g++','-std=c++17','-O2','-pthread','-I'+str(S),str(S/'CPU_counter_contract.cpp'),'-o',str(counter)],'CPU-counter-compile');run([str(counter)],'CPU-counter-contract',30)
    run(['python3','-B',str(S/'CPU_all86_oracle.py')],'CPU-all86-oracle',180);oracle=B/'CPU-all86-oracle.json';oq=json.loads(oracle.read_bytes());assert oq['passed']and oq['actual_HC_groups']==86 and oq['oracle_cases']==172 and not oq['GPU_initialized']and not oq['Native_dlopen']
    objects={p:h for p,h in oldobjects.items()if p!=str(oldunit)};unchanged=dict(objects);objects[str(obj)]=sha(obj);objects[str(kernel)]=sha(kernel);assert len(unchanged)==47 and len(objects)==49
    q.update(completed=True,passed=True,parent_Native_SHA256=sha(oldlib),parent_CPU_receipt_SHA256=sha(parent_path),library=str(lib),library_SHA256=sha(lib),replaced_object=str(oldunit),replaced_object_SHA256=sha(oldunit),new_HC_object=str(obj),new_HC_object_SHA256=sha(obj),unchanged_objects_SHA256=unchanged,all_objects_SHA256=objects,link_argv=link,source_SHA256={p.name:sha(p)for p in S.iterdir()if p.is_file()},dependencies_SHA256=before,MD_equals_fresh_M=True,all_original_HC_project_and_system_dependencies_unchanged=True,original_HC_header_unchanged=True,strip_insertions_recovers_original_HC_source=True,all_old_dynamic_exports_retained=True,new_dynamic_exports=sorted(newnames-oldnames),original_NEEDED=oldneeded,new_NEEDED=newneeded,added_direct_NEEDED=sorted(set(newneeded)-set(oldneeded)),dependency_note='HIP object needs directlibgalaxyhip edge; exactsamealreadyloadedparenttransitiveprovider retained, no SDK/library replacement.',existing_HIP_provider=dict(path=str(hip),resolved=str(hip.resolve()),SHA256=hipsha),actual_DTK_bindings=bindings,ldd=ldd,CPU_all86_oracle_SHA256=sha(oracle),HC_module_instances_expected_per_owner=86,Native_dlopen=False,weights_payload_opened=True,only_small_HC_parameters=True,seconds=time.perf_counter()-started,Model_speed='UNMEASURED',GPU='NOT_RUN')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','seconds','error']}));assert q['passed']
