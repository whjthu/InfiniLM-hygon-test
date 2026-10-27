"""Bounded HOST build only. No Torch import, DSO loading, weights or GPU."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-prefix-native-qualification-r129';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.monotonic()
def run(a,name,limit=180):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=limit);(B/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-5000:]);return p.stdout
def deps(s):return {str(pathlib.Path(p).resolve())for p in shlex.split(s.replace('\\\n',' ').split(':',1)[1])}
def prefix(argv):
 # Preserve all semantic/ordered flags. Replace only build-output/source flags.
 out=[];i=0
 while i<len(argv):
  x=argv[i]
  if x in ['-o','-MF','-MT','-MQ']:i+=2;continue
  if x in ['-c','-MD','-MMD','-M','-MM']or x.endswith('.cpp'):i+=1;continue
  out.append(x);i+=1
 return out
try:
 m=json.loads((D/'manifest-r129.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 ctx=json.loads((D/'source-contexts-r128.json').read_bytes());parent=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(parent)==ctx['current_CPU_receipt_SHA256']=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c';p=json.loads(parent.read_bytes());assert p['completed']and p['CPU_only'];assert p['libraries']['full']['SHA256']==sha(p['libraries']['full']['path'])=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
 old=ctx['current_link']['objects'];assert old==p['link']['full']['objects'];assert all(sha(n)==h for n,h in old.items());before=dict(old)
 for n,h in p['actual_DTK_bindings'].items():assert sha(n)==h['SHA256'],n
 private_names={pathlib.Path(n).name:str((D/n).resolve())for n in m['private_dependency_names']};compiled=[]
 additions=['-I'+str(D/'src'),'-I'+str(D/'src/engine'),'-I'+str(D),'-I/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc/engine']
 for i,u in enumerate(ctx['units']):
  oldobj=u.get('object');src=D/u['private_source'];assert sha(src)==m['source_SHA256'][u['private_source']]
  if oldobj:assert old[oldobj]==u['object_SHA256']and sha(u['source'])==u['source_SHA256']
  flags=prefix(u['compile_argv']);flags=[flags[0],*additions,*flags[1:]];obj=B/(str(i)+'.o');md=B/(str(i)+'.d')
  fresh=run(flags+['-M',str(src)],str(i)+'-fresh');ds=deps(fresh)
  required={str((D/n).resolve())for n in u['required_private_dependencies']};assert required<=ds,('required-private-header-missing',u['private_source'],sorted(required-ds))
  for dep in ds:
   if pathlib.Path(dep).name in private_names:assert dep==private_names[pathlib.Path(dep).name],('foreign-private-header',dep)
  pins={n:sha(n)for n in ds};cmd=flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(cmd,str(i)+'-compile');assert deps(md.read_text())==ds and all(sha(n)==h for n,h in pins.items())
  compiled.append({'source':str(src),'source_SHA256':sha(src),'replaces':oldobj,'object':str(obj),'object_SHA256':sha(obj),'compile_argv':cmd,'original_compile_context':u['provenance'],'dependency_SHA256':pins,'fresh_M_equal_MD':True,'required_private_headers_present':True})
 replacement={u['replaces']:u['object']for u in compiled if u['replaces']};assert len(replacement)==12
 link=[replacement.get(n,n)for n in ctx['current_link']['argv']];extra=[u['object']for u in compiled if not u['replaces']];assert len(extra)==1;link.insert(link.index('-o'),extra[0]);lib=B/'libDeep_prefix_native_r128.so';link[link.index('-o')+1]=str(lib);run(link,'link',120)
 oldnm=run(['nm','-D','--defined-only',p['libraries']['full']['path']],'old-exports');nm=run(['nm','-D','--defined-only',str(lib)],'new-exports');names=lambda t:{v.split()[-1]for v in t.splitlines()if v.split()};missing=names(oldnm)-names(nm);known_fmt={'_ZNK3fmt3v1217basic_format_argsINS0_7contextEE3getEi','_ZZN3fmt3v126detail12format_floatIdEEiT_iRKNS0_12format_specsEbRNS1_6bufferIcEEENKUljPcE_clEjSA_'};assert missing<=known_fmt,missing
 oldtypes={v.split()[-1]:v.split()[-2]for v in oldnm.splitlines()if len(v.split())>=3};assert all(oldtypes[n]=='W'for n in missing);q['exact_private_fmt_weak_exceptions']={n:oldtypes[n]for n in sorted(missing)};q['all_old_nonexception_symbols_retained']=True
 required=['modern_deep_prefix_'+v+'_r129'for v in ['begin','verify','commit','cancel','ticket']]+['modern_deep_prefix_status_r128'];required+=['modern_deep_prefix_inspect_r129','modern_deep_prefix_fault_r129'];assert set(required)<=names(nm)
 ldd=run(['ldd',str(lib)],'ldd',30);assert 'not found'not in ldd;dynamic=run(['readelf','-d',str(lib)],'dynamic',30);assert 'SYMBOLIC'in dynamic
 contract=B/'prefix_contract';run(['g++','-std=c++17','-O2','-pthread','-I'+str(D),str(D/'prefix_contract_r128.cpp'),'-o',str(contract)],'contract-compile',45);contract_result=json.loads(run([str(contract)],'contract-run',75));assert contract_result['passed']
 assert all(sha(n)==h for n,h in before.items());q.update(completed=True,library=str(lib),library_SHA256=sha(lib),HOST_TUs_compiled=len(compiled),existing_HOST_TUs_replaced=12,new_HOST_TUs=1,reused_objects_SHA256={n:h for n,h in old.items()if n not in replacement},compiled=compiled,link=link,all_old_exports_retained=not bool(missing),actual_DTK_bindings=p['actual_DTK_bindings'],required_exports=required,CPU_contract=contract_result,ldd=ldd,ELF_dynamic=dynamic,source_manifest_SHA256=sha(D/'manifest-r129.json'),default_OFF=True,normal_CPU_exit=True,seconds=time.monotonic()-start,scope='HOST build/source/ABI/link + actual HOST all8 transaction contract only. NativeGPUstate/43layeracceptance/quality/performance/MTPhead unqualified.')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','library_SHA256','error','seconds']}));assert q['completed']
