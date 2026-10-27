"""Proposed repair verifier: no Native TU compile/relink, no DSO load/GPU.

Requires an immutable root-reviewed EXACT missing-symbol allowlist and actual
binary hashes. A wildcard weak-symbol exemption is deliberately impossible.
"""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback
R=pathlib.Path('/refactor');D=R/'incoming/Deep-prefix-native-CPU-metadata-r128b';S=R/'incoming/Deep-prefix-native-r128';B=R/'candidates/Deep-prefix-native-CPU-metadata-r128b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not B.exists();B.mkdir();m=json.loads((D/'manifest-r128b.json').read_bytes());q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'relink':False,'CPU_contract_TUs_compiled':0,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};tick=time.monotonic()
def run(args,name,limit=120):
 q['commands'].append(args);p=subprocess.run(args,text=True,capture_output=True,timeout=limit);(B/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-4000:]);return p.stdout
def nm_rows(text):
 out={}
 for line in text.splitlines():
  v=line.split()
  if len(v)>=3:assert v[-1]not in out;out[v[-1]]=v[-2]
 return out
def deps(text):return {str(pathlib.Path(p).resolve())for p in shlex.split(text.replace('\\\n',' ').split(':',1)[1])}
try:
 assert m['reviewed_exact_missing_symbols']is True
 failpath=R/'candidates/Deep-prefix-native-r128/CPU-build.json';assert sha(failpath)==m['original_failed_receipt_SHA256']=='76ddaf145324582ef46e04310f8ce92e949dd222838d00ec5c3145856e6a73f7';failed=json.loads(failpath.read_bytes());assert not failed['completed']and 'AssertionError'in failed['error']
 assert sha(S/'manifest-r128.json')==m['original_source_manifest_SHA256']=='1cdfb539d5e9fb3a13e45ac588e80ace761a7ca4b1c55beb0f7d878fdb5251be';original=json.loads((S/'manifest-r128.json').read_bytes())
 for name,h in original['source_SHA256'].items():assert sha(S/name)==h,name
 lib=R/'candidates/Deep-prefix-native-r128/libDeep_prefix_native_r128.so';assert sha(lib)==m['library_SHA256'];parent=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(parent)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c';parent=json.loads(parent.read_bytes());oldlib=pathlib.Path(parent['libraries']['full']['path']);assert sha(oldlib)=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
 old=nm_rows(run(['nm','-D','--defined-only',str(oldlib)],'old-nm'));new=nm_rows(run(['nm','-D','--defined-only',str(lib)],'new-nm'));missing={n:old[n]for n in old.keys()-new.keys()};allowed={e['symbol']:e for e in m['missing_symbols_exact']};assert set(missing)==set(allowed)
 for name,kind in missing.items():
  entry=allowed[name];assert kind==entry['old_symbol_type']and kind in ['W','V']and entry['private_compiler_or_implementation_symbol']is True and entry['reason']
  assert name.startswith('_Z')and not name.startswith(('_ZTV','_ZTI','_ZTS')),'Cannot waive a public vtable/typeinfo or a CAPI'
 assert all(n in new and new[n]==kind for n,kind in old.items()if kind not in ['W','V'])
 required=['modern_native_engine_'+v for v in ['create','keys','load','finish','forward','reset','destroy']]+['modern_deep_prefix_'+v+'_r128'for v in ['begin','verify','commit','cancel','status']]
 assert set(required)<=new.keys()
 contexts=json.loads((S/'source-contexts-r128.json').read_bytes());commands=[a for a in failed['commands']if '-c'in a];assert len(commands)==len(contexts['units'])==13
 dependency_hashes={};compiled=[]
 private={pathlib.Path(n).name:str((S/n).resolve())for n in original['private_dependency_names']}
 for i,(unit,cmd)in enumerate(zip(contexts['units'],commands)):
  src=pathlib.Path(cmd[cmd.index('-c')+1]);obj=pathlib.Path(cmd[cmd.index('-o')+1]);md=pathlib.Path(cmd[cmd.index('-MF')+1]);assert str(src)==str(S/unit['private_source'])and obj==R/'candidates/Deep-prefix-native-r128'/(str(i)+'.o')
  # Exact actual successful argv, remove only -MD/-MF/-c/-o operands.
  flags=[];j=0
  while j<len(cmd):
   x=cmd[j]
   if x in ['-MF','-c','-o']:j+=2;continue
   if x=='-MD':j+=1;continue
   flags.append(x);j+=1
  current=deps(run(flags+['-M',str(src)],str(i)+'-fresh-M'));assert current==deps(md.read_text());required_deps={str((S/n).resolve())for n in unit['required_private_dependencies']};assert required_deps<=current
  for dep in current:
   if pathlib.Path(dep).name in private:assert dep==private[pathlib.Path(dep).name]
   dependency_hashes[dep]=sha(dep)
  compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':unit['object'],'original_compile_argv':cmd,'fresh_M_equal_MD':True,'required_private_headers_present':True})
 for name,h in contexts['current_link']['objects'].items():assert sha(name)==h
 ldd=run(['ldd',str(lib)],'ldd',30);assert 'not found'not in ldd;dynamic=run(['readelf','-d',str(lib)],'dynamic',30);assert 'SYMBOLIC'in dynamic
 # Only the small actual HOST contract test is compiled, no Native/math object.
 test=B/'prefix_contract';run(['g++','-std=c++17','-O2','-pthread','-I'+str(S),str(S/'prefix_contract_r128.cpp'),'-o',str(test)],'CPU-contract-compile',45);q['CPU_contract_TUs_compiled']=1;contract=json.loads(run([str(test)],'CPU-contract-run',75));assert contract['passed']and contract['cases']==9
 assert sha(lib)==m['library_SHA256']and sha(failpath)==m['original_failed_receipt_SHA256']
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),original_failed_receipt_SHA256=sha(failpath),original_source_manifest_SHA256=sha(S/'manifest-r128.json'),repair_manifest_SHA256=sha(D/'manifest-r128b.json'),compiled_native_units_reused=13,compiled=compiled,dependency_SHA256=dependency_hashes,exact_missing_symbols_review=m['missing_symbols_exact'],all_old_strong_symbols_retained=True,all_required_CAPI_retained=True,required_exports=required,ABI='PrivateRankWorkerlayoutchanged with12consumerHOSTrebuild; no modelvirtualinterface change. Exact privateweakimplementation compatibility exceptions explicitly listed.',CPU_contract=contract,ldd=ldd,ELF_dynamic=dynamic,default_OFF=True,normal_CPU_exit=True,seconds=time.monotonic()-tick,Native_GPU_qualified=False,Model_gain='UNKNOWN')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-tick)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','library_SHA256','error']}));assert q['completed']
