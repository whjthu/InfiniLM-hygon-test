"""One HOST replacement, immutable parent453 objects; no CUDA/Torch initialization."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-prefix-device-scope-r135';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert not B.exists();B.mkdir();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(a,name,limit=180):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=limit);(B/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-5000:]);return p.stdout
def deps(s):return {str(pathlib.Path(p).resolve())for p in shlex.split(s.replace('\\\n',' ').split(':',1)[1])}
try:
 m=json.loads((D/'manifest-r135.json').read_bytes());assert all(sha(D/n)==h for n,h in m['source_SHA256'].items())
 cp=R/'candidates/Deep-prefix-native-qualification-r129/CPU-build.json';assert sha(cp)=='0961cdd5676987d6741bfa3522f2dcb9a51ae23ec6af8f158653eeb29669c305';p=json.loads(cp.read_bytes());assert p['completed']and p['CPU_only']and sha(p['library'])==p['library_SHA256']=='d90c14f8de06f6913068e00344da0b4aa677b364990fc21e00c83a5caa419861'
 before=dict(p['reused_objects_SHA256']);before.update({u['object']:u['object_SHA256']for u in p['compiled']});assert all(sha(n)==h for n,h in before.items());q['parent_CPU_receipt_SHA256']=sha(cp)
 for n,h in p['actual_DTK_bindings'].items():assert sha(n)==h['SHA256']
 unit=[u for u in p['compiled']if pathlib.Path(u['source']).name=='native_sparse_model_r22.cpp'];assert len(unit)==1;u=unit[0];oldD=str(pathlib.Path(u['source']).parent.parent);oldargv=u['compile_argv'];flags=[];i=0
 while i<len(oldargv):
  x=oldargv[i]
  if x in ['-o','-MF','-MT','-MQ']:i+=2;continue
  if x in ['-MD','-MMD','-M','-MM','-c']or x.endswith('.cpp'):i+=1;continue
  flags.append(x.replace(oldD,str(D)));i+=1
 src=D/'src/native_sparse_model_r22.cpp';obj=B/'native.o';md=B/'native.d';fresh=run(flags+['-M',str(src)],'fresh');ds=deps(fresh);assert str(D/'deep_prefix_device_scope_r135.hpp')in ds
 # All private dependencies must resolve to exact r135 copies with pinned bytes.
 oldprivate=json.loads((D/'manifest-r129.json').read_bytes())['private_dependency_names']
 private={pathlib.Path(n).name:str((D/n).resolve())for n in oldprivate}
 for n in ds:
  if pathlib.Path(n).name in private:assert n==private[pathlib.Path(n).name],n
 pins={n:sha(n)for n in ds};cmd=flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(cmd,'compile');assert deps(md.read_text())==ds and all(sha(n)==h for n,h in pins.items())
 link=[str(obj)if n==u['object']else n for n in p['link']];lib=B/'libDeep_prefix_device_r135.so';link[link.index('-o')+1]=str(lib);run(link,'link',120)
 oldnm=run(['nm','-D','--defined-only',p['library']],'old-exports');nm=run(['nm','-D','--defined-only',str(lib)],'exports');names=lambda s:{v.split()[-1]for v in s.splitlines()if v.split()};assert names(oldnm)<=names(nm);assert 'modern_deep_prefix_device_contract_r135'in names(nm)
 ldd=run(['ldd','-r',str(lib)],'ldd',60);assert 'not found'not in ldd and 'undefined symbol:'not in ldd;dynamic=run(['readelf','-d',str(lib)],'dynamic');assert 'SYMBOLIC'in dynamic
 test=B/'device_scope_mock';run(['g++','-std=c++17','-O2','-I'+str(D),str(D/'device_scope_mock_r135.cpp'),'-o',str(test)],'mock-compile');mock=json.loads(run([str(test)],'mock-run'));assert mock['passed']
 assert all(sha(n)==h for n,h in before.items());q.update(completed=True,library=str(lib),library_SHA256=sha(lib),HOST_TUs_compiled=1,new_HOST_TUs=0,reused_objects_SHA256={n:h for n,h in before.items()if n!=u['object']},compiled=[{'source':str(src),'source_SHA256':sha(src),'replaces':u['object'],'object':str(obj),'object_SHA256':sha(obj),'compile_argv':cmd,'dependency_SHA256':pins,'fresh_M_equal_MD':True}],link=link,actual_DTK_bindings=p['actual_DTK_bindings'],required_exports=p['required_exports']+['modern_deep_prefix_device_contract_r135'],all_old_exports_retained=True,CPU_mock=mock,ldd=ldd,ELF_dynamic=dynamic,source_manifest_SHA256=sha(D/'manifest-r135.json'),default_OFF=True,normal_CPU_exit=True,seconds=time.monotonic()-start,scope='Scoped raw driver query binding; strict current==Runtime gate retained. No model/43layerGPU qualification yet.')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','library_SHA256','error','seconds']}));assert q['completed']
