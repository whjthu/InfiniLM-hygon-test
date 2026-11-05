"""8HOST+5GPU source/ELF closure only. No Native/SDK/Runtime execution."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback,shlex,re,runpy
D=Path(__file__).resolve().parent
O=Path('/refactor/candidates/GLM-global-TP16-full-native-r470-CPU-v1')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dep(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').partition(':')[2]))
def exports(text):return {v.split()[-1]:v.split()[-2]for v in text.splitlines()if len(v.split())>=3}
def needed(text):return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',text)
def loader(text):
 assert 'not found'not in text and'undefined symbol:'not in text,text[-5000:]
 p=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M));assert p;return {x:H(x)for x in p}
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],new_HOST=0,new_GPU=0,Native_dlopen=False,Runtime=False,SDK_execution=False,GPU=False,weights=False,Model_READY=False,commands=[])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(argv,name):
  remaining=265-(time.monotonic()-start);assert remaining>3,(name,'Original265s work cap exhausted')
  q['commands'].append(dict(name=name,argv=argv));save();z=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=remaining-2);(O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr)
  assert z.returncode==0,(name,z.returncode,z.stderr[-3000:]);return z.stdout
 try:
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['GLM470_CLAIM_SHA'];assert os.environ['SLURM_JOB_ID']=='910407';assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/p)==h for p,h in mf['files_SHA256'].items())
  plan=json.loads((D/'BUILD_PLAN.json').read_bytes());allowed=json.loads((D/'DEPENDENCY_PINS.json').read_bytes())['dependency_SHA256'];tools=json.loads((D/'TOOLCHAIN.json').read_bytes())['tools']
  for p,v in tools.items():assert str(Path(p).resolve())==v['canonical']and H(p)==v['sha256'],('Tool identity',p)
  assert os.environ['ROCM_PATH']=='/opt/dtk';q['effective_ROCM_PATH']=os.environ['ROCM_PATH']
  r=plan['parent_receipt'];assert H(r['path'])==r['SHA256'];parent=json.loads(Path(r['path']).read_bytes());assert parent['library_SHA256']==plan['parent_library']['SHA256']and H(parent['library'])==parent['library_SHA256'];assert parent['final_objects_SHA256']==plan['parent_objects_SHA256']
  for p,h in plan['parent_objects_SHA256'].items():assert H(p)==h,('Original object',p)
  for p,h in plan['actual_dynamic_dependency_SHA256'].items():assert H(p)==h,('Same sandbox dynamic dependency',p)
  q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),parent_receipt_SHA256=r['SHA256'],parent_Native_SHA256=parent['library_SHA256'],reused_objects_SHA256=plan['retained_objects_SHA256'],verified_dynamic_dependency_SHA256=plan['actual_dynamic_dependency_SHA256']);save()
  # All13 actual fresh-M closures must succeed BEFORE the first compile. No
  # directory whitelist and no per-missing-header speculative compile loop.
  jobs=[];fresh={}
  for kind,names in [('HOST',plan['HOST_sources']),('GPU',plan['GPU_sources'])]:
   for name in names:
    stem=Path(name).stem;compiler='/usr/bin/g++'if kind=='HOST'else'/opt/dtk/cuda/cuda/bin/nvcc';flags=['-I'+str(D),*(plan['HOST_flags']if kind=='HOST'else plan['GPU_flags'])];source=D/name
    text=run([compiler,*flags,'-M',str(source)],stem+'-M');paths=dep(text);assert paths
    values={p:H(p)for p in paths};fresh[name]=values
    for p,h in values.items():assert (str(Path(p).parent)==str(D)and mf['files_SHA256'].get(Path(p).name)==h)or allowed.get(p)==h,('Unpinned dependency',name,p,h)
    jobs.append((kind,name,compiler,flags))
  assert len(fresh)==13;q['all13_fresh_M_before_any_compile']=True;q['fresh_dependencies_SHA256']=fresh;save();built={}
  for kind,name,compiler,flags in jobs:
   stem=Path(name).stem;obj=O/(stem+'.o');md=O/(stem+'.d');run([compiler,*flags,'-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],stem+'-compile');assert dep(md.read_text())==set(fresh[name])and all(H(p)==h for p,h in fresh[name].items());built[name]=dict(object=str(obj),object_SHA256=H(obj),MD=str(md),MD_SHA256=H(md),source_SHA256=H(D/name));q['new_'+kind]+=1;q['compiled']=built;save()
  assert q['new_HOST']==8 and q['new_GPU']==5
  argv=plan['original_link_argv'][:];seen=set()
  for i,x in enumerate(argv):
   if x in plan['replacements']:name=plan['replacements'][x];seen.add(x);argv[i]=built[name]['object']
  assert seen==set(plan['replacements']);assert len([v for v in argv if v.endswith('.o')])==29
  argv[1:1]=['-L/refactor/candidates/latest-family-native-r269/GLM/python']
  pos=argv.index('-o');argv[pos:pos]=[built[n]['object']for n in plan['additional_sources']];lib=O/'libGLM_global_TP16_full_native_r470.so';argv[argv.index('-o')+1]=str(lib);assert len([v for v in argv if v.endswith('.o')])==31;run(argv,'full-Native-link')
  before=exports(run(['/usr/bin/nm','-D','--defined-only',parent['library']],'parent-exports'));after=exports(run(['/usr/bin/nm','-D','--defined-only',str(lib)],'candidate-exports'));strong={n:t for n,t in before.items()if t not in ['W','V','u','w','v']};assert all(after.get(n)==t for n,t in strong.items()),[n for n,t in strong.items()if after.get(n)!=t]
  required={'modern_glm_global_'+n+'_r470'for n in ['id_size','id','create','error','keys','load_batch','finish','forward','control','reset','geometry','destroy']}|{'glm_r470_i128_validate','glm_r470_i128_gate','glm_r470_i128_down'};assert all(after.get(n)=='T'for n in required)
  x=run(['/usr/bin/readelf','-d',parent['library']],'parent-dynamic');y=run(['/usr/bin/readelf','-d',str(lib)],'candidate-dynamic');dynamic_needed=needed(y);assert needed(x)==dynamic_needed
  x=run(['/usr/bin/ldd','-r',parent['library']],'parent-ldd')+(O/'parent-ldd.stderr').read_text();y=run(['/usr/bin/ldd','-r',str(lib)],'candidate-ldd')+(O/'candidate-ldd.stderr').read_text();oldpaths,newpaths=loader(x),loader(y);assert oldpaths==newpaths,('Loader resolution changed',set(oldpaths.items())^set(newpaths.items()))
  # Only the new Python typed caller is executed; all Native leaves are explicit
  # CFUNCTYPE recording. No puregeometry/1061/967/old suite or SDK dlopen.
  typed=runpy.run_path(str(D/'check_typed_CPU.py'))['run']();assert typed['passed']and not typed['Native_dlopen']
  maps=Path('/proc/self/maps').read_text();assert not re.search(r'lib(?:infiniops|infinirt|infiniccl|infinilm_runtime|torch|amdhip64|galaxyhip)',maps)
  q.update(passed=True,compiled=built,library=str(lib),library_SHA256=H(lib),final_object_count=31,new_CABI=sorted(required),all_original_strong_exports_retained=True,original_strong_count=len(strong),NEEDED_unchanged=dynamic_needed,ldd_r_stdout_and_stderr_retained=True,actual_dynamic_dependency_SHA256=newpaths,all_fresh_M_equal_MD=True,typed=typed,device_qualification=False,cache_floor='UNKNOWN; no Runtime constructor executed',TP8PP2_default_source_unchanged=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','new_HOST','new_GPU','seconds','library_SHA256','error']}));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
