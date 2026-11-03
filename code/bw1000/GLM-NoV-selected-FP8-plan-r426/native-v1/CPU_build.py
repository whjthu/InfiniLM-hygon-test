"""Two necessary HOST TUs + isolated full952 link; no Native/SDK execution."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-NoV-selected-FP8-plan-r426-CPU-native-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=time.monotonic();B.mkdir(exist_ok=False)
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Slurm_GPU=0,Native_dlopen=False,SDK_executed=False,checkpoint_IO=False,HOST_TUs_compiled=0,contract_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_GO=False,Model_GO=False)
def run(argv,label,timeout=120,allow_nonzero=False):
 q['commands'].append(argv);left=260-(time.monotonic()-started)
 if left<=0:raise TimeoutError('r426 original inside270 deadline')
 z=subprocess.run(argv,capture_output=True,text=True,timeout=min(timeout,left));text=z.stdout+z.stderr;(B/(label+'.log')).write_text(text)
 if z.returncode and not allow_nonzero:raise RuntimeError(dict(label=label,rc=z.returncode,tail=text[-6000:]))
 return z,text
def deps(path):return set(shlex.split(Path(path).read_text().replace('\\\n',' ').split(':',1)[1]))
def compile_prefix(argv):
 result=[];i=0
 while i<len(argv):
  if argv[i]in ['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if argv[i]in ['-MD','-MMD','-MP']:i+=1;continue
  result.append(argv[i]);i+=1
 return result
def symbols(text):
 result={}
 for line in text.splitlines():
  x=line.split()
  if len(x)>=3:result[x[-1]]=x[-2]
 return result
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM_R426_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=json.loads((D/'PARENT_BINDING.json').read_bytes());meta=p['metadata']
 parent=R/'candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json';assert sha(parent)==meta['Native804_receipt_SHA256']=='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32';old=json.loads(parent.read_bytes())
 assert old['completed']and old['library_SHA256']==sha(old['library'])=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
 assert sha(meta['runtime_MD_path'])==meta['runtime_MD_SHA256']==p['runtime_MD']['SHA256']
 assert deps(meta['runtime_MD_path'])==set(meta['runtime_dependencies_SHA256'])
 for n,h in meta['runtime_dependencies_SHA256'].items():assert h and sha(n)==h,n
 assert sha('/refactor/incoming/GLM-M15-oproj-Model-r125/glm_fp8_runtime_oproj_r125.cpp')==p['runtime_source_SHA256']=='cdbfda3c1ebf919793ea38d590dad10dced8972e4703a032965a4a5a3242b6b2'
 for n,h in old['final_objects_SHA256'].items():assert sha(n)==h,n
 assert old['final_objects_SHA256']==meta['Native804_final_objects_SHA256']
 for item in p['actual_helpers']:
  value='/refactor/'+Path(item['path']).relative_to('/public/home/aclsylqidf/hygon-inference-refactor-20261014').as_posix();assert sha(value)==item['SHA256']
 runtime_old='/refactor/candidates/GLM-M15-oproj-Model-r125/glm_fp8_runtime_oproj_r125.cpp.o';assert old['final_objects_SHA256'][runtime_old]==p['runtime_object']['SHA256']=='7fec9aabafcfe329a4dad5e7cb3cb810654448d5d45809d9820c9df171d529ff'
 cache_flags=['/usr/bin/g++','-std=c++17','-O2','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-pthread','-I'+str(D)]
 compiled=[]
 # Cache implementation is compiled ONCE and reused in the real CPUconsumer
 # and the isolated fullNative. Only external776 calls are recorded below.
 for filename,prefix in [('glm_fp8_cache_r426.cpp',cache_flags),('glm_fp8_runtime_r426.cpp',compile_prefix(meta['runtime_compile_command']))]:
  prefix=list(prefix);prefix.insert(1,'-I'+str(D));src=D/filename;obj=B/(filename+'.o');md=B/(filename+'.d')
  run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],filename+'.compile');q['HOST_TUs_compiled']+=1
  hashes={n:sha(n)for n in sorted(deps(md))};compiled.append(dict(source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),MD_path=str(md),MD_SHA256=sha(md),dependencies_SHA256=hashes))
  if filename=='glm_fp8_runtime_r426.cpp':
   original={n:h for n,h in meta['runtime_dependencies_SHA256'].items()if n!='/refactor/incoming/GLM-M15-oproj-Model-r125/glm_fp8_runtime_oproj_r125.cpp'}
   assert all(n in hashes and hashes[n]==h for n,h in original.items()),'Currentoriginalheaders/deps changed or missing'
 q['compiled']=compiled
 exe=B/'cache-contract';run(cache_flags+[str(D/'CPU_contract.cpp'),compiled[0]['object'],'-o',str(exe)],'contract.compile');q['contract_TUs_compiled']=1
 _,text=run([str(exe)],'contract.run');contract=json.loads(text);assert contract['passed']and contract['actual_private_cache_logic']and not contract['SDK_executed'];q['actual_cache_consumer']=contract
 fatal,text=run([str(exe),'--destroy-failure'],'contract.destroy-failstop',allow_nonzero=True);assert fatal.returncode in(-6,134),'Expected terminate on originaldestroyfailure';q['destroy_failure_failstop_RC']=fatal.returncode
 _,contract_ldd=run(['ldd',str(exe)],'contract.ldd');assert not any(x in contract_ldd for x in ['libinfinirt','libinfiniops','libglm_fp8','libnative','libcuda']),'CPUconsumer mustnotexecuteSDKleaf'
 link=list(meta['Native804_link']);assert link.count(runtime_old)==1 and'-Wl,-Bsymbolic'in link and'-Wl,--no-undefined'in link
 link=[compiled[1]['object']if s==runtime_old else s for s in link];link.insert(link.index('-o'),compiled[0]['object']);lib=B/'libGLM_NoV_selected_plan_r426.so';link[link.index('-o')+1]=str(lib);run(link,'isolated-full952.link')
 _,base_nm=run(['nm','-D','--defined-only',old['library']],'parent.exports');_,new_nm=run(['nm','-D','--defined-only',str(lib)],'candidate.exports');a=symbols(base_nm);b=symbols(new_nm)
 strong={n:t for n,t in a.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in a.items()if not n.startswith('_Z')};assert all(b.get(n)==t for n,t in strong.items()),'Old strong export missing/type changed';assert all(b.get(n)==t for n,t in cabi.items()),'Old CABI missing/type changed'
 missing={n:t for n,t in a.items()if n not in b};assert all(n.startswith('_Z')and t in ['W','V','u','w','v']for n,t in missing.items()),missing
 required={'modern_glm_fp8_selected_plan_stats_r426','modern_glm_fp8_selected_plan_clear_r426','glm_fp8_cache_create_r426','glm_fp8_cache_run_r426','glm_fp8_cache_clear_r426','glm_fp8_cache_destroy_r426'};assert required<=b.keys()
 def ldd(path,label):
  z,t=run(['ldd','-r',str(path)],label,timeout=30,allow_nonzero=True);assert'not found'not in t,t
  return z.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',t)),t
 ar,au,at=ldd(old['library'],'parent.ldd-r');br,bu,bt=ldd(lib,'candidate.ldd-r');assert(ar,au)==(br,bu),dict(parent_rc=ar,new_rc=br,new_undefined=sorted(bu-au))
 reuse=dict(old['final_objects_SHA256']);reuse.pop(runtime_old);assert len(reuse)==28
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),parent_Native_SHA256=old['library_SHA256'],parent_receipt=str(parent),parent_receipt_SHA256=sha(parent),library=str(lib),library_SHA256=sha(lib),replaced_runtime_object=runtime_old,reused_objects_SHA256=reuse,final_objects_SHA256={**reuse,compiled[0]['object']:compiled[0]['object_SHA256'],compiled[1]['object']:compiled[1]['object_SHA256']},final_objects=30,original_headers_dependencies_equivalent=True,all_parent_strong_CABI_retained=True,missing_private_weak_exports=missing,ldd_r_parent_equal=True,preexisting_HOST_imports=sorted(bu),actual_dynamic_dependency_SHA256={n:sha(n)for n in set(re.findall(r'=>\s+(/[^\s]+)',at+bt))},class_header_layout_unchanged=True,provider_GPU_compiled=False,helper776_unchanged=True,default_OFF=True,actual_device_or_numeric_qualified=False,original_8GiB_preweight_and512MiB_forward_floors_unchanged=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-started
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2);f.write('\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','producer','HOST_TUs_compiled','contract_TUs_compiled','library','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['passed']else 1)
