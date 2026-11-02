"""One necessary HOST caller TU + original full038d object graph. No dlopen."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
from derive_caller import derive
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-V-on-demand-Model-integration-r403-CPU'
P=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=time.monotonic();B.mkdir(exist_ok=False)
q=dict(completed=False,CPU_only=True,GPU=0,GPU_initialized=False,Native_dlopen=False,checkpoint_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Model_GO=False,GPU_GO=False)
def run(argv,label):
 q['commands'].append(argv);remaining=260-(time.monotonic()-started)
 if remaining<=0:raise TimeoutError('r403 original inside270 deadline')
 p=subprocess.run(argv,capture_output=True,text=True,timeout=min(120,remaining));text=p.stdout+p.stderr;(B/(label+'.log')).write_text(text)
 if p.returncode:raise RuntimeError(dict(label=label,rc=p.returncode,tail=text[-5000:]))
 return text
def dependencies(md):return set(shlex.split(Path(md).read_text().replace('\\\n',' ').split(':',1)[1]))
def compile_flags(command):
 out=[];i=0
 while i<len(command):
  if command[i]in ['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if command[i]in ['-MD','-MMD','-MP']:i+=1;continue
  out.append(command[i]);i+=1
 return out
def symbols(text):
 out={}
 for line in text.splitlines():
  parts=line.split()
  if len(parts)>=3:out[parts[-1]]=parts[-2]
 return out
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM403_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(P)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';p=json.loads(P.read_bytes())
 assert p['completed']and sha(p['library'])==p['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 old=p['compiled'][0];assert sha(old['source'])==old['source_SHA256']=='8aa1de7218120cc8d6687fbc3110c419d8651028f6cac92982b03fd5e7578dcd'and sha(old['object'])==old['object_SHA256']
 for field in ['dependency_SHA256','parent_actual_dependency_SHA256','actual_binary_SHA256','reused_objects_SHA256']:
  for f,h in p[field].items():assert sha(f)==h,(field,f)
 src,proof=derive(Path(old['source']).read_bytes());assert src==(D/'caller_v_on_demand_r403.cpp').read_bytes();q['exact_source_derivation']=proof
 recipe=next(a for a in p['commands']if isinstance(a,list)and'-c'in a and'-o'in a and a[a.index('-o')+1]==old['object'])
 prefix=compile_flags(recipe);prefix.insert(1,'-I'+str(D))
 exe=B/'host-contract';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_contract.cpp'),'-o',str(exe)],'contract-build');run([str(exe)],'contract-run')
 q['actual_HOST_contract']=dict(passed=True,actual_policy_header=True,default_OFF=True,strict_01=True,original_C1_batchflag1_inactive=True,actual_host_ID_container=True,async_source_after_local_scope_valid=True,exception_drain_before_host_release=True,fixed8192_block_order=True,Native_executed=False)
 obj=B/'caller_v_on_demand_r403.cpp.o';md=B/'caller_v_on_demand_r403.cpp.d'
 run(prefix+['-MD','-MF',str(md),'-c',str(D/'caller_v_on_demand_r403.cpp'),'-o',str(obj)],'caller-compile');q['HOST_TUs_compiled']=1
 ds={f:sha(f)for f in sorted(dependencies(md))}
 q['compiled']=[dict(source=str(D/'caller_v_on_demand_r403.cpp'),source_SHA256=sha(D/'caller_v_on_demand_r403.cpp'),object=str(obj),object_SHA256=sha(obj),MD_path=str(md),MD_SHA256=sha(md),dependencies_SHA256=ds)]
 # Every parent header and project dependency remains the exact original file.
 checked={f:h for f,h in p['dependency_SHA256'].items()if f!=old['source']}
 assert all(f in ds and ds[f]==h for f,h in checked.items()),'Original header/project dependency missing or changed'
 link=list(next(a for a in p['commands']if isinstance(a,list)and'-shared'in a and'-o'in a));assert link.count(old['object'])==1
 original_objects={f:sha(f)for f in link if f.endswith('.o')};assert len(original_objects)==29
 reuse=dict(original_objects);reuse.pop(old['object']);link=[str(obj)if a==old['object']else a for a in link]
 lib=B/'libGLM_V_on_demand_Model_r403.so';link[link.index('-o')+1]=str(lib);run(link,'full-Native-link')
 parent_syms=symbols(run(['nm','-D','--defined-only',p['library']],'parent-nm'));new_syms=symbols(run(['nm','-D','--defined-only',str(lib)],'candidate-nm'))
 missing={n:t for n,t in parent_syms.items()if n not in new_syms};strong={n:t for n,t in parent_syms.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in parent_syms.items()if not n.startswith('_Z')}
 assert all(n in new_syms and new_syms[n]==t for n,t in strong.items()),'Old strong export missing/type changed'
 assert all(n in new_syms and new_syms[n]==t for n,t in cabi.items()),'Old CABI export missing/type changed'
 assert all(n.startswith('_Z')and t in ['W','V','u','w','v']for n,t in missing.items()),missing
 def ldd(path,label):
  argv=['ldd','-r',str(path)];q['commands'].append(argv);pr=subprocess.run(argv,capture_output=True,text=True,timeout=30);text=pr.stdout+pr.stderr;(B/(label+'.log')).write_text(text)
  assert'not found'not in text,text
  return pr.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',text)),text
 a,u,ldp=ldd(p['library'],'parent-ldd');b,v,ldn=ldd(lib,'candidate-ldd');assert(a,u)==(b,v),dict(parent_rc=a,candidate_rc=b,new_undefined=sorted(v-u))
 q.update(completed=True,source_manifest_SHA256=sha(mf),parent_receipt=str(P),parent_receipt_SHA256=sha(P),parent_Native_SHA256=p['library_SHA256'],replaced_object=old,reused_objects_SHA256=reuse,parent_objects_SHA256=original_objects,final_objects_SHA256={**reuse,str(obj):sha(obj)},old_object_count=29,final_object_count=29,library=str(lib),library_SHA256=sha(lib),all_parent_strong_exports_retained=True,all_parent_CABI_retained=True,missing_private_weak_exports=missing,all_parent_exports_retained=not bool(missing),ldd_r_parent_equal=True,preexisting_HOST_imports=sorted(v),actual_dynamic_dependency_SHA256={f:sha(f)for f in set(re.findall(r'=>\s+(/[^\s]+)',ldp+ldn))},original_headers_and_project_dependencies_SHA256=checked,class_header_layout_unchanged=True,default_OFF=True,added_sync=0,provider_GPU_compiled=False,whole_Model_linked=True,actual_device_Model_qualified=False,loaded_service_8GiB_floor_changed=False,forward_512MiB_floor_changed=False,allocator_peak_qualified=False,loader_GPU_staging_peak='UNKNOWN',host_ID_copy_cost='New owned copy in real Model forward wall; not free and not covered by r400 timing')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-started
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','HOST_TUs_compiled','GPU','Native_dlopen','library','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['completed']else 1)
