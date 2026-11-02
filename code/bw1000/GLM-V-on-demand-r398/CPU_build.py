"""Two HOST TUs + current038d object reuse; no Native dlopen or checkpoint I/O."""
from pathlib import Path
import hashlib,json,subprocess,os,time,traceback,shlex,re
from consumer import recording_contract
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-V-on-demand-r398-CPU'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json'
start=time.monotonic();B.mkdir(exist_ok=False)
q=dict(completed=False,CPU_only=True,GPU=0,GPU_initialized=False,Native_dlopen=False,checkpoint_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def run(a,label):
 q['commands'].append(a);remaining=260-(time.monotonic()-start)
 if remaining<=0:raise TimeoutError('r398 original inside270 budget')
 p=subprocess.run(a,capture_output=True,text=True,timeout=min(120,remaining));text=p.stdout+p.stderr;(B/(label+'.log')).write_text(text)
 if p.returncode:raise RuntimeError(dict(label=label,returncode=p.returncode,tail=text[-6000:]))
 return text
def deps(md):return set(shlex.split(Path(md).read_text().replace('\\\n',' ').split(':',1)[1]))
def flags(cmd):
 out=[];i=0
 while i<len(cmd):
  if cmd[i]in ['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if cmd[i]in ['-MD','-MMD','-MP']:i+=1;continue
  out.append(cmd[i]);i+=1
 return out
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM398_CPU_SOURCE_SHA256'];manifest=json.loads(mf.read_bytes())
 for n,h in manifest['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(P)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';parent=json.loads(P.read_bytes())
 assert parent['completed']and sha(parent['library'])==parent['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 # Reproduce the ACTUAL184b caller recipe, not an unrelated old layer build.
 old=parent['compiled'][0];assert old['source_SHA256']=='8aa1de7218120cc8d6687fbc3110c419d8651028f6cac92982b03fd5e7578dcd'and sha(old['source'])==old['source_SHA256']and sha(old['object'])==old['object_SHA256']
 recipe=next(a for a in parent['commands']if isinstance(a,list)and'-c'in a and'-o'in a and a[a.index('-o')+1]==old['object'])
 prefix=flags(recipe);prefix.insert(1,'-I'+str(D))
 for field in ['dependency_SHA256','parent_actual_dependency_SHA256','actual_binary_SHA256','reused_objects_SHA256']:
  for p,h in parent[field].items():assert sha(p)==h,(field,p)
 q['actual_consumer_recording']=recording_contract();assert q['actual_consumer_recording']['passed']
 # Check source derivation against the original real remote source.
 from derive_caller import derive
 derived,edits=derive(Path(old['source']).read_bytes());assert derived==(D/'caller_v_on_demand_r398.cpp').read_bytes();q['source_edits']=len(edits)
 headers=dict(parent['dependency_SHA256']);selection=next(p for p in headers if p.endswith('/glm_selection_map_r302.hpp'))
 selection_dir=str(Path(selection).parent);exe=B/'selection-contract'
 run(['/usr/bin/g++','-std=c++17','-O2','-I'+selection_dir,str(D/'CPU_selection_contract.cpp'),'-o',str(exe)],'selection-build')
 q['actual_selection_contract']=json.loads(run([str(exe)],'selection-run'));assert q['actual_selection_contract']['passed']
 objects=[];compiled=[]
 for name in ['caller_v_on_demand_r398.cpp','attention_component_r398.cpp']:
  obj=B/(name+'.o');md=B/(name+'.d');run(prefix+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile')
  compiled.append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(deps(md))}));objects.append(str(obj));q['HOST_TUs_compiled']+=1
 link=list(next(a for a in parent['commands']if isinstance(a,list)and'-shared'in a and'-o'in a));assert link.count(old['object'])==1
 reuse={p:sha(p)for p in link if p.endswith('.o')};link=[objects[0]if a==old['object']else a for a in link];link.insert(link.index('-o'),objects[1]);lib=B/'libGLM_V_on_demand_r398.so';link[link.index('-o')+1]=str(lib)
 run(link,'link');syms=lambda s:{v.split()[-1]for v in s.splitlines()if v.split()}
 baseline=syms(run(['nm','-D','--defined-only',parent['library']],'parent-nm'));actual=syms(run(['nm','-D','--defined-only',str(lib)],'candidate-nm'))
 expected=['glm_attention_'+n+'_r398'for n in ['create','keys','load','finish','begin','seed','query','destroy','error','report']]
 assert baseline<=actual and set(expected)<=actual
 # Existingqualified parent's unresolved HOST imports are compared explicitly.
 def ldd(path,label):
  a=['ldd','-r',str(path)];q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=30);s=p.stdout+p.stderr;(B/(label+'.log')).write_text(s);assert'not found'not in s;return p.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',s)),s
 a,u,base_ldd=ldd(parent['library'],'parent-ldd');b,v,new_ldd=ldd(lib,'candidate-ldd');assert a==b and u==v,dict(parent_rc=a,candidate_rc=b,new_undefined=sorted(v-u))
 reuse.pop(old['object']);q.update(completed=True,source_manifest_SHA256=sha(mf),parent_receipt_SHA256=sha(P),parent_Native_SHA256=parent['library_SHA256'],compiled=compiled,reused_objects_SHA256=reuse,replaced_object=old,library=str(lib),library_SHA256=sha(lib),all_parent_exports_retained=True,new_exports=expected,ldd_r_parent_equal=True,preexisting_HOST_imports=sorted(v),actual_dynamic_dependency_SHA256={p:sha(p)for p in set(re.findall(r'=>\s+(/[^\s]+)',base_ldd+new_ldd))},class_header_layout_unchanged=True,default_OFF=True,allocator_peak_qualified=False,device_math_qualified=False,checkpoint_fixture_qualified=False,Model_GO=False,GPU_GO=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','HOST_TUs_compiled','GPU','Native_dlopen','library','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['completed']else 1)
