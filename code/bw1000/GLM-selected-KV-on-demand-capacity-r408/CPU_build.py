"""Two necessary HOST TUs, thirty-object Native+component link; no dlopen/GPU."""
from pathlib import Path
import json,hashlib,subprocess,os,time,traceback,shlex,re
from derive_caller import derive
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-selected-KV-on-demand-capacity-r408-CPU';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Model_GO=False,GPU_GO=False)
def run(a,label,allow=False):
 left=260-(time.monotonic()-start);assert left>0;q['commands'].append(a);p=subprocess.run(a,capture_output=True,text=True,timeout=min(100,left));t=p.stdout+p.stderr;(B/(label+'.log')).write_text(t)
 if not allow:assert p.returncode==0,(label,p.returncode,t[-4000:])
 return p.returncode,t
def deptext(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
def nm(p,label):
 _,t=run(['nm','-D','--defined-only',str(p)],label);return {l.split()[-1]:l.split()[-2]for l in t.splitlines()if len(l.split())>=3}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM408_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 pins=json.loads((D/'PARENT_PINS.json').read_bytes());assert sha(pins['Model_receipt'])==pins['Model_receipt_SHA256'];p=json.loads(Path(pins['Model_receipt']).read_bytes());assert p['completed']and sha(p['library'])==p['library_SHA256']==pins['Model_library_SHA256']
 assert sha(pins['component_failed_receipt'])==pins['component_failed_receipt_SHA256'];old=json.loads(Path(pins['component_failed_receipt']).read_bytes());assert old['HOST_TUs_compiled']==1
 assert sha(pins['component_qualified_receipt'])==pins['component_qualified_receipt_SHA256'];qualified=json.loads(Path(pins['component_qualified_receipt']).read_bytes());assert qualified['completed']and sha(qualified['library'])==qualified['library_SHA256']
 assert sha(pins['E32'])==pins['E32_SHA256'];assert all(sha(f)==h for f,h in p['final_objects_SHA256'].items())and len(p['final_objects_SHA256'])==29
 for f,h in p['actual_dynamic_dependency_SHA256'].items():assert sha(f)==h,f
 parent=p['compiled'][0];raw=Path(parent['source']).read_bytes();derived,_=derive(raw);assert derived==(D/'caller_selected_kv_r408.cpp').read_bytes()
 recipes=[('caller_selected_kv_r408',next(a for a in p['commands']if isinstance(a,list)and'-c'in a and'-o'in a and a[a.index('-o')+1]==parent['object']),parent['source'],parent['dependencies_SHA256']),('attention_component_r408',next(a for a in old['commands']if isinstance(a,list)and'-c'in a and'-o'in a),qualified['compiled']['source'],qualified['compiled']['dependencies_SHA256'])]
 objs=[];compiled=[]
 for name,recipe,source,history in recipes:
  cmd=list(recipe);cmd[1:1]=['-I'+str(D),'-I'+str(Path(source).parent),'-ffunction-sections','-fdata-sections'];obj=B/(name+'.o');md=B/(name+'.d');cmd[cmd.index('-c')+1]=str(D/(name+'.cpp'));cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(md)
  fresh=[v for v in cmd if v not in ['-MD','-MMD']];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M';_,t=run(fresh,name+'-fresh-M');before={f:sha(f)for f in deptext(t)}
  for f,h in before.items():
   if Path(f).is_relative_to(D):continue
   if f in history:assert h==history[f],f
   elif f in p['compiled'][0]['dependencies_SHA256']:assert h==p['compiled'][0]['dependencies_SHA256'][f],f
   else:assert not f.startswith('/refactor/'),('Unknown project dependency',f)
  run(cmd,name+'-compile');q['HOST_TUs_compiled']+=1;assert deptext(md.read_text())==set(before)and all(sha(f)==h for f,h in before.items());objs.append(str(obj));compiled.append(dict(source=str(D/(name+'.cpp')),source_SHA256=sha(D/(name+'.cpp')),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),dependencies_SHA256=before))
 shared=['selected_kv_r408.hpp','policy_r408.hpp']
 for n in shared:assert all(c['dependencies_SHA256'][str(D/n)]==sha(D/n)for c in compiled),n
 exe=B/'HOST-policy-chunk-contract';run(['g++',objs[1],'-Wl,--gc-sections','-pthread','-o',str(exe)],'HOST-contract-link');_,ld=run(['ldd',str(exe)],'HOST-contract-ldd');assert not any(t in ld.lower()for t in ['infini','cuda','amdhip'])
 _,s=run([str(exe)],'HOST-actual-policy-chunk');contract=json.loads(s.splitlines()[-1]);assert contract['passed']
 _,s=run(['python3','-B',str(D/'check_CPU.py')],'CPU-whole-consumer');consumer=json.loads(s.splitlines()[-1]);assert consumer['passed']
 link=list(next(a for a in p['commands']if isinstance(a,list)and'-shared'in a and'-o'in a));assert link.count(parent['object'])==1;link=[objs[0]if a==parent['object']else a for a in link];link.insert(1,objs[1]);lib=B/'libGLM_selected_KV_r408.so';link[link.index('-o')+1]=str(lib);link.append('-Wl,-Bsymbolic');run(link,'Native29-plus-component-link')
 oldsyms=nm(p['library'],'parent-nm');newsyms=nm(lib,'candidate-nm');strong={n:t for n,t in oldsyms.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in oldsyms.items()if not n.startswith('_Z')};assert all(newsyms.get(n)==t for n,t in strong.items())and all(newsyms.get(n)==t for n,t in cabi.items())
 missing={n:t for n,t in oldsyms.items()if n not in newsyms};assert all(n.startswith('_Z')and t in ['W','V','u','w','v']for n,t in missing.items())
 assert all('glm_attention_'+n+'_r398'in newsyms for n in ['create','query','compare_k','control','destroy'])
 def ldd(path,label):
  rc,t=run(['ldd','-r',str(path)],label,True);assert'not found'not in t;return rc,set(re.findall(r'undefined symbol:\s*([^\s(]+)',t))
 assert ldd(p['library'],'parent-ldd')==ldd(lib,'candidate-ldd')
 _,dynamic=run(['readelf','-d',str(lib)],'dynamic');needed=[l.split('[',1)[1].split(']',1)[0]for l in dynamic.splitlines()if'NEEDED'in l]
 reused=dict(p['final_objects_SHA256']);reused.pop(parent['object']);assert len(reused)==28;actualobjects=[a for a in link if a.endswith('.o')];assert len(actualobjects)==30 and set(actualobjects)==set(reused)|set(objs)
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),library=str(lib),library_SHA256=sha(lib),compiled=compiled,actual_HOST_contract=contract,actual_CPU_consumer=consumer,all_original28_objects_SHA256=reused,final_objects_SHA256={f:sha(f)for f in actualobjects},final_object_count=30,Native_object_count=29,additional_component_objects=1,actual_dynamic_dependency_SHA256=p['actual_dynamic_dependency_SHA256'],E32_SHA256=sha(pins['E32']),original_class_header_layout_unchanged=True,all_old_strong_and_C_exports_retained=True,missing_private_weak_diagnostic=missing,ldd_r_parent_equal=True,DT_NEEDED=needed,shared_pair_geometry_state_copy_MD_proved=True,fresh_M_before_compile=True,original_source_parent_include_priority=True,added_sync=0,FP8_GPU_provider_recompiled=False,default_Model_NoK_OFF=True,Model_nullable_caller_device_qualified=False,component_numerical_GPU_qualified=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','producer','HOST_TUs_compiled','library_SHA256','elapsed_seconds','error']}));assert q['passed']
