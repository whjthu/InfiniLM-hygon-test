"""Compile only appended diagnostic consumer TU; reuse both prior graphs, no dlopen."""
from pathlib import Path
import os,json,hashlib,subprocess,traceback,time,shlex,re
from check_CPU import run as check_contract
from checkpoint_loader import KEYS,PREFIX
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-V-on-demand-r398-device-v1-CPU';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def run(args,label):
 q['commands'].append(args);remaining=260-(time.monotonic()-start);assert remaining>0
 p=subprocess.run(args,text=True,capture_output=True,timeout=min(120,remaining));s=p.stdout+p.stderr;(B/(label+'.log')).write_text(s)
 if p.returncode:raise RuntimeError(dict(label=label,RC=p.returncode,tail=s[-5000:]))
 return s
def deps(md):return set(shlex.split(Path(md).read_text().replace('\\\n',' ').split(':',1)[1]))
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM398_DEVICE_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 parent=R/'candidates/GLM-V-on-demand-r398-CPU/CPU-build.json';assert sha(parent)=='399f26b7668f3fa21a22ab81a041cb5b1b1835aad50a3d18fa6a6ae8196f311b';a=json.loads(parent.read_bytes());assert a['completed']and sha(a['library'])==a['library_SHA256']=='431c5b85730549861202ed9aff1b479fa3d23727a1af575777b2c6fc915f88bc'
 old=next(x for x in a['compiled']if x['source'].endswith('/attention_component_r398.cpp'));caller=next(x for x in a['compiled']if x['source'].endswith('/caller_v_on_demand_r398.cpp'))
 reused=dict(a['reused_objects_SHA256'])
 for x in a['compiled']:
  assert sha(x['source'])==x['source_SHA256']and sha(x['object'])==x['object_SHA256']
  for p,h in x['dependencies_SHA256'].items():assert sha(p)==h,p
  reused[x['object']]=x['object_SHA256']
 for p,h in reused.items():assert sha(p)==h,p
 for p,h in a['actual_dynamic_dependency_SHA256'].items():assert sha(p)==h,p
 assert (D/'attention_component_r398.cpp').read_bytes().startswith(Path(old['source']).read_bytes())
 q['actual_recording_contract']=check_contract();assert q['actual_recording_contract']['passed']
 exe=B/'metrics-contract';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_metrics_contract.cpp'),'-o',str(exe)],'metrics-build');q['actual_metric_contract']=json.loads(run([str(exe)],'metrics-run'));assert q['actual_metric_contract']['passed']
 cmd=list(next(x for x in a['commands']if isinstance(x,list)and'-c'in x and'-o'in x and x[x.index('-o')+1]==old['object']));obj=B/'attention_component_r398.cpp.o';md=B/'attention_component_r398.cpp.d';cmd[cmd.index('-c')+1]=str(D/'attention_component_r398.cpp');cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(md);cmd.insert(1,'-I'+str(D));run(cmd,'consumer-compile');q['HOST_TUs_compiled']=1
 link=list(next(x for x in a['commands']if isinstance(x,list)and'-shared'in x));assert link.count(old['object'])==1;link=[str(obj)if x==old['object']else x for x in link];lib=B/'libGLM_V_on_demand_diagnostic_r398.so';link[link.index('-o')+1]=str(lib);run(link,'link');reused.pop(old['object']);assert reused[caller['object']]==caller['object_SHA256']
 syms=lambda s:{l.split()[-1]for l in s.splitlines()if l.split()};oldsyms=syms(run(['nm','-D','--defined-only',a['library']],'old-nm'));newsyms=syms(run(['nm','-D','--defined-only',str(lib)],'new-nm'));extra=['glm_attention_compare_v_r398','glm_attention_control_r398'];assert oldsyms<=newsyms and set(extra)<=newsyms
 def ldd(p,label):
  args=['ldd','-r',str(p)];q['commands'].append(args);z=subprocess.run(args,text=True,capture_output=True,timeout=30);s=z.stdout+z.stderr;(B/(label+'.log')).write_text(s);assert'not found'not in s;return z.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',s)),s
 rc,u,s0=ldd(a['library'],'parent-ldd');rc1,u1,s1=ldd(lib,'candidate-ldd');assert(rc,u)==(rc1,u1)
 model=Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_bytes());index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];assert all(PREFIX+k in index for k in KEYS)and cfg['hidden_size']==6144 and cfg['num_attention_heads']==64 and cfg['max_position_embeddings']>=32768
 q.update(completed=True,source_manifest_SHA256=sha(mf),parent_CPU_receipt_SHA256=sha(parent),parent_library_SHA256=a['library_SHA256'],old_consumer_body_byteidentical_prefix=True,caller_object_recompiled=False,caller_object_SHA256=caller['object_SHA256'],all_reused_objects_SHA256=reused,compiled=dict(source=str(D/'attention_component_r398.cpp'),source_SHA256=sha(D/'attention_component_r398.cpp'),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(deps(md))}),library=str(lib),library_SHA256=sha(lib),all_parent_exports_retained=True,new_exports=extra,ldd_r_parent_equal=True,actual_dynamic_dependency_SHA256={p:sha(p)for p in set(re.findall(r'=>\s+(/[^\s]+)',s0+s1))},checkpoint_metadata_SHA256={n:sha(model/n)for n in ['config.json','model.safetensors.index.json']},checkpoint_keys_present=KEYS,checkpoint_metadata_only=True,device_numeric_qualified=False,GPU_GO=False,Model_GO=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','HOST_TUs_compiled','caller_object_recompiled','library','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['completed']else 1)
