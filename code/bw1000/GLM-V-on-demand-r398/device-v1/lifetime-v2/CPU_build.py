"""Only HOST owner-storage lifetime fix. Reuse CPU796 graph, no dlopen/weights."""
from pathlib import Path
import hashlib,json,subprocess,os,shlex,time,traceback,re
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-V-on-demand-r398-lifetime-v2-CPU';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,caller_object_recompiled=False,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def run(a,label):
 remaining=260-(time.monotonic()-start);assert remaining>0;q['commands'].append(a);z=subprocess.run(a,text=True,capture_output=True,timeout=min(120,remaining));s=z.stdout+z.stderr;(B/(label+'.log')).write_text(s)
 if z.returncode:raise RuntimeError(dict(label=label,RC=z.returncode,tail=s[-6000:]))
 return s
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM398_LIFETIME_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=R/'candidates/GLM-V-on-demand-r398-device-v1-CPU/CPU-build.json';assert sha(p)=='3b4e14f883afade96706b26e19fa7a2322903eec7dea92c7bb1df4a456ced24c';a=json.loads(p.read_bytes());assert a['completed']and sha(a['library'])==a['library_SHA256']=='0c8cffd41a36a230da9f98c9ba191a77eb1c70e6a32f191f5fb8f7b12c24958e'
 old=a['compiled'];assert sha(old['source'])==old['source_SHA256']and sha(old['object'])==old['object_SHA256']
 derivation=json.loads((D/'DERIVATION.json').read_bytes());raw=Path(old['source']).read_bytes();assert sha(old['source'])==derivation['parent_SHA256'];text=raw.decode()
 for e in derivation['edits']:assert text.count(e['old'])==1;text=text.replace(e['old'],e['new'])
 assert text.encode()==(D/'attention_component_r398.cpp').read_bytes()and len(derivation['edits'])==3 and derivation['added_sync']==0 and not derivation['math_changed']
 reuse=dict(a['all_reused_objects_SHA256'])
 for path,h in reuse.items():assert sha(path)==h,path
 for path,h in old['dependencies_SHA256'].items():assert sha(path)==h,path
 for path,h in a['actual_dynamic_dependency_SHA256'].items():assert sha(path)==h,path
 obj=B/'attention_component_r398.cpp.o';md=B/'attention_component_r398.cpp.d';cmd=list(next(c for c in a['commands']if isinstance(c,list)and'-c'in c and'-o'in c and c[c.index('-o')+1]==old['object']));cmd[cmd.index('-c')+1]=str(D/'attention_component_r398.cpp');cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(md);run(cmd,'consumer-compile');q['HOST_TUs_compiled']=1
 link=list(next(c for c in a['commands']if isinstance(c,list)and'-shared'in c));assert link.count(old['object'])==1;link=[str(obj)if x==old['object']else x for x in link];lib=B/'libGLM_V_on_demand_lifetime_r398.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 syms=lambda s:{x.split()[-1]for x in s.splitlines()if x.split()};oldsyms=syms(run(['nm','-D','--defined-only',a['library']],'parent-nm'));newsyms=syms(run(['nm','-D','--defined-only',str(lib)],'candidate-nm'));assert oldsyms<=newsyms
 def ldd(p,label):
  cmd=['ldd','-r',str(p)];q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=30);s=z.stdout+z.stderr;(B/(label+'.log')).write_text(s);assert'not found'not in s;return z.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',s)),s
 r,u,t0=ldd(a['library'],'parent-ldd');r1,u1,t1=ldd(lib,'candidate-ldd');assert(r,u)==(r1,u1)
 q.update(completed=True,source_manifest_SHA256=sha(mf),parent_CPU_receipt_SHA256=sha(p),parent_library_SHA256=a['library_SHA256'],compiled=dict(source=str(D/'attention_component_r398.cpp'),source_SHA256=sha(D/'attention_component_r398.cpp'),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(deps(md))}),all_reused_objects_SHA256=reuse,caller_object_SHA256=a['caller_object_SHA256'],library=str(lib),library_SHA256=sha(lib),all_parent_exports_retained=True,ldd_r_parent_equal=True,actual_dynamic_dependency_SHA256={p:sha(p)for p in set(re.findall(r'=>\s+(/[^\s]+)',t0+t1))},checkpoint_metadata_SHA256=a['checkpoint_metadata_SHA256'],checkpoint_keys_present=a['checkpoint_keys_present'],inherited_CPU796_recording_contract=a['actual_recording_contract'],inherited_CPU796_metric_contract=a['actual_metric_contract'],HOST_selection_lifetime='Work member logical+mapped; destructor drain before member release on normal/throw',added_sync=0,math_changed=False,Runtime_exception_guarantee_assumed=False,GPU_GO=False,Model_GO=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','HOST_TUs_compiled','caller_object_recompiled','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['completed']else 1)
