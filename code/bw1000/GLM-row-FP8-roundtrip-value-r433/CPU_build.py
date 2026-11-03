"""One private component HOST replacement, reuse exact CPU802 object graph.

No Native dlopen, Runtime, device, checkpoint payload load, or GPU compilation.
"""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-row-FP8-roundtrip-value-r433-CPU'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();B.mkdir(exist_ok=False)
q=dict(completed=False,passed=False,Native_dlopen=False,Runtime=False,GPU=0,checkpoint_payload_IO=False,
 production_HOST_TUs_compiled=0,GPU_TUs_compiled=0,CPU_geometry_contract_compiled=False,link_invocations=0,
 commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_GO=False,Model_GO=False)
def run(args,name,allow=False):
 left=260-(time.monotonic()-start);assert left>0;q['commands'].append(args)
 z=subprocess.run(args,capture_output=True,text=True,timeout=min(120,left));text=z.stdout+z.stderr;(B/(name+'.log')).write_text(text)
 if z.returncode and not allow:raise RuntimeError(dict(operation=name,returncode=z.returncode,tail=text[-5000:]))
 return z.returncode,text
def exports(path,label):
 _,text=run(['nm','-D','--defined-only',str(path)],label);return {x.split()[-1]:x.split()[-2]for x in text.splitlines()if len(x.split())>=3}
def ldd(path,label):
 rc,text=run(['ldd','-r',str(path)],label,True);assert'not found'not in text
 paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
 return rc,set(re.findall(r'undefined symbol:\s*([^\s(]+)',text)),paths
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM433_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 helper=R/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so';assert sha(helper)=='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
 run(['python3','-B',str(D/'check_CPU.py')],'changedPython.contract');q['actual_changedPython_FFI_source_contract_passed']=True
 parent=R/'candidates/GLM-V-on-demand-prefill-r400-gate-v2-CPU/CPU-build.json'
 assert sha(parent)=='895304d5617b8bc01cfd291c9ce6165733385ee7d000f2e085561a226b7a2064'
 a=json.loads(parent.read_bytes());assert a['completed']and sha(a['library'])=='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7'
 old=a['compiled'];assert old['source_SHA256']=='fcf8441fd5effc8fb1a8397868c02c15ba69ac9471e0ba8b62fe986dedfb6c97'
 for p,h in old['dependencies_SHA256'].items():assert sha(p)==h,p
 assert sha(old['object'])==old['object_SHA256']and sha(old['source'])==old['source_SHA256']
 md_old=Path(old['object']).with_suffix('.d');assert sha(md_old)==old['MD_SHA256']
 assert set(shlex.split(md_old.read_text().replace('\\\n',' ').split(':',1)[1]))==set(old['dependencies_SHA256'])
 for p,h in a['all_reused_objects_SHA256'].items():assert sha(p)==h,p
 original=R/'candidates/GLM-V-on-demand-prefill-r400-CPU/CPU-build.json'
 assert sha(original)=='84a1169921ae9d4f296ce39ed3809e207d43da4256538d9be9e3c5809cb7c2a3'
 failed=json.loads(original.read_bytes());assert failed['HOST_TUs_compiled']==1
 cmd=list(next(c for c in failed['commands']if isinstance(c,list)and'-c'in c and'-o'in c))
 assert cmd[cmd.index('-c')+1]==old['source']and cmd[cmd.index('-o')+1]==old['object']
 obj=B/'attention_component_r433.o';md=B/'attention_component_r433.d'
 cmd[cmd.index('-c')+1]=str(D/'attention_component_r433.cpp');cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(md);cmd.insert(1,'-I'+str(D))
 run(cmd,'component.compile');q['production_HOST_TUs_compiled']=1
 fresh=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));shared=fresh&set(old['dependencies_SHA256'])
 assert len(shared)>850 and all(sha(p)==old['dependencies_SHA256'][p]for p in shared)
 for suffix in ['runtime/tensor.hpp','glm_moe_dsa_attention_prepare.hpp','glm_moe_dsa_attention_output.hpp']:
  found=[p for p in shared if p.endswith('/'+suffix)];assert len(found)==1,suffix
  assert not[other for other in fresh if other.endswith('/'+suffix)and other not in found],suffix
 contract=B/'CPU_geometry';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_geometry.cpp'),'-o',str(contract)],'geometry.compile')
 q['CPU_geometry_contract_compiled']=True;run([str(contract)],'geometry.run');q['actual_geometry_contract_passed']=True
 link=list(next(c for c in failed['commands']if isinstance(c,list)and'-shared'in c));assert link.count(old['object'])==1
 prior_objects={p:sha(p)for p in link if p.endswith('.o')};assert len(prior_objects)==len(a['all_reused_objects_SHA256'])+1
 assert set(prior_objects)==set(a['all_reused_objects_SHA256'])|{old['object']}
 lib=B/'libGLM_row_FP8_roundtrip_value_r433.so';link=[str(obj)if x==old['object']else x for x in link];link[link.index('-o')+1]=str(lib)
 run(link,'component.link');q['link_invocations']=1
 before=exports(a['library'],'parent.exports');after=exports(lib,'candidate.exports')
 strong={n:t for n,t in before.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in before.items()if not n.startswith('_Z')}
 assert all(after.get(n)==t for n,t in strong.items())and all(after.get(n)==t for n,t in cabi.items())
 assert after['glm_attention_value_roundtrip_r433']=='T'
 rc,u,p0=ldd(a['library'],'parent.ldd');rc1,u1,p1=ldd(lib,'candidate.ldd');assert(rc,u)==(rc1,u1)
 assert {p:sha(p)for p in p0}=={p:sha(p)for p in p1},'Exact actual dynamic provider paths/hashes changed'
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),parent_CPU_receipt_SHA256=sha(parent),
  parent_library=a['library'],parent_library_SHA256=sha(a['library']),library=str(lib),library_SHA256=sha(lib),
  compiled=dict(source=str(D/'attention_component_r433.cpp'),source_SHA256=sha(D/'attention_component_r433.cpp'),object=str(obj),object_SHA256=sha(obj),MD_path=str(md),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(fresh)}),
  reused_objects_SHA256=a['all_reused_objects_SHA256'],final_objects_SHA256={**a['all_reused_objects_SHA256'],str(obj):sha(obj)},
  shared_dependency_count=len(shared),shared_dependencies_exact=True,all_parent_strong_and_CABI_retained=True,
  private_weak_removed=sorted(set(before)-set(after)),ldd_r_parent_equal=True,actual_dynamic_dependency_SHA256={p:sha(p)for p in sorted(p1)},
  checkpoint_metadata_SHA256=a['checkpoint_metadata_SHA256'],default_OFF=True,old_CPU_suites_rerun=False,shared_GPU_provider_recompiled=False,original_helper776_SHA256=sha(helper),helper_F32_device_qualified=False,
  actual_numeric_device=False,full_attention_runtime=False,scope='One private consumer HOST+pure geometry contract; all SDK/Native provider objects reused; no dlopen or device qualification')
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2);f.write('\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','producer','production_HOST_TUs_compiled','CPU_geometry_contract_compiled','link_invocations','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['passed']else 1)
