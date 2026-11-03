"""Only changed diagnostic HOST+link and new typed persistence consumers."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-row-FP8-roundtrip-value-r433-diagnostic-v2-CPU';start=time.monotonic();B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,Native_dlopen=False,Runtime=False,GPU=0,checkpoint_payload_IO=False,production_HOST_TUs_compiled=0,GPU_TUs_compiled=0,link_invocations=0,old872_suites_rerun=False,geometry_recompiled=False,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def run(args,name,allow=False):
 left=260-(time.monotonic()-start);assert left>0;q['commands'].append(args)
 z=subprocess.run(args,capture_output=True,text=True,timeout=min(120,left));text=z.stdout+z.stderr;(B/(name+'.log')).write_text(text)
 if z.returncode and not allow:raise RuntimeError(dict(operation=name,returncode=z.returncode,tail=text[-5000:]))
 return z.returncode,text
def exports(path,name):
 _,text=run(['nm','-D','--defined-only',str(path)],name);return{x.split()[-1]:x.split()[-2]for x in text.splitlines()if len(x.split())>=3}
def ldd(path,name):
 rc,text=run(['ldd','-r',str(path)],name,True);assert'not found'not in text
 return rc,set(re.findall(r'undefined symbol:\s*([^\s(]+)',text)),set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM433_DIAG_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
 parent=R/'candidates/GLM-row-FP8-roundtrip-value-r433-CPU/CPU-build.json';assert sha(parent)=='10531980f62fd38406a4b68b065758bfca5b768b73cfc0ef2475d5ac8f1ca339';a=json.loads(parent.read_bytes())
 assert a['completed']and a['passed']and a['producer']=='910407.872'and a['production_HOST_TUs_compiled']==1 and sha(a['library'])=='f1ea4ee6e0fbb9ff334dbf71ba7dea2b7db592e8b1d8544d3b04dbcdb226ce72'
 old=a['compiled'];assert sha(old['source'])==old['source_SHA256']and sha(old['object'])==old['object_SHA256']and sha(old['MD_path'])==old['MD_SHA256']
 for p,h in old['dependencies_SHA256'].items():assert sha(p)==h,p
 md=set(shlex.split(Path(old['MD_path']).read_text().replace('\\\n',' ').split(':',1)[1]));assert md==set(old['dependencies_SHA256'])
 for p,h in a['reused_objects_SHA256'].items():assert sha(p)==h,p
 assert len(a['reused_objects_SHA256'])==29
 helper=R/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so';assert sha(helper)=='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
 run(['python3','-B',str(D/'check_diagnostic_CPU.py')],'changedDiagnostic.contract');q['actual_new_diagnostic_FFI_rawfile_contract_passed']=True
 cmd=list(next(c for c in a['commands']if isinstance(c,list)and'-c'in c and'-o'in c));assert cmd[cmd.index('-c')+1]==old['source']and cmd[cmd.index('-o')+1]==old['object']
 obj=B/'attention_component_r433_diag_v2.o';mdpath=B/'attention_component_r433_diag_v2.d';cmd[cmd.index('-c')+1]=str(D/'attention_component_r433.cpp');cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(mdpath);cmd.insert(1,'-I'+str(D))
 run(cmd,'component.compile');q['production_HOST_TUs_compiled']=1
 fresh=set(shlex.split(mdpath.read_text().replace('\\\n',' ').split(':',1)[1]));shared=fresh&set(old['dependencies_SHA256']);assert len(shared)>850 and all(sha(p)==old['dependencies_SHA256'][p]for p in shared)
 link=list(next(c for c in a['commands']if isinstance(c,list)and'-shared'in c));assert link.count(old['object'])==1
 prior_objects={p:sha(p)for p in link if p.endswith('.o')};assert prior_objects==a['final_objects_SHA256']
 lib=B/'libGLM_row_FP8_roundtrip_value_r433_diag_v2.so';link=[str(obj)if x==old['object']else x for x in link];link[link.index('-o')+1]=str(lib)
 run(link,'component.link');q['link_invocations']=1
 before=exports(a['library'],'parent.exports');after=exports(lib,'candidate.exports');strong={n:t for n,t in before.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in before.items()if not n.startswith('_Z')}
 assert all(after.get(n)==t for n,t in strong.items())and all(after.get(n)==t for n,t in cabi.items())
 for n in ['arm','desc','read','clear']:assert after['glm_attention_diag_'+n+'_r433']=='T'
 rc,u,p0=ldd(a['library'],'parent.ldd');rc1,u1,p1=ldd(lib,'candidate.ldd');assert(rc,u)==(rc1,u1)and{p:sha(p)for p in p0}=={p:sha(p)for p in p1}
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),parent_CPU_receipt_SHA256=sha(parent),parent_library=a['library'],parent_library_SHA256=sha(a['library']),library=str(lib),library_SHA256=sha(lib),compiled=dict(source=str(D/'attention_component_r433.cpp'),source_SHA256=sha(D/'attention_component_r433.cpp'),object=str(obj),object_SHA256=sha(obj),MD_path=str(mdpath),MD_SHA256=sha(mdpath),dependencies_SHA256={p:sha(p)for p in sorted(fresh)}),reused_objects_SHA256=a['reused_objects_SHA256'],final_objects_SHA256={**a['reused_objects_SHA256'],str(obj):sha(obj)},shared_dependency_count=len(shared),shared_dependencies_exact=True,all_parent_strong_and_CABI_retained=True,private_weak_removed=sorted(set(before)-set(after)),ldd_r_parent_equal=True,actual_dynamic_dependency_SHA256={p:sha(p)for p in sorted(p1)},checkpoint_metadata_SHA256=a['checkpoint_metadata_SHA256'],default_OFF=True,actual_identity_device=False,actual_numeric_device=False,full_attention_runtime=False,GPU_GO=False,Model_GO=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2);f.write('\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','producer','production_HOST_TUs_compiled','link_invocations','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['passed']else 1)
