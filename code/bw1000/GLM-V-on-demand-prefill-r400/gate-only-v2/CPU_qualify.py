"""Exact private-template gate fix. No compiler/link/Native/device operation."""
from pathlib import Path
import os,json,hashlib,subprocess,re,shlex,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-V-on-demand-prefill-r400-gate-v2-CPU';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,link_invocations=0,caller_object_recompiled=False,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def command(args,label,allow_nonzero=False):
 assert args[0]in['nm','ldd'];q['commands'].append(args);z=subprocess.run(args,text=True,capture_output=True,timeout=25);s=z.stdout+z.stderr;(B/(label+'.log')).write_text(s)
 if not allow_nonzero:assert z.returncode==0,(label,s[-3000:])
 return z.returncode,s
def nm(p,label):
 _,s=command(['nm','-D','--defined-only',str(p)],label);return {x.split()[-1]:x.split()[-2]for x in s.splitlines()if x.split()}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM400_GATE_CPU_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 failed=R/'candidates/GLM-V-on-demand-prefill-r400-CPU/CPU-build.json';assert sha(failed)=='84a1169921ae9d4f296ce39ed3809e207d43da4256538d9be9e3c5809cb7c2a3';f=json.loads(failed.read_bytes());assert not f['completed']and f['producer']=='910407.800'and f['HOST_TUs_compiled']==1 and f['actual_chunk_contract']['passed']and f['actual_recording_contract']['passed']
 parent=R/'candidates/GLM-V-on-demand-r398-lifetime-v2-CPU/CPU-build.json';assert sha(parent)=='92103aba80235b27780abf942a43ad2e4a3cdc80119143297285e768c89d9f7c';p=json.loads(parent.read_bytes());assert p['completed']
 lib=R/'candidates/GLM-V-on-demand-prefill-r400-CPU/libGLM_V_on_demand_prefill_r400.so';assert sha(lib)=='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7';assert sha(p['library'])=='fe3c3f39bc2903b2d6076a1275f25db9304579eccb0f8a6fb494c157e1f75eed'
 obj=lib.parent/'attention_component_r398.cpp.o';md=lib.parent/'attention_component_r398.cpp.d';assert sha(obj)=='d9dfb86aa5547f409d3eda61c72b3cdc39148e01ac505fe3b2e38362672ec5b3'and sha(md)=='9591bb3b9a21399b68b4e0c1aba9263a7d973407ad334a41587b13db8faf7ed6'
 original=R/'incoming/GLM-V-on-demand-prefill-r400';originalMF=original/'CPU_SOURCE_MANIFEST.json';assert sha(originalMF)=='fe58b9c3745fd1ee1a46ecbd7f2873fa13b1d4ff90fe7c7e44fe527742e90c47'
 for n,h in json.loads(originalMF.read_bytes())['files_SHA256'].items():assert sha(original/n)==h,n
 a=nm(p['library'],'parent-nm');b=nm(lib,'candidate-nm');allow=json.loads((D/'SYMBOL_ALLOWLIST.json').read_bytes());allowed={x['mangled']for x in allow['missing_exact']};missing=a.keys()-b.keys()
 assert missing==allowed and len(missing)==2 and all(a[x]=='W'for x in missing)
 assert not [x for x in missing if a[x]!='W']and not[x for x in missing if not x.startswith('_Z')]
 changed_types=[dict(symbol=n,parent=a[n],candidate=b[n])for n in a.keys()&b.keys()if a[n]!=b[n]];assert not changed_types,changed_types
 required=['glm_attention_'+x+'_r398'for x in ['create','keys','load','finish','begin','seed','query','compare_v','control','destroy','error','report']]
 assert set(required)<=b.keys()
 def ldd(path,label):
  rc,s=command(['ldd','-r',str(path)],label,True);assert'not found'not in s;return rc,set(re.findall(r'undefined symbol:\s*([^\s(]+)',s)),s
 rc,u,t0=ldd(p['library'],'parent-ldd');rc1,u1,t1=ldd(lib,'candidate-ldd');assert(rc,u)==(rc1,u1),dict(new_undefined=sorted(u1-u),RC=[rc,rc1])
 dep=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));reuse=dict(p['all_reused_objects_SHA256'])
 for path,h in reuse.items():assert sha(path)==h,path
 compile_cmd=next(c for c in f['commands']if isinstance(c,list)and'-c'in c and'-o'in c);source=Path(compile_cmd[compile_cmd.index('-c')+1]);assert source==original/'attention_component_r398.cpp'
 link=next(c for c in f['commands']if isinstance(c,list)and'-shared'in c);assert link[link.index('-o')+1]==str(lib)
 actual_objects={x:sha(x)for x in link if x.endswith('.o')};assert actual_objects[str(obj)]==sha(obj)
 for path,h in reuse.items():assert actual_objects[path]==h,path
 assert len(actual_objects)==len(reuse)+1
 q.update(completed=True,source_manifest_SHA256=sha(mf),original_source_manifest_SHA256=sha(originalMF),original_CPU800_remains_FAILED=True,failed_CPU_receipt_SHA256=sha(failed),parent_CPU798_receipt_SHA256=sha(parent),library=str(lib),library_SHA256=sha(lib),compiled=dict(source=str(source),source_SHA256=sha(source),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(dep)},compiled_by='910407.800; reused without compile/link'),all_reused_objects_SHA256=reuse,exact_removed_private_weak=allow['missing_exact'],all_other_parent_weak_retained=True,all_parent_strong_retained=True,all_parent_CABI_retained=True,all_parent_exports_retained=False,private_template_exception_only=True,ldd_r_parent_equal=True,preexisting_HOST_imports=sorted(u1),actual_dynamic_dependency_SHA256={p:sha(p)for p in set(re.findall(r'=>\s+(/[^\s]+)',t0+t1))},checkpoint_metadata_SHA256=p['checkpoint_metadata_SHA256'],checkpoint_keys_present=p['checkpoint_keys_present'],actual_chunk_contract_inherited_CPU800=f['actual_chunk_contract'],actual_recording_contract_inherited_CPU800=f['actual_recording_contract'],GPU_GO=False,Model_GO=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','HOST_TUs_compiled','link_invocations','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['completed']else 1)
