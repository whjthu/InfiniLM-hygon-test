"""0compile: exact CPU85230objects + qualifiedCPU802 consumer.o, 31object DSO."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/GLM-NoV-selected-FP8-plan-r426-component-link-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();B.mkdir(exist_ok=False);start=time.monotonic()
q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,SDK_executed=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,link_invocations=0,commands=[],producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_GO=False,Model_GO=False)
def run(a,name,allow=False):
 q['commands'].append(a);z=subprocess.run(a,capture_output=True,text=True,timeout=min(35,max(1,100-(time.monotonic()-start))));text=z.stdout+z.stderr;(B/(name+'.log')).write_text(text)
 if z.returncode and not allow:raise RuntimeError(dict(name=name,rc=z.returncode,tail=text[-6000:]))
 return z,text
def syms(text):
 out={}
 for l in text.splitlines():
  x=l.split()
  if len(x)>=3:out[x[-1]]=x[-2]
 return out
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['GLM_R426_LINK_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=R/'candidates/GLM-NoV-selected-FP8-plan-r426-CPU-native-v1/CPU-build.json';assert sha(p)=='06f607a40a75543ef225b76464212b89307f510ec98a41cf7cb50cc9a9545b62';a=json.loads(p.read_bytes());assert a['passed']and sha(a['library'])==a['library_SHA256']=='c29e6e5212b51145a7c206df5b2b947b4159771983bb84b5720d4f5419d148a2'
 r=R/'candidates/GLM-V-on-demand-prefill-r400-gate-v2-CPU/CPU-build.json';assert sha(r)=='895304d5617b8bc01cfd291c9ce6165733385ee7d000f2e085561a226b7a2064';c=json.loads(r.read_bytes());assert c['completed']and sha(c['library'])==c['library_SHA256']=='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7'
 obj=c['compiled'];assert sha(obj['object'])==obj['object_SHA256']=='d9dfb86aa5547f409d3eda61c72b3cdc39148e01ac505fe3b2e38362672ec5b3';assert sha(obj['source'])==obj['source_SHA256']=='fcf8441fd5effc8fb1a8397868c02c15ba69ac9471e0ba8b62fe986dedfb6c97'
 md=Path(obj['object']).with_suffix('.d');assert sha(md)==obj['MD_SHA256']=='9591bb3b9a21399b68b4e0c1aba9263a7d973407ad334a41587b13db8faf7ed6'
 actual=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));assert actual==set(obj['dependencies_SHA256'])
 for n,h in obj['dependencies_SHA256'].items():assert sha(n)==h,n
 for n,h in a['final_objects_SHA256'].items():assert sha(n)==h,n
 full=R/'candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json';assert sha(full)==a['parent_receipt_SHA256'];old=json.loads(full.read_bytes());fullheaders=old['compiled'][0]['dependencies_SHA256']
 common={n:h for n,h in obj['dependencies_SHA256'].items()if n.endswith(('.h','.hpp'))and n in fullheaders};assert len(common)==904 and all(fullheaders[n]==h for n,h in common.items())
 for suffix in ['glm_moe_dsa_attention_prepare.hpp','glm_moe_dsa_attention_output.hpp','glm_moe_dsa_sparse_attention.hpp','glm_selection_map_r302.hpp']:
  selected=[n for n in common if n.endswith('/'+suffix)];assert len(selected)==1,suffix
 for n,h in c['actual_dynamic_dependency_SHA256'].items():assert sha(n)==h,n
 link=list(next(x for x in a['commands']if isinstance(x,list)and'-shared'in x));assert'-Wl,-Bsymbolic'in link and'-Wl,--no-undefined'in link
 assert set(x for x in link if x.endswith('.o'))==set(a['final_objects_SHA256'])and obj['object']not in link
 link.insert(link.index('-o'),obj['object']);lib=B/'libGLM_NoV_selected_plan_attention_r426.so';link[link.index('-o')+1]=str(lib);run(link,'component.link');q['link_invocations']=1
 _,parentnm=run(['nm','-D','--defined-only',c['library']],'parent.exports');_,newnm=run(['nm','-D','--defined-only',str(lib)],'component.exports');parent=syms(parentnm);new=syms(newnm)
 strong={n:t for n,t in parent.items()if t not in ['W','V','u','w','v']};cabi={n:t for n,t in parent.items()if not n.startswith('_Z')};assert all(new.get(n)==t for n,t in strong.items()),'Oldcomponent strongABI mismatch';assert all(new.get(n)==t for n,t in cabi.items()),'Oldcomponent CABI mismatch'
 assert {'glm_attention_query_r398','glm_attention_create_r398','glm_attention_destroy_r398','modern_glm_fp8_selected_plan_stats_r426','modern_glm_fp8_selected_plan_clear_r426','glm_fp8_cache_run_r426'}<=new.keys()
 def ldd(path,label):
  z,t=run(['ldd','-r',str(path)],label,allow=True);assert'not found'not in t;return z.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',t)),t
 ar,au,at=ldd(a['library'],'native852.ldd-r');br,bu,bt=ldd(lib,'component.ldd-r');assert(ar,au)==(br,bu),dict(new_undefined=sorted(bu-au))
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),library=str(lib),library_SHA256=sha(lib),CPU852_receipt=str(p),CPU852_receipt_SHA256=sha(p),CPU852_library_SHA256=a['library_SHA256'],consumer_CPU802_receipt=str(r),consumer_CPU802_receipt_SHA256=sha(r),consumer=obj,consumer_MD_path=str(md),all_objects_SHA256={**a['final_objects_SHA256'],obj['object']:obj['object_SHA256']},objects=31,oldconsumer_runtime_ABI_headers_equal=True,shared_header_count=len(common),shared_headers_SHA256=common,all_parent_strong_CABI_retained=True,ldd_r_native852_parent_equal=True,actual_dynamic_dependency_SHA256={n:sha(n)for n in set(re.findall(r'=>\s+(/[^\s]+)',at+bt))},checkpoint_metadata_SHA256=c['checkpoint_metadata_SHA256'],checkpoint_keys_present=c['checkpoint_keys_present'],consumer_byte_recompiled=False,caller_or_provider_recompiled=False,actual_symbol_provider_pending_device=True,actual_attention_or_numeric_qualified=False)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2);f.write('\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','producer','link_invocations','library','library_SHA256','elapsed_seconds','error']}));raise SystemExit(0 if q['passed']else 1)
