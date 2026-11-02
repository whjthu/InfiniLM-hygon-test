"""One actual FFN HOST replacement; all headers/GPU objects/SDK unchanged."""
from pathlib import Path
import os,json,hashlib,subprocess,time,traceback,re
R=Path('/refactor');D=R/'incoming/GLM-shared-M1-direct';B=R/'candidates/GLM-shared-M1-direct-cpu-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,weights_opened=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
def run(args,label,timeout=120):
 q['commands'].append(args);p=subprocess.run(args,text=True,capture_output=True,timeout=timeout);(B/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(label+': '+p.stderr[-4000:])
 return p.stdout
def main():
 mf=D/'SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());assert sha(mf)==os.environ['GLM_SHARED_M1_SOURCE_SHA256']
 assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 B.mkdir(exist_ok=False)
 contract=json.loads(run(['python3','-B',str(D/'check_layout_CPU.py')],'CPU-layout'));assert contract['passed'];q['CPU_layout_contract']=contract
 p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(p)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';parent=json.loads(p.read_bytes())
 assert parent['completed']and parent['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and sha(parent['library'])==parent['library_SHA256']
 prior_path=R/'candidates/glm-vector-model-r8542/CPU-build.json';assert sha(prior_path)=='39f873bffae4750a514ff958d9eb411fdfa6f8258901015b40761ed72b5212f4'
 prior=json.loads(prior_path.read_bytes())
 argv=next(list(a)for a in prior['commands']if '-c'in a and a[a.index('-c')+1].endswith('/glm_moe_dsa_routed_ffn_r19.cpp'))
 original=argv[argv.index('-c')+1];assert sha(original)==sha(D/'original_ffn.cpp')=='c716b37fd3fe786cece00a5b4a309d9eb13b76e4bbdb8cc8843be936c708e64b'
 old=R/'candidates/glm-vector-model-r8542/vector-0.o';assert sha(old)=='4b36b711d264f8e7f28ea048d3f6176ffeec9291618ea106ba4a6ba8f3941c41'
 obj=B/'shared-FFN.o';md=B/'shared-FFN.d';argv[argv.index('-c')+1]=str(D/'candidate_ffn.cpp');argv[argv.index('-o')+1]=str(obj);argv[argv.index('-MF')+1]=str(md);argv.insert(1,'-I'+str(D))
 run(argv,'compile-FFN');q['HOST_TUs_compiled']=1
 deps=md.read_text().replace('\\\n',' ').split(':',1)[1].split();q['dependency_SHA256']={n:sha(n)for n in deps};q['MD_SHA256']=sha(md)
 # Byte-identical public headers, with no fields/methods/friends added.
 for name in ['glm_moe_dsa_routed_ffn_r19.hpp','glm_moe_dsa_attention_prepare.hpp','weights.hpp']:
  bound=[n for n in deps if Path(n).name==name];assert bound
  assert all(sha(n)==sha(D/name)==m['files_SHA256'][name]for n in bound)
 link=next(list(a)for a in parent['commands']if isinstance(a,list)and'-shared'in a)
 objects={a:sha(a)for a in link if a.endswith('.o')};assert objects[str(old)]==sha(old)
 for n,h in parent['reused_objects_SHA256'].items():assert objects[n]==h
 library=B/'libGLM_shared_M1_direct.so';link=[str(obj)if a==str(old)else a for a in link];link[link.index('-o')+1]=str(library);run(link,'link')
 oldnm=run(['nm','-D','--defined-only',parent['library']],'parent-exports');newnm=run(['nm','-D','--defined-only',str(library)],'candidate-exports')
 rows=lambda s:{x.split()[-1]:x.split()[-2]for x in s.splitlines()if x.strip()}
 oldsymbols,newsymbols=rows(oldnm),rows(newnm)
 required={'glm_layer_create_r5670','glm_layer_forward_r5670','glm_layer_destroy_r5670','glm_routed_gate_r19','glm_routed_down_r19','glm_shared_M1_mode_v1','glm_shared_M1_counts_v1'}
 assert required<=set(newsymbols)
 missing_strong=[n for n,t in oldsymbols.items()if t.upper()not in ['W','V']and n not in newsymbols];assert not missing_strong,missing_strong
 weak_delta=[n for n in oldsymbols if n not in newsymbols];q['non_public_weak_export_delta_diagnostic']=weak_delta
 dependencies={}
 for tag,path in [('parent',parent['library']),('candidate',str(library))]:
  text=run(['ldd','-r',path],tag+'-ldd',45);assert 'undefined symbol'not in text and'not found'not in text
  for n in re.findall(r'=>\s+(/[^\s]+)',text):dependencies[n]=sha(n)
 assert dependencies[str(R/'candidates/glm-grouped-native-r5123/libnative.so')]=='fa7a9c3c4315f6684227512e87ce67e38a495fe3421d2a0ece8cf09520b28594'
 scalar=R/'candidates/glm-grouped-native-model-r5117/helpers/routed.o';assert sha(scalar)=='e10632453a7b950afd3078076da7f2a68e0828436ee2d5c725045c30e4a4ef69'
 for n,h in objects.items():assert sha(n)==h
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),library=str(library),library_SHA256=sha(library),parent_Native_SHA256=parent['library_SHA256'],parent_CPU_receipt_SHA256=sha(p),
          replaced_object={'path':str(old),'SHA256':objects.pop(str(old))},reused_objects_SHA256=objects,object=str(obj),object_SHA256=sha(obj),candidate_source_SHA256=sha(D/'candidate_ffn.cpp'),
          actual_dynamic_dependencies_SHA256=dependencies,header_or_class_layout_changed=False,GPU_objects_changed=False,raw_provider_object_SHA256=sha(scalar),all_parent_strong_exports_retained=True,
          scope='HOST compile/link/source/layout closure only; no Native load, real GEMV numeric, fullFFN wall or Model qualification. Public5123 route1 remains refused; exact existing raw_r19 provider used only for shared fixed expert0 view.')
if B.exists():raise SystemExit('immutable CPU output exists; no mutation')
try:main()
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:
 B.mkdir(exist_ok=True);q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in['completed','passed','actual_step','error','HOST_TUs_compiled','library_SHA256','seconds']}))
 if not q['passed']:raise SystemExit(1)
