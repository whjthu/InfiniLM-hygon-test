"""Read existing CPU774 artifacts/logs only; no Slurm/compile/link/dlopen.

The original FAILED CPU report is immutable. Its successful compile/link/ELF
subranges are audited into this separate report. Login-host system libraries
are never substituted for the original container's successful ldd-r evidence.
"""
import hashlib,json,pathlib,re,subprocess
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
B=R/'candidates/GLM-shared-M1-direct-cpu-v1'
S=R/'incoming/GLM-shared-M1-direct'
host=lambda p:pathlib.Path(str(p).replace('/refactor',str(R),1))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(pathlib.Path(p).read_bytes())
cpu=B/'CPU-build.json';assert sha(cpu)=='e4f26bfc0782f5ff0e21f3696c631a455c2359213968a758f3280a544d2f109f'
old=load(cpu);assert old['actual_step']=='910407.774'and not old['passed']and old['HOST_TUs_compiled']==1
assert old['error']=="KeyError('/refactor/candidates/glm-grouped-native-r5123/libnative.so')"
assert old['CPU_layout_contract']['passed']and old['CPU_layout_contract']['checks']==196642
mf=S/'SOURCE_MANIFEST.json';assert sha(mf)=='37c7592fc0cd749a77ebd142e4bb52034b7543087651b723ae52a759ae908468'
assert all(sha(S/n)==h for n,h in load(mf)['files_SHA256'].items())
parent_receipt=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(parent_receipt)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529'
parent=load(parent_receipt);parent_lib=host(parent['library']);assert sha(parent_lib)==parent['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
library=B/'libGLM_shared_M1_direct.so';assert sha(library)=='c5321d47ae13df7da3bffd7f43b263a53c86c2675e422ed435863e9fce37d9a2'
obj=B/'shared-FFN.o';assert sha(obj)=='ffadd66ff34099aabeaef2ad5a4b7bd2d0ac47e41c37dd21ea5bbcf8fd99c7cb'
md=B/'shared-FFN.d';assert sha(md)==old['MD_SHA256']
for p,h in old['dependency_SHA256'].items():
 if p.startswith('/refactor/'):assert sha(host(p))==h
def needed(p):
 return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',subprocess.check_output(['readelf','-d',str(p)],text=True,timeout=20))
assert needed(library)==needed(parent_lib)
def resolved(log):
 text=log.read_text();assert 'not found'not in text and'undefined symbol'not in text
 result={}
 for line in text.splitlines():
  match=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
  if match:result[match[1]]=match[2]
  else:
   match=re.match(r'\s*(/\S+)\s+\(',line)
   if match:result[pathlib.PurePosixPath(match[1]).name]=match[1]
 return result
plog,clog=B/'parent-ldd.log',B/'candidate-ldd.log';paths=resolved(plog);assert paths==resolved(clog)
assert set(needed(library))<=set(paths)|{'ld-linux-x86-64.so.2'}
assert paths['libnative.so']=='/refactor/candidates/glm-hybrid-decode-r5386/libnative.so'
sdk_proof=R/'candidates/GLM-smallM-FFN-r367/CPU-build.json';prior=load(sdk_proof)
assert prior['completed']and prior['CPU_only']and not prior['GPU_initialized']
assert prior['parent_native_SHA256']==parent['library_SHA256']
actual_project_libraries={n:sha(host(n))for n in paths.values()if n.startswith('/refactor/')}
for p,h in actual_project_libraries.items():assert prior['actual_dynamic_dependency_SHA256'][p]==h
def symbols(p):
 text=subprocess.check_output(['nm','-D','--defined-only',str(p)],text=True,timeout=20)
 return {x.split()[-1]:x.split()[-2]for x in text.splitlines()if x.strip()}
ps,cs=symbols(parent_lib),symbols(library)
assert all(n in cs for n,t in ps.items()if t.upper()not in ['W','V'])
assert all(cs[n]=='T'for n in ['glm_routed_gate_r19','glm_routed_down_r19','glm_shared_M1_mode_v1','glm_shared_M1_counts_v1'])
link=next(a for a in old['commands']if '-shared'in a)
objects={p:sha(host(p))for p in link if p.endswith('.o')}
replaced='/refactor/candidates/glm-vector-model-r8542/vector-0.o'
for p,h in parent['reused_objects_SHA256'].items():
 if p!=replaced:assert objects[p]==h
scalar='/refactor/candidates/glm-grouped-native-model-r5117/helpers/routed.o'
assert objects[scalar]=='e10632453a7b950afd3078076da7f2a68e0828436ee2d5c725045c30e4a4ef69'
out={
 'completed':True,'passed':True,'qualification_only':True,'new_Slurm_jobs':0,'new_HOST_compiles':0,'new_GPU_compiles':0,'new_links':0,
 'Native_dlopen':False,'GPU':0,'Model':False,'original_CPU':{'step':'910407.774','state':'FAILED','exit':'1:0','receipt':str(cpu),'receipt_SHA256':sha(cpu),'overall_unchanged':True,
   'successful_subranges':['one FFN HOST compile','full Native link from existing objects','all original strong exports; weak delta empty','same-container parent and candidate ldd-r without unresolved symbols','196642 layout checks']},
 'source_manifest_SHA256':sha(mf),'parent_Native_SHA256':parent['library_SHA256'],
 'library':'/refactor/candidates/GLM-shared-M1-direct-cpu-v1/libGLM_shared_M1_direct.so','library_SHA256':sha(library),
 'object':'/refactor/candidates/GLM-shared-M1-direct-cpu-v1/shared-FFN.o','object_SHA256':sha(obj),'MD_SHA256':sha(md),
 'dependency_SHA256':old['dependency_SHA256'],'all_objects_SHA256':objects,'replaced_object':{'path':replaced,'SHA256':'4b36b711d264f8e7f28ea048d3f6176ffeec9291618ea106ba4a6ba8f3941c41'},
 'link_argv':link,'DT_NEEDED_equal':needed(library),'same_container_ldd_resolved_paths':paths,
 'container_ldd_evidence':[{'path':str(p),'SHA256':sha(p)}for p in [plog,clog]],'actual_project_library_SHA256':actual_project_libraries,
 'existing_SDK_closure_reference':{'path':str(sdk_proof),'SHA256':sha(sdk_proof)},'raw_provider':'current Native static strong raw_r19 from unchanged e106 GPU object',
 'wrong_assumption_corrected':'libnative resolves to existing hybrid5386 in both original038d and candidate, not old5123/fa7. Public5123 wrapper is unchanged; no dependency/SDK/provider replaced.',
 'GPU_GO':False,'GPU_numerics_fullFFN_wall_Model_gain':'NOT_TESTED'}
destination=R/'results/GLM-shared-M1-direct-readonly-qualification-v2.json'
with destination.open('x')as f:f.write(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':True,'qualification_only':True,'path':str(destination),'SHA256':sha(destination),'library_SHA256':sha(library),'actual_libnative_SHA256':actual_project_libraries[paths['libnative.so']],'new_Slurm_jobs':0,'GPU':0}))
