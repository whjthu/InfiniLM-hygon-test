"""Read exact Native/source/object/MD ancestry. CPU-only; no dlopen/code execution."""
import pathlib,json,hashlib
R=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CPU=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json'
CPU_SHA='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529'
NATIVE_SHA='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
def checked_receipt(path,pin):
 assert sha(path)==pin,str(path);q=json.loads(pathlib.Path(path).read_bytes());assert q['completed']and q['CPU_only']and not q['GPU_initialized']and not q['dlopen'];return q
def verify(cpu_path=CPU,cpu_sha=CPU_SHA):
 assert pathlib.Path(cpu_path)==CPU and cpu_sha==CPU_SHA
 q=checked_receipt(CPU,CPU_SHA);assert q['library_SHA256']==NATIVE_SHA and sha(q['library'])==NATIVE_SHA
 assert(q['HOST_TUs_compiled'],q['GPU_TUs_compiled'],q['ABI'])==(1,0,1)
 assert q['all_parent_public_symbols_retained']and q['ldd_r_same_as_qualified_parent']and q['parent_ldd_r_returncode']==q['candidate_ldd_r_returncode']==0
 assert not q['Context_layout_changed']and not q['NativeAttention_class_layout_changed']and q['math_changes']==q['cache_layout_changes']==0
 assert q['PP4_crossrequest_optimization_forced_OFF']and q['PP2_optimization_behavior_unchanged']and q['geometry_contract']['passed']and q['geometry_contract']['exact_members_checked']==156
 source_mf=R/'incoming/GLM-PP4-registry-repair-r184b/source-manifest-r184b.json';assert sha(source_mf)==q['source_manifest_SHA256']=='14692240be90dcff11a3d3b0811155a1ee76c45d1d6e3f8c2be4ebcff1aca5ac'
 for n,h in json.loads(source_mf.read_bytes())['files'].items():assert sha(source_mf.parent/n)==h,n
 parent_path=R/'candidates/GLM-critical-observer-r131/CPU-build.json';parent=checked_receipt(parent_path,q['parent_receipt_SHA256']);assert sha(parent_path)=='8c478b86c887d8f235c3eaa8ee9cd369d053112bc5fb437eb5bd0a8bb9f4a88d'
 assert parent['library_SHA256']==q['parent_native_SHA256']=='ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'and sha(parent['library'])==parent['library_SHA256']
 assert q['parent_fresh_compiled']==parent['compiled']and q['parent_actual_dependency_SHA256']==parent['dependency_SHA256']
 checked=0
 for receipt in[q,parent]:
  for field in['reused_objects_SHA256','actual_binary_SHA256','dependency_SHA256']:
   for p,h in receipt[field].items():assert sha(p)==h,p;checked+=1
  for c in receipt['compiled']:
   assert sha(c['source'])==c['source_SHA256']and sha(c['object'])==c['object_SHA256']and c['fresh_M_equal_MD']and c['full_semantic_argv_replayed']
   md=pathlib.Path(c['object']).with_suffix('.d');assert sha(md)==c['MD_SHA256'],str(md);checked+=3
 original=R/'candidates/GLM-B32-Model-r9183/resident_caller_r9183.cpp.o';old=dict(parent['reused_objects_SHA256']);assert old.pop(str(original))=='363c429f9c0f5a92d8457870a2e6a6e0b41f2ae1668a8f0ec08af0d4aa5f7cdb';assert q['reused_objects_SHA256']==old
 c=q['compiled'][0];assert c['replaces']==str(original)and c['source_SHA256']=='8aa1de7218120cc8d6687fbc3110c419d8651028f6cac92982b03fd5e7578dcd'and c['object_SHA256']=='fa2bc3f46b653c369ffbe9d893ec4784db78639f8cf15a1b699c879f4229c8d3'and c['MD_SHA256']=='eb1ce21975b4a31baa50d13ae0b65afc1d884f46e0f2ca1b4a45f817174c119c'
 before=q['actual_parent_caller_MD'];assert sha(before['path'])==before['SHA256']and sha(before['source_path'])==before['source_SHA256']=='7e8899d4189d9b8dfffe39f8d67efee097b70aafa780202251005674f11619cf'
 return {'cpu_receipt_actual_path':str(CPU),'cpu_receipt_SHA256':CPU_SHA,'library_actual_path':q['library'],'library_declared_path':q['library'],'library_SHA256':NATIVE_SHA,'GPU':False,'dlopen':False,'parent648':parent['parent_native_SHA256'],'parent_ce830_SHA256':parent['library_SHA256'],'M15_default_OFF':True,'M15_stats_count':17,'observer_default_OFF':True,'events_per_owner':512,'PP4_registry_geometry_CPU_passed':True,'HOST_replacement_count':1,'ancestry_files_checked':checked,'constructor_GPU_qualified':False}
