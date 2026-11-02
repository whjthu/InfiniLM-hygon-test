"""Actual afc6 graph: one HC TU replacement plus the exact qualified kernel."""
import pathlib,json,hashlib,copy
from native_binding_parent_r348 import BASE,binding as parent_binding
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
PIN=json.loads((D/'NATIVE_BUILD_PIN.json').read_bytes());NATIVE_SHA=PIN['library_SHA256']
def binding():
 old_binding,old_receipt,old_normal=parent_binding();assert old_binding['library_SHA256']==PIN['parent_Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 p=pathlib.Path(PIN['CPU_receipt']);assert sha(p)==PIN['CPU_receipt_SHA256'];build=json.loads(p.read_bytes())
 assert build['completed']and build['passed']and build['CPU_only']and not build['GPU_initialized']and not build['Native_dlopen']and not build['whole_Target_Model_loaded']
 assert build['library_SHA256']==NATIVE_SHA and sha(build['library'])==NATIVE_SHA
 previous=old_normal['link']['full']['objects'];original=str(R/'candidates/deep-combined-r7169/objects/unit-3.o');assert len(previous)==48 and previous[original]==build['replaced_object_SHA256']
 unchanged={n:h for n,h in previous.items()if n!=original};assert len(unchanged)==47 and unchanged==build['unchanged_objects_SHA256']
 objects=dict(build['all_objects_SHA256']);assert len(objects)==49 and all(sha(n)==h for n,h in objects.items())
 assert build['MD_equals_fresh_M']and build['original_HC_header_unchanged']and build['all_original_HC_project_and_system_dependencies_unchanged']and build['strip_insertions_recovers_original_HC_source']and build['all_old_dynamic_exports_retained']
 for n,h in build['dependencies_SHA256'].items():assert sha(n)==h
 oracle=p.with_name('CPU-all86-oracle.json');assert sha(oracle)==PIN['CPU_all86_oracle_SHA256']==build['CPU_all86_oracle_SHA256'];o=json.loads(oracle.read_bytes());assert o['passed']and o['actual_HC_groups']==86 and o['oracle_cases']==172 and o['prior_CPU700_cases_verified']==8 and not o['GPU_initialized']and not o['Native_dlopen']and not o['whole_Target_Model_loaded']
 assert build['actual_DTK_bindings']==old_receipt['actual_DTK_bindings']
 for n,v in build['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
 hip=build['existing_HIP_provider'];assert sha(hip['path'])==hip['SHA256'];assert build['added_direct_NEEDED']in [[],['libgalaxyhip.so.5']]
 b=dict(old_binding,CPU_receipt=str(p),CPU_receipt_SHA256=PIN['CPU_receipt_SHA256'],library_SHA256=NATIVE_SHA,hc_epilogue_modes_supported=[0,1],hc_epilogue_default_mode=0,hc_epilogue_only_T1_BF16=True,hc_actual_modules_per_owner=86,hc_counter_GPU_readback=False)
 q=copy.deepcopy(old_receipt);q.update(library=build['library'],library_SHA256=NATIVE_SHA,actual_DTK_bindings=build['actual_DTK_bindings'],derived_binding=True,derivation_receipt_SHA256=PIN['CPU_receipt_SHA256'],parent_Native_SHA256=old_binding['library_SHA256'],HC_T1_candidate_default_OFF=True)
 normal=dict(completed=True,CPU_only=True,libraries={'full':{'path':build['library'],'SHA256':NATIVE_SHA}},link={'full':{'objects':objects,'argv':build['link_argv']}},actual_DTK_bindings=build['actual_DTK_bindings'],derived_binding=True,original_guard_proof_preserved=True)
 return b,q,normal
def prebuild():return binding()[2]
def manifest():
 b,q,_=binding();m=json.loads(BASE.with_name('manifest.json').read_bytes());relative=str(pathlib.Path(b['CPU_receipt']).relative_to(R));m['cpu_receipt_path']=relative;m['DSO_SHA256']=NATIVE_SHA;m['existing_remote_SHA256'][relative]=b['CPU_receipt_SHA256'];return m
