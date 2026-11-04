"""Qualified afc6 48object factory -> exact a314 50object HC replacement; no dlopen."""
from pathlib import Path
import copy,ctypes as C,hashlib,json
from public_deep_native.native_binding import binding as qualified_binding
PIN=json.loads((Path(__file__).resolve().parents[1]/'MODEL_PIN.json').read_bytes())
NATIVE_SHA=PIN['library_SHA256']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding():
 original,receipt,normal=qualified_binding();assert receipt['library_SHA256']==PIN['baseline_Native_SHA256']
 assert sha(PIN['CPU_receipt'])==PIN['CPU_receipt_SHA256']
 q=json.loads(Path(PIN['CPU_receipt']).read_bytes())
 assert q['completed']and q['passed']and q['CPU_only']and q['actual_step']==PIN['actual_CPU']
 assert not any(q[k]for k in ['GPU_initialized','Native_dlopen','Runtime_created','weights_payload_opened','Model_loaded'])
 assert q['source_manifest_SHA256']==PIN['source_manifest_SHA256']and q['library']==PIN['library']and q['library_SHA256']==NATIVE_SHA and sha(q['library'])==NATIVE_SHA
 assert sha(PIN['parent366_receipt'])==PIN['parent366_receipt_SHA256']==q['parent_CPU_receipt_SHA256']
 p=json.loads(Path(PIN['parent366_receipt']).read_bytes());assert p['passed']and p['library_SHA256']==PIN['parent366_library_SHA256']==q['parent_library_SHA256']
 assert q['old_header_identical']and q['old_CABI_and_exports_retained']and q['control_contract_passed']and q['MD_equals_fresh_M']and q['new_HIP_reused']
 assert not q['added_NEEDED']and not q['removed_NEEDED']
 old=normal['link']['full']['objects'];new=q['all50_objects_SHA256'];assert len(old)==48 and len(new)==50
 unchanged={p:h for p,h in old.items()if new.get(p)==h};assert len(unchanged)==47
 assert set(old)-set(new)=={PIN['replaced_baseline_HC_object']}
 assert set(new)-set(old)=={PIN['new_HC_object'],PIN['old358_object'],PIN['reused372_object']}
 assert new[PIN['old358_object']]==PIN['old358_object_SHA256']and new[PIN['reused372_object']]==PIN['reused372_object_SHA256']
 assert len(q['reused_current48_objects_SHA256'])==48 and all(new[path]==h for path,h in q['reused_current48_objects_SHA256'].items())
 for path,h in new.items():assert sha(path)==h,path
 for path,h in q['dependencies_SHA256'].items():assert sha(path)==h,path
 for path,proof in q['actual_DTK_bindings'].items():assert sha(path)==proof['SHA256']and receipt['actual_DTK_bindings'][path]['SHA256']==proof['SHA256']
 derived=copy.deepcopy(receipt);derived.update(library=q['library'],library_SHA256=NATIVE_SHA,derived_binding=True,HC_fullpre_derivation_receipt_SHA256=PIN['CPU_receipt_SHA256'],actual_Model_unexecuted_CPU_only=True)
 descriptor=dict(CPU_receipt=PIN['CPU_receipt'],CPU_receipt_SHA256=PIN['CPU_receipt_SHA256'],library_SHA256=NATIVE_SHA,rows=1,capacity=32768,prefill_chunk=1024,prefill_chunks_supported=[1024,2048],original_guard_binding=original,Model50objects=True)
 normalized=dict(completed=True,CPU_only=True,libraries={'full':{'path':q['library'],'SHA256':NATIVE_SHA}},link={'full':{'objects':new,'argv':q['link_argv']}},actual_DTK_bindings=derived['actual_DTK_bindings'],original_guard_proof_preserved=True)
 return descriptor,derived,normalized
def validate_mode_functions(lib):
 from .fullpre_control import bind,read_legacy,FULL_NAMES,LEGACY_NAMES
 bind(lib)
 class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int
 actual={}
 for name in FULL_NAMES+LEGACY_NAMES:
  info=Info();assert dl(C.cast(getattr(lib,name),C.c_void_p),C.byref(info));p=Path(info.fname.decode()).resolve()
  assert p==Path(PIN['library']).resolve()and sha(p)==NATIVE_SHA;actual[name]=str(p)
 assert lib.deep_hc_model_epilogue_get_r366()==0 and lib.deep_hc_fullpre_enabled_r377()==0
 rows=read_legacy(lib)
 assert all(r['created']==r['ready']==86 for r in rows)
 return dict(providers=actual,HC0_fixed=True,fullpre_default=0,actual_HC_groups_per_owner=86,old366_width=7,new377_width=10,HostReturned_is_not_GPU_completed=True)
