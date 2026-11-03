"""Actual911 binder -> exact921 TextModel replacement; no library loading here."""
import copy,hashlib,importlib.util,json,re
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OLD_MODEL='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9'
MODE_NAMES=['deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']]
CAPTURE_NAMES=['deep_boundary_'+n+'_r447'for n in ['arm','seal','abandon','clear','count','info','copy']]

def validate_pin(pin,*,require_ready):
 assert pin['CPU_READY']and pin['actual_CPU']=='910407.921'
 assert pin['Native_object_count']==49 and pin['reused_parent_objects']==48 and pin['new_HOST_TUs']==1
 assert pin['parent_library_SHA256']=='1b81667f28892a840c8396c57d9d8134ca40f94f6d0fa7c7c2deab5d1782643f'
 for k in ['CPU_receipt_SHA256','library_SHA256','source_manifest_SHA256','parent_binding_manifest_SHA256']:assert re.fullmatch('[0-9a-f]{64}',pin[k])
 return True

def derive(pin):
 validate_pin(pin,require_ready=True)
 root=Path(pin['parent_binding_root']);assert sha(root/'SOURCE_MANIFEST.json')==pin['parent_binding_manifest_SHA256']
 manifest=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in manifest['files_SHA256'].items():assert sha(root/name)==h,name
 spec=importlib.util.spec_from_file_location('qualified1378_native_binding',root/'binding_proof.py')
 previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
 old_pin=json.loads((root/'BINDING_PIN.json').read_bytes());receipt=previous.derive(old_pin)
 assert receipt['library_SHA256']==pin['parent_library_SHA256']
 assert sha(pin['parent_CPU_receipt'])==pin['parent_CPU_receipt_SHA256']==old_pin['CPU_receipt_SHA256']
 old=json.loads(Path(pin['parent_CPU_receipt']).read_bytes())
 assert sha(pin['CPU_receipt'])==pin['CPU_receipt_SHA256']
 q=json.loads(Path(pin['CPU_receipt']).read_bytes())
 assert q['completed']and q['passed']and q['CPU_only']and q['actual_CPU']==pin['actual_CPU']
 assert q['source_manifest_SHA256']==pin['source_manifest_SHA256']
 assert q['library']==pin['library']and q['library_SHA256']==pin['library_SHA256']and sha(q['library'])==pin['library_SHA256']
 assert not any(q[k]for k in ['GPU','Native_dlopen','Runtime_created','SDK_executed','weights_payload_opened','actual_Model_forward_executed'])
 assert q['Native_object_count']==49 and q['reused_objects']==48 and q['production_HOST_compiles']==1 and q['HIP_compiles']==0
 assert q['all_old_strong_retained']and q['all_old_C_exports_retained']and q['DT_NEEDED_unchanged']and q['default_unarmed_OFF']
 assert not q['changed_recording_executed_this_job']and not q['original911_suites_replayed']
 before=old['all_objects_SHA256'];after=q['all_objects_SHA256']
 removed='/refactor/candidates/deep-c4-projection-model-r5920/objects/unit-10.o'
 assert len(before)==len(after)==49 and set(before)-set(after)=={removed}
 assert len({p:h for p,h in before.items()if after.get(p)==h})==48
 assert set(after)-set(before)=={q['compiled'][0]['object']}
 assert set(CAPTURE_NAMES)==set(q['new_CABI'])
 for p,h in after.items():assert sha(p)==h,p
 for c in q['compiled']:
  assert sha(c['object'])==c['object_SHA256']and sha(c['MD'])==c['MD_SHA256']and c['fresh_M_equals_MD']
  for p,h in c['dependencies_SHA256'].items():assert sha(p)==h,p
 receipt=copy.deepcopy(receipt);receipt.update(library=pin['library'],library_SHA256=pin['library_SHA256'],
  boundary_derivation_receipt_SHA256=pin['CPU_receipt_SHA256'],actual_Model_unexecuted_CPU_only=True)
 return receipt

def verify_diagnostic_functions(lib,pin):
 """After real Native bind: each additive CABI must resolve to this exact DSO."""
 import ctypes as C
 class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int
 expected=Path(pin['library']).resolve();actual={}
 for name in MODE_NAMES+CAPTURE_NAMES:
  f=getattr(lib,name);assert f.argtypes is not None
  info=Info();assert dl(C.cast(f,C.c_void_p),C.byref(info))
  p=Path(info.fname.decode()).resolve();assert p==expected and sha(p)==pin['library_SHA256'];actual[name]=str(p)
 meta=json.loads(lib.deep_model_actquant_metadata_r438())
 assert meta['owners']==8 and meta['counter_width']==12 and meta['default_mode']==0 and not meta['default_armed']
 assert meta['HostReturned_is_not_GPU_completed']and meta['whole_engine_quiescence_required_for_set_clear']
 return dict(providers=actual,mode_metadata=meta,boundary_new_CABI=7)
