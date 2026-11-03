"""Private exact49-object Model derivation of the qualified afc6 binder; no dlopen here."""
import copy,ctypes,hashlib,json
from pathlib import Path
from public_deep_native.native_binding import binding as parent_binding
PIN=json.loads((Path(__file__).resolve().parents[1]/'MODEL_PIN.json').read_bytes())
NATIVE_SHA=PIN['library_SHA256']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding():
 old_binding,old_receipt,old_normal=parent_binding()
 assert old_receipt['library_SHA256']==PIN['parent_Native_SHA256']
 p=Path(PIN['CPU_receipt']);assert sha(p)==PIN['CPU_receipt_SHA256'];cpu=json.loads(p.read_bytes())
 assert cpu['completed']and cpu['passed']and cpu['actual_CPU']==PIN['actual_CPU']
 assert cpu['CPU_only']and not any(cpu[k]for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','SDK_initialized','Model_loaded'])
 assert cpu['source_manifest_SHA256']==PIN['source_manifest_SHA256']
 assert cpu['library']==PIN['library']and cpu['library_SHA256']==NATIVE_SHA and sha(cpu['library'])==NATIVE_SHA
 assert cpu['Native_object_count']==49 and cpu['unchanged_actual_Model_objects']==46 and cpu['production_HOST_compiles']==2 and cpu['GPU_compiles']==0
 assert cpu['all_old_strong_exports_retained']and cpu['all_old_C_exports_retained']and cpu['DT_NEEDED_unchanged']and cpu['default_unarmed_OFF']
 assert cpu['actual_scope_CPU_contract']['passed']and cpu['actual_quiescent_CPU_consumer']['passed']
 old=old_normal['link']['full']['objects'];new=cpu['all_objects_SHA256']
 assert len(old)==48 and len(new)==49
 same={p:h for p,h in old.items()if p in new and new[p]==h};assert len(same)==46
 assert set(old)-set(new)==set(PIN['replaced_original_objects']),'Only exact MoE+Bank may change'
 assert all(old[p]==h for p,h in PIN['replaced_original_objects'].items())
 for path,h in new.items():assert sha(path)==h
 for unit in cpu['compiled']:
  assert sha(unit['object'])==unit['object_SHA256']and sha(unit['MD'])==unit['MD_SHA256']and unit['fresh_M_equals_MD']
  for path,h in unit['dependencies_SHA256'].items():assert sha(path)==h
 for path,v in old_receipt['actual_DTK_bindings'].items():assert sha(path)==v['SHA256']
 derived=copy.deepcopy(old_receipt);derived.update(library=cpu['library'],library_SHA256=NATIVE_SHA,derived_binding=True,derivation_receipt_SHA256=PIN['CPU_receipt_SHA256'],actual_Model_unexecuted_CPU_only=True)
 descriptor=dict(CPU_receipt=PIN['CPU_receipt'],CPU_receipt_SHA256=PIN['CPU_receipt_SHA256'],library_SHA256=NATIVE_SHA,rows=1,capacity=32768,prefill_chunk=1024,original_guard_binding=old_binding,Model49objects=True)
 link=next(c for c in cpu['commands']if '-o'in c and c[c.index('-o')+1]==cpu['library'])
 normalized=dict(completed=True,CPU_only=True,libraries={'full':{'path':cpu['library'],'SHA256':NATIVE_SHA}},link={'full':{'objects':new,'argv':link}},actual_DTK_bindings=derived['actual_DTK_bindings'],original_guard_proof_preserved=True)
 return descriptor,derived,normalized
def validate_mode_functions(lib):
 class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int
 names=['deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']];actual={}
 for name in names:
  info=Info();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info));p=Path(info.fname.decode()).resolve()
  assert p==Path(PIN['library']).resolve()and sha(p)==NATIVE_SHA;actual[name]=str(p)
 metadata=json.loads(lib.deep_model_actquant_metadata_r438())
 assert metadata['owners']==8 and metadata['counter_width']==12 and metadata['default_mode']==0 and not metadata['default_armed']and metadata['HostReturned_is_not_GPU_completed']and metadata['whole_engine_quiescence_required_for_set_clear']
 return {'providers':actual,'metadata':metadata}
