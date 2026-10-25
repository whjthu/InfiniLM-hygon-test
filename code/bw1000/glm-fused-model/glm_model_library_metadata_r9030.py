"""Manifest-bound original8854 parent plus already built615 attention DSO; no dlopen."""
import pathlib,json,hashlib
def read_model_library_metadata(root,manifest):
 root=pathlib.Path(root)
 def mapped(value):
  p=pathlib.Path(value);return root/p.relative_to('/refactor')if str(p).startswith('/refactor/')else p
 parent=mapped(manifest['cpu_receipt_path']);proof=json.loads(parent.read_bytes())
 assert proof['completed']and proof['CPU_only']and not proof['GPU']and not proof['dlopen']
 assert proof['library_SHA256']=='68acfe089f757014e564c0d0ca5c896ac1f47cd98fa3a843df8b2dd5bf8c63c6'
 assert proof['resident_exact_budget']['resident_bank_bytes']==manifest['resident_physical_bytes_per_layer']==manifest['resident_logical_bytes_per_layer']
 assert proof['resident_exact_budget']['reserve_bytes']==manifest['reserve_bytes']and proof['resident_exact_budget']['allowance_bytes']==manifest['allowance_bytes']
 receipt=mapped(manifest['fused_cpu_receipt_path']);raw=receipt.read_bytes();q=json.loads(raw)['build']
 assert q['completed']and not q['GPU']and not q['dlopen']and q['producer']=='910407.257'
 assert q['library_SHA256']==manifest['model_SHA256']=='8ee7abcc8ab5ef029bb8be7813565cf14af03d6020bb0794f109bdf5fc329d52'
 for path,h in {**q['reused_objects_SHA256'],**q['reused_fused_objects_SHA256']}.items():assert hashlib.sha256(mapped(path).read_bytes()).hexdigest()==h,path
 library=mapped(q['library']);assert hashlib.sha256(library.read_bytes()).hexdigest()==q['library_SHA256']
 return {'cpu_receipt_actual_path':str(receipt),'cpu_receipt_SHA256':hashlib.sha256(raw).hexdigest(),'parent_cpu_receipt_SHA256':hashlib.sha256(parent.read_bytes()).hexdigest(),'library_actual_path':str(library),'library_declared_path':q['library'],'library_SHA256':q['library_SHA256'],'GPU':False,'dlopen':False},proof
