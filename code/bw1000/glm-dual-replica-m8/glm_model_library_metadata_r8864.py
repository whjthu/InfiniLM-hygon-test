"""Read the one manifest-bound Model receipt and ELF bytes without loading it."""
import pathlib,json,hashlib
def read_model_library_metadata(root,manifest):
 root=pathlib.Path(root)
 def mapped(value):
  p=pathlib.Path(value)
  return root/p.relative_to('/refactor') if str(p).startswith('/refactor/') else p
 receipt_path=mapped(manifest['cpu_receipt_path'])
 raw=receipt_path.read_bytes();receipt=json.loads(raw)
 assert receipt['completed']and receipt['CPU_only']and not receipt['GPU']and not receipt['dlopen']
 assert receipt['library_SHA256']==manifest['model_SHA256'],'CPU_receipt_manifest_ModelSHA_mismatch'
 assert receipt['resident_exact_budget']['resident_bank_bytes']==manifest['resident_physical_bytes_per_layer']==manifest['resident_logical_bytes_per_layer']
 assert receipt['resident_exact_budget']['reserve_bytes']==manifest['reserve_bytes']and receipt['resident_exact_budget']['allowance_bytes']==manifest['allowance_bytes']
 library=mapped(receipt['library']);actual=hashlib.sha256(library.read_bytes()).hexdigest()
 assert actual==receipt['library_SHA256'],'actual_Model_ELF_SHA_mismatch'
 return {'cpu_receipt_actual_path':str(receipt_path),'cpu_receipt_SHA256':hashlib.sha256(raw).hexdigest(),'library_actual_path':str(library),'library_declared_path':receipt['library'],'library_SHA256':actual,'GPU':False,'dlopen':False},receipt
