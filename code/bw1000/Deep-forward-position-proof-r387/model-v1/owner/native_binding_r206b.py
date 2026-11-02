"""Current976f graph with only decoder/compressor proof HOST units replaced."""
import pathlib,json,hashlib,copy
from native_binding_976f_r387 import BASE,binding as parent_binding
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
PIN=json.loads((D/'NATIVE_BUILD_PIN.json').read_bytes());NATIVE_SHA=PIN['library_SHA256']
def binding():
 old,oldreceipt,oldnormal=parent_binding();assert old['library_SHA256']==PIN['parent_Native_SHA256']
 p=pathlib.Path(PIN['CPU_receipt']);assert sha(p)==PIN['CPU_receipt_SHA256'];q=json.loads(p.read_bytes())
 assert q['completed']and q['passed']and q['CPU_only']and not q['GPU_initialized']and not q['Native_dlopen']and not q['Runtime_created']and not q['weights_payload_opened']and not q['Model_loaded']
 assert q['library_SHA256']==NATIVE_SHA and sha(q['library'])==NATIVE_SHA
 assert len(q['compiled'])==2 and all(u['MD_equals_fresh_M']for u in q['compiled'])
 previous=oldnormal['link']['full']['objects'];replaced={u['replaces']for u in q['compiled']}
 assert len(previous)==49 and len(replaced)==2
 assert {n:h for n,h in previous.items()if n not in replaced}==q['unchanged47_objects_SHA256']
 objects=q['all49_objects_SHA256'];assert len(objects)==49 and all(sha(n)==h for n,h in objects.items())
 for u in q['compiled']:
  assert sha(u['source'])==u['source_SHA256']and sha(u['original_source'])==u['original_source_SHA256']
  assert all(sha(n)==h for n,h in u['dependencies_SHA256'].items())
 assert q['all_old_strong_and_C_exports_retained']and q['added_NEEDED']==q['removed_NEEDED']==[]
 assert q['CPU_contract']['passed']and q['CPU_contract']['checks']==34 and q['default_OFF']
 assert q['actual_DTK_bindings']==oldreceipt['actual_DTK_bindings']
 for n,v in q['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
 b=dict(old,CPU_receipt=str(p),CPU_receipt_SHA256=PIN['CPU_receipt_SHA256'],library_SHA256=NATIVE_SHA,
  positions_modes_supported=[0,1],positions_default_mode=0,positions_copies_per_owner_forward=62,positions_scopes_per_owner_forward=43)
 receipt=copy.deepcopy(oldreceipt);receipt.update(library=q['library'],library_SHA256=NATIVE_SHA,
  derivation_receipt_SHA256=PIN['CPU_receipt_SHA256'],parent_Native_SHA256=old['library_SHA256'])
 normal=dict(completed=True,CPU_only=True,libraries={'full':{'path':q['library'],'SHA256':NATIVE_SHA}},
  link={'full':{'objects':objects,'argv':q['link_argv']}},actual_DTK_bindings=q['actual_DTK_bindings'],
  derived_binding=True,original_guard_proof_preserved=True)
 return b,receipt,normal
def prebuild():return binding()[2]
def manifest():
 b,q,_=binding();m=json.loads(BASE.with_name('manifest.json').read_bytes());rel=str(pathlib.Path(b['CPU_receipt']).relative_to(R))
 m['cpu_receipt_path']=rel;m['DSO_SHA256']=NATIVE_SHA;m['existing_remote_SHA256'][rel]=b['CPU_receipt_SHA256'];return m
