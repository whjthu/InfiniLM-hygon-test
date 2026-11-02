"""Original038d ancestry plus actual804 one-caller fullNative replacement."""
from pathlib import Path
import hashlib,json
from parent_native_chain_r186 import verify as original
R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NATIVE_SHA='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
CPU=R/'candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json'
CPU_SHA='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32'
def verify():
 base=original();assert base['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 assert sha(CPU)==CPU_SHA;q=json.loads(CPU.read_bytes())
 assert q['completed']and q['CPU_only']and q['GPU']==0 and not q['Native_dlopen']and not q['checkpoint_IO']and q['producer']=='910407.804'
 assert q['parent_Native_SHA256']==base['library_SHA256']and q['parent_receipt_SHA256']==base['cpu_receipt_SHA256']
 assert q['library_SHA256']==NATIVE_SHA and sha(q['library'])==NATIVE_SHA
 assert q['all_parent_CABI_retained']and q['all_parent_strong_exports_retained']and q['ldd_r_parent_equal']and q['class_header_layout_unchanged']and q['default_OFF']
 assert q['old_object_count']==q['final_object_count']==29 and len(q['reused_objects_SHA256'])==28 and len(q['compiled'])==1
 for f,h in q['final_objects_SHA256'].items():assert sha(f)==h,f
 for c in q['compiled']:
  assert sha(c['source'])==c['source_SHA256']and sha(c['MD_path'])==c['MD_SHA256']
  for f,h in c['dependencies_SHA256'].items():assert sha(f)==h,f
 for f,h in q['actual_dynamic_dependency_SHA256'].items():assert sha(f)==h,f
 base.update(cpu_receipt_actual_path=str(CPU),cpu_receipt_SHA256=CPU_SHA,library_actual_path=q['library'],library_declared_path=q['library'],library_SHA256=NATIVE_SHA,
             qualified_original_parent_Native_SHA256=q['parent_Native_SHA256'],context_policy_env='GLM_V_ON_DEMAND_R403',context_policy='no-resident-V',
             HOST_replacement_count=1,constructor_GPU_qualified=False,Model_GPU_qualified=False,private_r403_source_manifest_SHA256=q['source_manifest_SHA256'])
 return base
