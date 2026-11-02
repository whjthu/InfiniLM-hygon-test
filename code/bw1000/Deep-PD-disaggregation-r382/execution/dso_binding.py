"""Derive the private state-codec DSO from the qualified49-object parent, no dlopen."""
from pathlib import Path
import json,hashlib,copy

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive(pin,parent_tuple,base):
 old_b,old_q,old_n=parent_tuple
 assert old_b['library_SHA256']==pin['parent_Native_SHA256']=='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
 path=Path(pin['CPU_receipt']);assert sha(path)==pin['CPU_receipt_SHA256'];build=json.loads(path.read_bytes())
 assert build['completed']and build['passed']and build['CPU_only']and not build['GPU_initialized']and not build['Native_dlopen']and not build['Model_loaded']and build['GPU_TUs_compiled']==0
 assert build['all_original_dynamic_exports_retained']and build['CPU_archive_test']['passed']and build['HOST_Native_TUs_compiled']==0 and build['Native_link_executed']is False
 assert build['library_SHA256']==pin['Native_SHA256']and sha(build['library'])==pin['Native_SHA256']
 assert sha(pin['Native_source_manifest'])==pin['Native_source_manifest_SHA256']and sha(pin['ABI_header'])==pin['ABI_header_SHA256']
 sm=json.loads(Path(pin['Native_source_manifest']).read_bytes())
 for name,h in sm['files_SHA256'].items():assert sha(Path(pin['Native_source_manifest']).parent/name)==h
 # v3 reuses actual765/766 objects and compiles one fmt-retainer. Its real
 # receipt intentionally has no fictitious `compiled` list. The separate
 # read-only graph is assembled from those original object/MD/source records.
 graph_path=Path(pin['object_graph_receipt']);assert sha(graph_path)==pin['object_graph_receipt_SHA256'];graph=json.loads(graph_path.read_bytes())
 assert graph['completed']and graph['passed']and graph['CPU_only']and not graph['Native_dlopen']and graph['GPU']==0
 assert build['object_graph']==str(graph_path)and build['object_graph_SHA256']==pin['object_graph_receipt_SHA256']and graph['link_argv']==build['link_argv']
 for ref in graph['native_source_manifests']:
  assert sha(ref['path'])==ref['SHA256'];source=json.loads(Path(ref['path']).read_bytes())
  assert all(sha(Path(ref['path']).parent/n)==h for n,h in source['files_SHA256'].items())
 previous=old_n['link']['full']['objects'];assert len(previous)==49 and graph['parent_objects_SHA256']==previous
 assert graph['all_objects_SHA256']==build['all_objects_SHA256']and len(graph['all_objects_SHA256'])==52
 for ref in graph['build_receipts']:
  assert sha(ref['path'])==ref['SHA256'];actual=json.loads(Path(ref['path']).read_bytes());assert actual['passed']==ref['passed']and actual['CPU_only']and not actual['GPU_initialized']and not actual['Native_dlopen']
  # The original765 failed receipt predates actual_step. Do not add that field
  # retroactively; its step is attributed by the immutable reuse graph evidence.
  if 'actual_step'in actual:assert actual['actual_step']==ref['actual_CPU']
 replacements={x['old_object']:x for x in graph['replacements']};assert set(replacements)<=set(previous)
 reconstructed={p:h for p,h in previous.items()if p not in replacements}
 for item in graph['replacements']+graph['added_objects']:
  assert sha(item['object'])==item['object_SHA256']and sha(item['source'])==item['source_SHA256']and sha(item['MD'])==item['MD_SHA256']
  assert item['compile_step']in {r['actual_CPU']for r in graph['build_receipts']}
  for p,h in item['dependencies_SHA256'].items():assert sha(p)==h
  reconstructed[item['object']]=item['object_SHA256']
 objects=build['all_objects_SHA256'];assert reconstructed==objects and all(sha(p)==h for p,h in objects.items())
 assert len(graph['added_objects'])==3
 b=dict(old_b,CPU_receipt=str(path),CPU_receipt_SHA256=pin['CPU_receipt_SHA256'],library_SHA256=pin['Native_SHA256'],PD_state_ABI_header_SHA256=pin['ABI_header_SHA256'])
 q=copy.deepcopy(old_q);q.update(library=build['library'],library_SHA256=pin['Native_SHA256'],parent_Native_SHA256=old_b['library_SHA256'],derivation_receipt_SHA256=pin['CPU_receipt_SHA256'],PD_source_manifest_SHA256=pin['Native_source_manifest_SHA256'],PD_Model_qualification=False)
 n=dict(completed=True,CPU_only=True,libraries={'full':{'path':build['library'],'SHA256':pin['Native_SHA256']}},link={'full':{'objects':objects,'argv':build['link_argv']}},actual_DTK_bindings=q['actual_DTK_bindings'],derived_binding=True)
 m=json.loads(Path(base).with_name('manifest.json').read_bytes());relative=str(path.relative_to('/refactor'));m['cpu_receipt_path']=relative;m['DSO_SHA256']=pin['Native_SHA256'];m['existing_remote_SHA256'][relative]=pin['CPU_receipt_SHA256']
 return b,q,n,m
