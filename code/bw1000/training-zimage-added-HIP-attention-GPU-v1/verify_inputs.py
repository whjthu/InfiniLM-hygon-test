"""Verify pinned framework and dependencies in the recorded controller epoch."""
from pathlib import Path
import hashlib,json,time
W=Path(__file__).resolve().parent.parent
intent=json.loads((W/'intent.json').read_bytes())
epoch=intent['epoch']
reuse=json.loads((W/'src/REUSED_SOURCE.json').read_bytes());prior=Path(reuse['root']);assert hashlib.sha256((prior/'outcome.json').read_bytes()).hexdigest()==reuse['outcome_SHA256']
o=json.loads((prior/'outcome.json').read_bytes());assert o['passed'] and o['accounting'][0][0]==reuse['CPU188_step'] and o['request_id']==reuse['CPU188_request']
qualified=json.loads((W/'src/LIBRARY_BINDING.json').read_bytes());qroot=Path(qualified['rocm_map_original_namespace']);qraw=(qroot/'outcome.json').read_bytes();assert len(qraw)==qualified['rocm_map_original_outcome_pin']['bytes'] and hashlib.sha256(qraw).hexdigest()==qualified['rocm_map_original_outcome_pin']['SHA256'];qo=json.loads(qraw);assert qo['passed'] and qo['accounting'][0][0]==qualified['rocm_map_original_step'] and qo['accounting'][0][2:4]==['COMPLETED','0:0']
qc=qo['CPU-result.json'];assert qc['bytes']==qualified['rocm_map_original_CPU_result_pin']['bytes'] and qc['SHA256']==qualified['rocm_map_original_CPU_result_pin']['SHA256'];qv=qc['value'];assert qv['passed'] and not qv['CUDA_initialized'] and qv['observed_names']==qualified['resolved_names'] and qv['loader_environment_pin']==qualified['loader_environment_pin']
for name,row in qv['observed_libraries'].items():
 p=qualified['resolved_libraries'][row['realpath']];assert p['bytes']==row['bytes'] and p['SHA256']==row['SHA256']
raw=(prior/'source/FRAMEWORK_FILES.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==reuse['framework_manifest_SHA256']
assert raw==(W/'src/FRAMEWORK_FILES.json').read_bytes() and len(json.loads(raw)['files'])==382
from frozen_bootstrap import bind_frozen_framework
reference=bind_frozen_framework(W/'witness',request_id=intent['request_id'],claim_commit=intent['claim_commit'],
    source_manifest_SHA256=intent['source_manifest_SHA256'],framework_root=prior/'framework',
    framework_manifest=raw,framework_manifest_SHA256=reuse['framework_manifest_SHA256'],
    epoch=epoch,deadline=epoch+100)
assert reference['files_verified']==382 and reference['framework_files_copied']==0
dep=Path(reuse['dependency_root']);raw=(dep/'dependency-extracted-files.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==reuse['dependency_manifest_SHA256'];mapping=json.loads(raw);assert len(mapping['files'])==2852
for name,pin in mapping['files'].items():
 assert time.time()<epoch+100
 src=dep/'dependencies/site'/name;assert src.is_file() and not src.is_symlink() and src.stat().st_size==pin['bytes'] and hashlib.sha256(src.read_bytes()).hexdigest()==pin['SHA256']
assert time.time()<epoch+100
