"""Original successful CPU902 Native/link and CPU904 actual-entry evidence; no reruns."""
from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(deploy,package,*,local_files=None):
 d=Path(deploy);card=json.loads((d/'CPU_EVIDENCE.json').read_bytes());out={}
 for item in card['references']:
  p=Path(item['path']) if local_files is None else Path(local_files)/item['local_untracked_file']
  assert sha(p)==item['SHA256'];q=json.loads(p.read_bytes());assert q['completed'] and q['passed'] and q['actual_CPU']==item['actual_CPU'];out[item['kind']]=q
 n=out['Native902'];e=out['entry904']
 assert n['library_SHA256']==card['Native_SHA256'] and n['Native_object_count']==49 and n['unchanged_actual_Model_objects']==46 and n['production_HOST_compiles']==2 and n['GPU_compiles']==0
 assert n['all_old_strong_exports_retained'] and n['all_old_C_exports_retained'] and n['DT_NEEDED_unchanged'] and n['default_unarmed_OFF']
 assert n['actual_scope_CPU_contract']['passed'] and n['actual_quiescent_CPU_consumer']['passed']
 assert not any(n[k]for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','SDK_initialized','Model_loaded'])
 assert e['source_manifest_SHA256']==card['provider_source_SHA256'] and e['worker_control']['passed'] and e['worker_control']['recording_CABI_only'] and not e['worker_control']['actual_Model_or_GPU_or_HTTP']
 assert e['actual_binding']==dict(library_SHA256=card['Native_SHA256'],objects=49,guard_retained=True,Native_dlopen=False)
 assert e['actual_whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 assert not any(e[k]for k in ['Native_dlopen','Runtime_execution','Model_loaded','GPU'])
 return dict(Native_actual_CPU='910407.902 COMPLETED0',entry_actual_CPU='910407.904 COMPLETED0',two_distinct_scopes_consumed=True,Native_or_Model_or_device_qualification=False)
