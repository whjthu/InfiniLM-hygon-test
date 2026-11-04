"""Two distinct CPU scopes: pinned750 Native objects,946 changed public glue."""
from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(deploy,package,*,local_files=None):
 card=json.loads((Path(deploy)/'CPU_EVIDENCE.json').read_bytes());out={}
 for item in card['references']:
  p=Path(item['path'])if local_files is None else Path(local_files)/item['local_untracked_file']
  assert sha(p)==item['SHA256'];q=json.loads(p.read_bytes());assert q['completed']and q['passed'];out[item['kind']]=q
 n=out['Native750'];e=out['entry946']
 assert n['actual_step']=='910407.750'and n['library_SHA256']==card['Native_SHA256']and len(n['all50_objects_SHA256'])==50
 assert n['old_header_identical']and n['old_CABI_and_exports_retained']and n['control_contract_passed']and n['MD_equals_fresh_M']and n['new_HIP_reused']and not n['added_NEEDED']and not n['removed_NEEDED']
 assert not any(n[k]for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','Model_loaded'])
 assert e['actual_CPU']=='910407.946'and e['source_manifest_SHA256']==card['provider_source_SHA256']and e['worker_control']['passed']and e['worker_control']['actual_worker_and_client']and not e['worker_control']['actual_Native_or_Model']
 assert len(e['worker_control']['cases'])==6 and all(c['passed']for c in e['worker_control']['cases'])
 a=e['actual_binding'];assert a['library_SHA256']==card['Native_SHA256']and a['objects']==50 and a['guard_retained']and not a['Native_dlopen']and not a['project_maps']and set(a['hidden'].values())=={'-1'}
 assert e['actual_whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 assert not any(e[k]for k in ['Native_dlopen','Runtime_execution','Model_loaded','GPU','new_HOST_TUs','new_GPU_TUs'])
 return dict(Native_actual_CPU='910407.750 COMPLETED0',entry_actual_CPU='910407.946 COMPLETED0',two_distinct_scopes_consumed=True,Native_or_Model_or_device_qualification=False)
