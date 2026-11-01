"""Exact307 ancestry and new whole-glue CPU; no enlargement of621/628 scope."""
import pathlib,json,hashlib
from parent_candidate_binding_r308 import verify_candidate as verify_parent,CPU628_SHA
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CHANGED={'owner_r290.py','model_entry_r290.py','deadline_r290.py','service_budget_r204.py','GPU_inside_r290.sh','client_driver_r290.py','PREPARATION.json'}
ADDED={'FIXTURE_CARD.json','workload_r349.py','client_crossworkload_r349.py'}
PARENT_SHA='3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
def verify_candidate(root,directory,source):
 root=pathlib.Path(root);source=pathlib.Path(source);parent=root/'incoming/GLM-prefill-W2-W4-owner-r307'
 verify_parent(root,directory,parent)
 old=json.loads((parent/'SOURCE_MANIFEST.json').read_bytes());new=json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
 assert new['parent_manifest_SHA256']==PARENT_SHA and set(new['files_SHA256'])==set(old['files_SHA256'])|ADDED
 assert {n for n in old['files_SHA256'] if old['files_SHA256'][n]!=new['files_SHA256'][n]}==CHANGED
 for n,h in old['files_SHA256'].items():
  if n not in CHANGED:assert sha(source/n)==h
 oldentry=(parent/'model_entry_r290.py').read_text();newentry=(source/'model_entry_r290.py').read_text()
 oldpair="'request_file':'/refactor/results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json','request_SHA256':'5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'"
 assert oldentry.replace(oldpair,"'fixture_card':str(pathlib.Path(__file__).parent/'FIXTURE_CARD.json')")==newentry
 return {'unchanged_Native_math_loader_identity_cleanup':True,'CPU621_628_original_scope_only':True}
def verify_whole_CPU(root,source,go):
 path=pathlib.Path(go['whole_CPU_receipt_path']);assert path.is_relative_to(pathlib.Path(root)/'results')
 assert sha(path)==go['whole_CPU_receipt_SHA256'];q=json.loads(path.read_bytes())
 assert q['completed']and q['passed']and q['CPU_only']and q['source_manifest_SHA256']==sha(pathlib.Path(source)/'SOURCE_MANIFEST.json')
 assert q['GPU']==q['Model']==q['Native_Context']==0 and not q['real_TCP_listener'] and not q['actual_HTTP'] and not q['real_GPU_GO']
 assert {c['name'] for c in q['cases']}=={'actual_emitter_owner_to_actual_supervisor_and_client_main','all32_actual_owner_loops_and_actual_worker_cleanup_suffix','original_start_deadline_not_reset','cancellation_before_forward_no_retry'}
 assert all(c['passed'] for c in q['cases']) and len(q['cases'][1]['owners'])==32
def require_baseline_terminal(root,go):
 p=pathlib.Path(root)/'results/GLM-prefill-W2-W4-Model-r308-watch.json';w=json.loads(p.read_bytes())
 assert w['actual_GPU']=='910376.823' and w['completed'] and w['post_empty'] and w['terminal'][0]=='910376.823' and w['terminal'][2:4]==['COMPLETED','0:0']
 assert go['baseline_result_consumed'] and go['baseline_result_record'].endswith('GLM308823-root-audited-W4-scoped-Model-gain-PASS-clean.json')
