"""Per-shard operational completion and honest quality rows; not all-correct gate."""
from pathlib import Path
import hashlib,json,time,sys

def collect(root,run,actual,source_sha,client_result,deadline,shard,planpath):
 root=Path(root);folder=root/'results/Deep-broad-fewshot-fullpre-quality-r450'/run
 def read(path):
  while time.time()<deadline:
   try:return json.loads(Path(path).read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Bounded terminal artifact visibility:'+str(path))
 client=read(client_result);assert client['completed']and client['execution_passed']and client['attempted_HTTP']==client['completed_HTTP']==64 and len(client['rows'])==64
 front,ready,closed,owner=[read(folder/name)for name in ['FRONTEND.json','MODEL_READY.json','closed-engine.json','owner.json']];identity=ready['identity']
 assert front['identity']==closed['identity']==owner['identity']==identity and identity['actual_step']==actual and identity['source_manifest_SHA256']==source_sha
 assert owner['completed']and owner['normal_Native_shutdown']and owner['engine_worker_shutdown_completed']and owner['observed_all8_threads_exited']and owner['loaded_key_count']==67569
 assert owner['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and owner['HC_mode']==0 and closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert len(aff)==8 and{v['device']for v in aff}==set(range(8));masks=[int(x)for v in aff for x in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32 and all(v['changes']==1 for v in aff)
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in owner['owner_thread_observation']['owners']})==8
 plan=json.loads(Path(planpath).read_bytes())['assigned'];assert len(owner['requests'])==len(plan)==64
 sys.path.insert(0,str(Path(__file__).resolve().parent.parent));from coverage import validate
 calls=0
 for entry,row,c in zip(plan,owner['requests'],client['rows']):
  proof=validate(row,entry);assert c['logical_index']==entry['logical_index']and c['case_id']==entry['case_id']and c['mode']==entry['mode']and c['actual_owner_coverage']==proof and c['prompt_IDs_SHA256']==entry['prompt_IDs_SHA256'];calls+=row['actual_forward_completed']
 assert owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==calls
 refs=[]
 for name in ['FRONTEND.json','MODEL_READY.json','owner.json','closed-engine.json','STOP.json']:
  p=folder/name;read(p);refs.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
 return dict(reports=refs,shard=shard['index'],actual_owner_count=8,Native_engines=1,external_requests=64,Native_CAPI_calls=calls,HC0_checked_forwards=calls,expected_node_PIDs={shard['node']:[front['PID']]},normal8internalowner_destroy=True,lifecycle_scope_pass=True,quality_data_complete=True,correct=sum(r['correct']for r in client['rows']),wrong_or_invalid=sum(not r['correct']for r in client['rows']),caps=sum(r['cap_hit']for r in client['rows']),parse_failures=sum(r['parse_failure']for r in client['rows']),HTTP_non200=sum(r['HTTP_status']!=200 for r in client['rows']),Model_promotion=False,online_SLO=False,warmups=0,all_planned_global_rows_must_be_aggregated_not_averaged=True)
