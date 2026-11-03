"""One public process/engine, eight actual internal Native owners; passive reports."""
from pathlib import Path
import hashlib,json,time
from model_counts import validate as validate_counts
def collect(root,run,actual,source_sha,client_result,deadline):
 root=Path(root);folder=root/'results/Deep-HC-fullpre-public-r449'/run
 def read(path):
  while time.time()<deadline:
   try:return json.loads(path.read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Terminal artifact visibility:'+str(path))
 client=read(Path(client_result));assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==6
 front=read(folder/'FRONTEND.json');ready=read(folder/'MODEL_READY.json');closed=read(folder/'closed-engine.json')
 assert front['identity']==ready['identity']==closed['identity']and ready['identity']['actual_step']==actual and ready['identity']['source_manifest_SHA256']==source_sha
 q=read(folder/'owner.json');assert q['identity']==ready['identity']and q['completed']and q['normal_Native_shutdown']and q['engine_worker_shutdown_completed']and q['observed_all8_threads_exited']
 assert q['loaded_key_count']==67569 and q['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and q['HC_mode']==0
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 owners=q['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert len(owners)==8 and{v['device']for v in owners}==set(range(8))
 masks=[int(x)for v in owners for x in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32 and all(v['changes']==1 for v in owners)
 observed=q['owner_thread_observation'];assert observed['status']=='OBSERVED'and len({v['TID']for v in observed['owners']})==8
 rows=q['requests'];assert len(rows)==6;calls=0
 for i,row in enumerate(rows):
  c=client['rows'][i];validate_counts(row,i);assert c['mode']==row['r449_mode'] and c['r449_index']==i and c['r449_T1_coverage']==row['r449_actual_T1_coverage'];assert row['request_id']==c['request_id']and row['normal_reset_before']and row['normal_reset_after']
  assert row['actual_forward_attempts']==row['actual_forward_completed']and not row.get('error')
  assert row['admission']['text_delivery']==c['delivery']and row['final_metadata']['buffered_P0_text']==(c['delivery']=='P0')
  assert row['native_finish_reason']=='eos_token'and row['generated_token_ids'][-1]==1
  assert row['actual_generated_tokens']==len(row['generated_token_ids'])==c['usage']['completion_tokens']
  h=row['admission']['history_render_audit'];assert h['Native_prompt_token_ids_SHA256']==c['prompt_IDs_SHA256']and h['output_budget']==c['budget']
  assert bool(h['pending_thinking_prefix_open'])==c['thinking']and h['qualified']
  calls+=row['actual_forward_completed']
 assert q['HC_dispatch_checked_forwards']==q['layout_dispatch_checked_forwards']==calls
 refs=[]
 for name in ['FRONTEND.json','MODEL_READY.json','owner.json','closed-engine.json','STOP.json']:
  p=folder/name;read(p);refs.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
 return dict(reports=refs,actual_owner_count=8,Native_engines=1,external_requests=6,Native_CAPI_calls=calls,HC0_checked_forwards=calls,
  expected_node_PIDs={'a02r4n04':[front['PID']]},semantic_gold_and_natural_EOS=True,normal8internalowner_destroy=True,
  public_API_scope_pass=True,warmups=2,formal=4,transport_latency_comparison=False,performance_comparison=True,Model_promotion=False,online_SLO=False,incremental_client_latency_qualified=False)
