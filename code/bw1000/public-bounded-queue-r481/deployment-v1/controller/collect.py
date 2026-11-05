"""C1 terminal lifecycle and complete finite-plan consumption, quality separate."""
from pathlib import Path
import hashlib,json,time
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(runtime,expected_identity,client_result,client_identity,deadline,*,private_plan,private_plan_SHA256):
 runtime=Path(runtime)
 def read(p):
  while time.time()<deadline:
   try:return json.loads(Path(p).read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Original terminal90s artifact deadline:'+str(p))
 assert H(private_plan)==private_plan_SHA256;plan=read(private_plan);assert len(plan['requests'])==20 and plan['ordinals']==[0,1,128,129];records={r['case_id']:r for r in plan['records']}
 client=read(client_result);assert client['completed']and client['planned']==20 and client['warm']==0 and len(client['rows'])==20 and{r['index']for r in client['rows']}==set(range(20))
 front=read(runtime/'FRONTEND.json');ready=read(runtime/'MODEL_READY.json');closed=read(runtime/'closed-engine.json');owner=read(runtime/'owner.json');stop=read(runtime/'STOP.json')
 assert all(x['identity']==expected_identity for x in[front,ready,closed,owner,stop])
 assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0 and ready['TP']==8 and ready['PP']==1
 assert owner['completed']and owner['normal_Native_shutdown']and owner['engine_worker_shutdown_completed']and owner['observed_all8_threads_exited']and owner['loaded_key_count']==67569
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 assert owner['fixed_profile']['fixed_mode']==owner['fixed_profile']['r366_fixed']==0 and owner['fixed_profile']['HC_groups']==86
 assert owner['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and owner['cache_rows']==1 and owner['cache_capacity']==32768
 previous=None
 for key in['owner_affinity_before_weight_load','owner_affinity_after_weight_load']:
  rows=owner[key]['actual_owner_report']['owners'];assert len(rows)==8 and{r['device']for r in rows}==set(range(8));allcores=[]
  for r in rows:
   cores=[int(x)for x in r['verified_mask'].split(',')];assert r['changes']==1 and len(set(cores))==4;allcores+=cores
  assert len(set(allcores))==32;signature=[r['verified_mask']for r in rows];assert previous is None or signature==previous;previous=signature
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({r['TID']for r in owner['owner_thread_observation']['owners']})==8
 assert all(f>=16*1024**3 for f,total in owner['GPU_memory_after_load'])
 requests=owner['requests'];by={r['request_id']:r for r in requests};assert len(by)==len(requests)
 calls=0
 for r in requests:
  assert r['normal_reset_before']and r['normal_reset_after']and r['r454_actual_T1_coverage']['passed']and not r['r454_poisoned']
  assert r['actual_forward_attempts']==r['actual_forward_completed'];calls+=r['actual_forward_completed'];ids=r['generated_token_ids']
  trace=r['Native_continuation'];prefills=[x for x in trace if x['kind']=='prefill'];decode=sum(x['calls']for x in trace if x['kind']=='decode_summary');t1=sum(x['tokens']==1 for x in prefills)+decode;forwards=len(prefills)+decode;assert forwards==r['actual_forward_completed'];coverage=r['r454_actual_T1_coverage'];assert coverage['completed_forwards']==forwards and coverage['one_token_prefill_forwards']==sum(x['tokens']==1 for x in prefills)and coverage['decode_forwards']==decode and coverage['T1_HC_per_owner']==86*t1
  counts=r['r454_actual_owner_counts'];assert len(counts)==8
  for i,counter in enumerate(counts):
   legacy=counter['legacy'];assert counter['owner']==i and counter['created']==counter['ready']==86 and legacy['pre']==86*forwards and legacy['eligible_bf16_t1']==86*t1 and legacy['other_t']==86*(forwards-t1)and all(legacy[k]==0 for k in ['created','ready','completed','f32_t1'])and all(v==0 for v in counter['fullpre'].values())
  assert len(ids)==r['actual_generated_tokens']<=4096 and all(type(i)is int and 0<=i<129280 for i in ids)
  reason=r['native_finish_reason'];assert reason in['eos_token','length','canceled','timeout']
  if reason=='eos_token':assert ids and ids[-1]==1 and ids.count(1)==1
  elif reason=='length':assert len(ids)==4096 and 1 not in ids
 ledger=owner['queue_ledger'];active={r['request_id']for r in ledger if r['outcome']=='owner_claimed'}
 assert active==set(by)
 skipped=[r for r in ledger if r['outcome']in['queued_cancel','queued_expired','queue_STOP_finished_after_reset']]
 assert all(r['Native_entered']is False and r['request_id']not in by for r in skipped)
 rejected=[r for r in ledger if r['outcome']=='capacity_rejected'];accepted={r['request_id']for r in ledger if r['outcome']=='accepted'};assert not({r['request_id']for r in rejected}&accepted);assert accepted==active|{r['request_id']for r in skipped},'Accepted queue orphan at terminal'
 outcomes=[]
 for row in client['rows']:
  # Client socket completion is neither Native finish nor natural EOS.
  rid=row.get('server_request_id');r=by.get(rid);events=[x for x in ledger if rid is not None and x['request_id']==rid]
  planned=plan['requests'][row['index']];original=records[planned['case_id']]
  if row.get('sent'):assert row['actual_body_SHA256']==planned['actual_body_SHA256']and row['prompt_IDs_SHA256']==planned['prompt_IDs_SHA256']
  if r is not None:
   admission=r['admission'];budget=admission['raw_budget_contract'];assert admission['Native_prompt_token_ids_SHA256']==row['prompt_IDs_SHA256']==original['prompt_IDs_SHA256'];assert r['prompt_tokens']==len(original['prompt_token_ids'])and r['max_new_tokens']==4096;assert budget['selected']==4096 and not budget['coercion']and not budget['clamp'];assert admission['chat_template_kwargs']['enable_thinking']is False
  outcomes.append(dict(index=row['index'],server_request_id=rid,linkage='exact_SSE_data_id'if rid is not None else'UNKNOWN_NO_SERVER_ID',planned_cancel=row.get('planned_cancel'),actual_Native_entered=r is not None,Native_forward_completed=r['actual_forward_completed']if r else 0 if events and any(x.get('Native_entered')is False for x in events)else None,Native_finish_reason=r['native_finish_reason']if r else None,Native_reset_after=r['normal_reset_after']if r else None,queue_outcomes=[x['outcome']for x in events],active_cancel_coverage_proven=bool(row['index']==4 and r and r['native_finish_reason']=='canceled'and r['actual_forward_completed']>0 and r['normal_reset_after']),clock_coverage_complete=row.get('clock',{}).get('clock_coverage_complete'),owned_finished_notice=row.get('clock',{}).get('owned_finished'),score=row['score']))
 refs=[]
 for name in['FRONTEND.json','MODEL_READY.json','owner.json','closed-engine.json','STOP.json']:
  p=runtime/name;refs.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=H(p)))
 starts=[r['request_start_ns']for r in client['rows']if 'request_start_ns'in r];ends=[r['ended_ns']for r in client['rows']if 'ended_ns'in r];wall=(max(ends)-min(starts))/1e9 if starts and ends else None;visible=sum(r.get('clock',{}).get('visible_nonEOS',0)for r in client['rows']);correct_visible=sum(r.get('clock',{}).get('visible_nonEOS',0)for r in client['rows']if r['score']['correct']);client['all20_goodput']=dict(client_scope_wall_s=wall,visible_ID_receipts=visible,correct_visible_ID_receipts=correct_visible,visible_per_s_per_all8cards=visible/wall/8 if wall and wall>0 else None,correct_visible_per_s_per_all8cards=correct_visible/wall/8 if wall and wall>0 else None,full_planned_denominator=20,not_completer_subset=True,excludes_load=True,token_ID_receipt_not_semantic_output=True)
 ci=read(client_identity)
 return dict(reports=refs,actual_owner_count=8,Native_engines=1,external_requests_planned=20,sent=client['sent'],unsent=client.get('unsent',client['missing']),incomplete=client.get('incomplete'),Native_requests=len(requests),Native_forward_CAPI_calls=calls,queue_ledger_entries=len(ledger),queued_skipped=len(skipped),capacity_rejected=len(rejected),outcomes=outcomes,quality_summary=client['quality_summary'],all20_goodput=client['all20_goodput'],quality_passed=None,execution_passed=True,normal8internalowner_destroy=True,expected_PIDs=[front['PID'],ci['PID']],active_cancel_actual_coverage=any(r['active_cancel_coverage_proven']for r in outcomes),active_cancel_limit='Client planned first-marker close may happen after EOS; only actual owner canceled+forward+reset proves coverage, no automatic rerun.',Model_promotion=False,client_incremental_SLO_qualification='Report rawr480 domains/coverage first; clientreceipt tokenmarker notsemanticTTFT or exclusiveGPUbusy.')
