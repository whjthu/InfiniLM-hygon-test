"""Terminal original-file audit; no runtime calls or retry/resubmission."""
from pathlib import Path
import hashlib,json,math,time
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def terminal(path,deadline,rejection=None):
 while True:
  if rejection is not None:
   try:refused=json.loads(Path(rejection).read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):refused=None
   if refused and refused.get('capacity_rejected_before_disk_loading'):
    raise RuntimeError(dict(capacity_rejected_before_disk_loading=True,path=str(rejection),SHA256=sha(rejection),cleanup_normal_is_not_capacity_PASS=True))
  try:q=json.loads(Path(path).read_bytes())
  except(FileNotFoundError,json.JSONDecodeError):q=None
  if isinstance(q,dict)and q.get('completed')is True:return q
  if time.time()>=deadline:raise TimeoutError('Original terminal visibility deadline:'+str(path))
  time.sleep(.5)
def collect(root,run,actual,owner_sha,deadline):
 R=Path(root);owners=[];signatures=[];pids={};emitter=None
 for stage in range(2):
  node=['a02r4n12','a02r4n13'][stage];pids[node]=[]
  for rank in range(8):
   p=R/'results'/('GLM-latest-full-owner-r290-'+run+'-stage%d-rank%d.json'%(stage,rank));refusal=R/'results'/('glm-large-preflight-r5394-'+run+'-stage%d-rank%d.json'%(stage,rank));q=terminal(p,deadline,refusal)
   assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['normal_reset_close']and q['native_service_loop_completed']and q['normal_shutdown']
   assert q['actual_producer']==actual and q['source_manifest_SHA256']==owner_sha and q['library_sha256']==NATIVE
   assert q['constructor_noV_verified']and q['context_policy']=='v403-no-resident-V'and q['context_policy_env']=='1'
   assert q['constructor_noV_census']['cache_category_bytes'].get('projected_value',0)==0
   assert q['constructor_noV_census']['rows']==1 and q['constructor_noV_census']['capacity']==32768
   assert q['actual_r403_Native_binding']['library_SHA256']==NATIVE and len(q['actual_r403_Native_binding']['CAPI_providers'])==8
   assert q['TP']==8 and q['PP']==2 and q['all_serving_cards']==16
   assert q['actual_partition']['begin']==[0,38][stage]and q['actual_partition']['end']==[38,78][stage]
   rows=q['service_requests'];assert len(rows)==q['full_requests_completed']==q['planned_Native']==q['attempted_Native']==q['completed_Native']==6
   assert [r['fixture_id']for r in rows]==['long-write']*6 and[r['prompt_tokens']for r in rows]==[9741]*6
   assert all(r['original_reset_all16_completed']and r['generated_natural_EOS']and r['reason']=='eos_token'and r['requested_output_budget']==4096 for r in rows)
   assert [r['pair_index']for r in rows]==list(range(6));assert[r['prefill_control_window']for r in rows]==[2,4,2,4,4,2];assert[r['paired_warmup']for r in rows]==[True,True,False,False,False,False]
   sig=[(r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason'])for r in rows];signatures.append(sig)
   life=R/'results'/('GLM-r404-lifecycle-'+run+'-stage%d-rank%d.json'%(stage,rank));l=terminal(life,deadline)
   assert l['normal']and not l['primary_error']and not l['cleanup_errors']and l['destroy_attempts']==l['shutdown_attempts']==1 and l['destroy_RC']==l['shutdown_RC']==0
   assert l['actual_step']==actual and l['source_manifest_SHA256']==owner_sha
   snapshot=Path(str(q['constructor_snapshot_path']).replace('/refactor',str(R),1));assert sha(snapshot)==q['constructor_snapshot_SHA256'];s=json.loads(snapshot.read_bytes());assert s['actual_step']==actual and s['Native_SHA256']==NATIVE and not s['loaded']
   pids[node]+=[q['PID'],q['parent_PID']];owners.append(dict(stage=stage,rank=rank,path=str(p),SHA256=sha(p),lifecycle_path=str(life),lifecycle_SHA256=sha(life),constructor_snapshot_path=str(snapshot),constructor_snapshot_SHA256=sha(snapshot),normal=True))
   if(stage,rank)==(1,0):emitter=q;assert q['normal_HTTP_shutdown']and not q['cleanup']['forced_cleanup']and not q['cleanup']['errors']
 assert all(x==signatures[0]for x in signatures),'16owner same-residency token/EOS/reset signatures differ'
 cp=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+run+'.json');c=terminal(cp,deadline)
 assert c['passed']and c['actual_GPU_handle']==actual and c['planned_HTTP']==c['planned_Native']==c['attempted_HTTP']==c['completed_HTTP']==c['actual_HTTP']==c['actual_Native']==6
 for phase in c['phases'].values():
  cases=phase['result']['cases'];assert len(cases)==6 and all(r['natural_EOS']and r['semantic_gold_passed']and r['passed']for r in cases)
 return dict(normal16owners=True,all16_signatures_equal=True,actual_HTTP=6,actual_Native=6,semantic_gold_and_EOS=True,constructor_noV_all16=True,one_context_perowner=True,reports=owners,CPU_supervisor_path=str(cp),CPU_supervisor_SHA256=sha(cp),expected_node_PIDs={n:sorted(set(v))for n,v in pids.items()},emitter_report_path=next(x['path']for x in owners if x['stage']==1 and x['rank']==0),fullbenchmark_qualified=False,paired_speedup_qualified=False)
