"""Exact terminal metadata audit for two disjoint normal quality replicas."""
from pathlib import Path,PurePosixPath
import hashlib,json,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def host(root,value):
 p=PurePosixPath(value)
 return Path(root).joinpath(*p.relative_to('/refactor').parts)if p.is_relative_to('/refactor')else Path(value)
def terminal(path,deadline):
 while True:
  try:q=json.loads(Path(path).read_bytes())
  except(FileNotFoundError,json.JSONDecodeError):q=None
  if q and q.get('completed'):return q
  if time.time()>=deadline:raise TimeoutError('Original terminal visibility:'+str(path))
  time.sleep(.5)
def collect(root,run,replicas,manifest,epoch):
 R=Path(root);deadline=min(time.time()+90,epoch+14880);pids={n:[]for nodes in [['a02r4n12','a02r4n13'],['a02r4n14','a02r4n15']]for n in nodes};allrefs=[];summaries=[]
 client=json.loads((R/'incoming/GLM-NoV-W4-quality-r418/client-v1/SOURCE_MANIFEST.json').read_bytes())
 encoding=json.loads((R/'candidates/GLM-NoV-W4-quality-encoding-r418-CPU-v2/encoding.json').read_bytes());encoded={c['case_id']:c for c in encoding['cases']}
 for rep in replicas:
  rec=rep['record'];assert rep['gpu'].returncode==rep['client'].returncode==0 and rec['stop_sent'];actual=rec['actual_GPU'];signatures=[]
  ci=json.loads(Path(rec['client_identity']).read_bytes());assert ci['actual_CPU']==rec['actual_client']and ci['node']==rec['nodes'][1]and ci['client_SHA256']==client['files_SHA256']['quality_client.py'];pids[ci['node']].append(ci['PID'])
  result=terminal(rec['client_result'],deadline);assert result['execution_passed']and not result['errors']and result['actual_HTTP']==result['planned_HTTP']==len(result['rows'])==64 and len(result['planned_case_ids'])==32
  assert result['loader_manifest_SHA256']==manifest['loader_manifest_SHA256']and result['profile_manifest_SHA256']==manifest['profile_manifest_SHA256']and result['shard_index']==rec['index']and result['shards']==2
  for stage in range(2):
   node=rec['nodes'][stage]
   for rank in range(8):
    name='GLM-latest-full-owner-r290-'+rec['run']+'-stage%d-rank%d.json'%(stage,rank);path=R/'results'/name;q=terminal(path,deadline)
    assert q['actual_producer']==actual and q['source_manifest_SHA256']==manifest['loader_manifest_SHA256']and q['service_profile_manifest_SHA256']==manifest['profile_manifest_SHA256']and q['library_sha256']==manifest['Native_SHA256']
    assert all(q[k]for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','native_service_loop_completed','normal_shutdown','normal_reset','constructor_noV_verified'])
    assert(q['TP'],q['PP'],q['all_serving_cards'],q['cache_rows'],q['cache_capacity'])==(8,2,16,1,32768)
    binding=q['execution_binding_r418'];assert binding['replica_index']==rec['index']and binding['nodes']==rec['nodes']and binding['original_GPU_start_unix']==epoch
    assert q['context_policy']=='v403-no-resident-V'and q['context_policy_env']=='1'
    assert q['constructor_noV_census']['cache_category_bytes'].get('projected_value',0)==0 and q['memory_after_create'][0]>8*1024**3 and q['minimum_driver_free_bytes']>=512*1024**2
    assert q['actual_partition']['begin']==[0,38][stage]and q['actual_partition']['end']==[38,78][stage]
    assert len(q['service_requests'])==q['completed_Native']==q['attempted_Native']==q['full_requests_completed']==64
    rows=q['service_requests'];assert [r['request_index']for r in rows]==list(range(64));assert all(r['requested_output_budget']==4096 and r['original_reset_all16_completed']and r['reason']in('eos_token','length')for r in rows)
    assert[r['prefill_control_window']for r in rows]==[r['window']for r in result['rows']]
    assert[r['actual_prompt_token_ids_SHA256']for r in rows]==[encoded[r['case_id']]['prompt_IDs_SHA256']for r in result['rows']]
    assert[r['generated_token_ids_SHA256']for r in rows]==[r['generated_IDs_SHA256']for r in result['rows']]
    signatures.append([(r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason'],r['prefill_control_window'])for r in rows])
    life=R/'results'/('GLM-r404-lifecycle-'+rec['run']+'-stage%d-rank%d.json'%(stage,rank));l=terminal(life,deadline);assert l['normal']and not l['primary_error']and not l['cleanup_errors']and l['destroy_attempts']==l['shutdown_attempts']==1 and l['destroy_RC']==l['shutdown_RC']==0 and l['actual_step']==actual and l['source_manifest_SHA256']==manifest['loader_manifest_SHA256']
    snapshot=host(R,q['constructor_snapshot_path']);assert sha(snapshot)==q['constructor_snapshot_SHA256']
    pids[node]+=[q['PID'],q['parent_PID']];allrefs.append(dict(replica=rec['index'],stage=stage,rank=rank,path=str(path),SHA256=sha(path),lifecycle_path=str(life),lifecycle_SHA256=sha(life),constructor_path=str(snapshot),constructor_SHA256=sha(snapshot),normal=True,preweight_free_bytes=q['memory_after_create'][0],minimum_sampled_forward_free_bytes=q['minimum_driver_free_bytes'],snapshots_are_peak=False))
    if(stage,rank)==(1,0):assert q['normal_HTTP_shutdown']and not q['cleanup']['forced_cleanup']and not q['cleanup']['errors']
  assert all(v==signatures[0]for v in signatures)
  logs=sorted((R/'logs').glob(rec['label']+'-'+actual+'-*.log'));assert len(logs)==2 and all('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage)in next(p.read_text(errors='replace')for p in logs if p.name.endswith('-%d.log'%stage))for stage in range(2))
  rec['client_result_SHA256']=sha(rec['client_result']);summaries.append(dict(index=rec['index'],actual_GPU=actual,actual_client=rec['actual_client'],client_result=rec['client_result'],client_result_SHA256=rec['client_result_SHA256'],quality_summary=result['summary'],planned_case_ids=result['planned_case_ids']))
 assert not set(summaries[0]['planned_case_ids'])&set(summaries[1]['planned_case_ids'])
 return dict(normal32owners=True,normal_create_destroy_shutdown_once=32,actual_HTTP=128,actual_Native_logical_requests=128,all16_per_replica_signatures_equal=True,quality_summaries=summaries,reports=allrefs,expected_node_PIDs={n:sorted(set(v))for n,v in pids.items()},caps_or_wrong_answers_not_excluded=True,no_regression_proven=False,full_benchmark=False)
