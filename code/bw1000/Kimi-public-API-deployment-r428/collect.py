"""Read original four-leader public reports; no device imports or jobs."""
from pathlib import Path
import hashlib
import json
import time

NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']


def collect(root, run, actual, source_sha, client_result, deadline):
 root=Path(root);folder=root/'results/public-Kimi-provider-r428';refs=[];signatures=[];pids={}
 def read(path):
  while time.time()<deadline:
   try:return json.loads(path.read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Original terminal artifact visibility:'+str(path))
 client=read(Path(client_result));assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==2
 model_sha=None;CAPI_calls=[]
 for node in range(4):
  path=folder/('Kimi-public-r428-'+run+'-node%d.json'%node);q=read(path)
  claim=read(folder/('claim-'+run+'-node%d.json'%node));assert claim['actual_step']==actual and claim['node']==node and claim['run']==run
  assert claim['source_manifest_SHA256']==q['public_source_manifest_SHA256']==source_sha
  assert q['completed']and q['protocol_before_destroy_passed']and q['all8_owned_workers_shutdown_completed']and not q['cleanup_errors']
  assert q['library_sha256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
  assert q['run']==run and q['node']==node and q['TP']==16 and q['PP']==2
  assert q['layout']==dict(node=node,local_workers=8,TP=16,TP_node_rank=node%2,PP=2,PP_stage=node//2,sampling_node=2,public_graph_enabled=False,owned_metadata_enabled=False)
  assert q['PP_ranges']==[[0,46],[46,93]]and q['cache_geometry']['max_total_tokens']==32768
  model_sha=q['manifest_SHA256']if model_sha is None else model_sha;assert q['manifest_SHA256']==model_sha
  assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3
  assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4
  masks=[]
  for owner in q['owner_affinity']['actual_owner_report']['owners']:
   assert owner['changes']==1;masks.extend(int(i)for i in owner['verified_mask'].split(','))
  assert len(masks)==len(set(masks))==32
  rows=q['agent_requests'];assert len(rows)==2 and q['serving_protocol_complete']
  sig=[]
  for index,row in enumerate(rows):
   c=client['rows'][index]
   assert row['request_id']==c['request_id']and row['input_tokens']==c['usage']['prompt_tokens']
   assert row['output_budget']==[4096,8192][index]and row['prefill_mode']=='W512'
   assert row['normal_reset']and row['finish']=='eos'and row['generated_ids'][-1]in[163585,163586]
   assert not row.get('error')and not row.get('cleanup_errors')
   h=row['history_render_audit'];assert h['qualified']and h['Native_prompt_token_ids_SHA256']==c['prompt_IDs_SHA256']
   assert h['RC_order_preserved']and h['RC_one_contribution_per_source_field']and not h['markers_sent_to_Native']
   observations=row['long_prefill_observations'];assert len(observations)==1 and observations[0]['full_prompt_consumed']
   assert observations[0]['input_tokens_committed']==row['input_tokens']
   sig.append([row['request_id'],h['Native_prompt_token_ids_SHA256'],row['input_tokens'],row['output_budget'],row['generated_ids'],row['finish']])
  assert not q.get('service_cleanup_errors')
  signatures.append(sig);pids[NODES[node]]=[claim['PID']];CAPI_calls.append(q['agent_native_forward_calls'])
  refs.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),claim_PID=claim['PID']))
 assert all(s==signatures[0]for s in signatures)and len(set(CAPI_calls))==1 and CAPI_calls[0]>0
 return dict(reports=refs,leader_count=4,Native_internal_owner_count=32,external_HTTP=2,Native_requests=2,
  expected_node_PIDs=pids,all4_prompt_output_reset_signatures_equal=True,
  prompt_output_signature_SHA256=hashlib.sha256(json.dumps(signatures[0],separators=(',',':')).encode()).hexdigest(),
  normal32destroy=True,finite_assertions_scope='Unchanged pinned Native.finish BF16 finite/flags assertions passed on actual tail outputs; no new hardware counter invented.',
  create_destroy_scope='Frozen owner source has exactly one create and one cleanup destroy per leader, no retry; actual successful create timing + normal8 flag per leader. No new CABI attempt counter.',
  semantic_gold_and_natural_EOS=True,public_API_scope_pass=True,performance_comparison=False,online_SLO=False)
