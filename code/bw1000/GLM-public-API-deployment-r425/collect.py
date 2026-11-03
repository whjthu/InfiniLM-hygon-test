"""Read actual public-provider terminal reports; no imports of device code."""
from pathlib import Path
import hashlib,json,time
def collect(root,run,actual,source_sha,client_result,deadline):
 root=Path(root);folder=root/'results/public-GLM-provider-r425'/run;reports=[];refs=[];pids={}
 def read(path):
  while time.time()<deadline:
   try:return json.loads(path.read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Terminal artifact visibility:'+str(path))
 client=read(Path(client_result));assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==2
 signatures=[]
 for stage in range(2):
  leader=read(folder/('leader-%d.json'%stage));actors=read(folder/('actors-%d.json'%stage));pids[('a02r4n12','a02r4n13')[stage]]=sorted(set([leader['PID']]+actors['owned_process_PIDs']))
  for rank in range(8):
   path=folder/('owner-%d-%d.json'%(stage,rank));q=read(path);life=read(folder/('lifecycle-%d-%d.json'%(stage,rank)))
   assert q['actual_step']==actual and q['source_manifest_SHA256']==source_sha
   assert q['completed']and q['normal_shutdown']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']
   assert life['normal']and life['destroy_attempts']==life['shutdown_attempts']==1 and life['destroy_RC']==life['shutdown_RC']==0
   assert q['all16_preweight_8GiB_passed']and q['all16_loaded_and_postload_4GiB_passed']
   def masks(value):return value['width'],sorted((r['device'],r['changes'],r['verified_mask'])for r in value['actual_owner_report']['owners'])
   assert masks(q['owner_affinity_after_load'])==masks(q['owner_affinity_final'])
   rows=q['service_requests'];assert q['attempted_Native']==q['completed_Native']==len(rows)==2
   sig=[]
   for index,r in enumerate(rows):
    c=client['rows'][index];assert r['native_profile']==c['profile']
    assert r['generated_natural_EOS']and r['original_reset_all16_completed']and r['reason']=='eos_token'
    assert r['actual_prompt_token_ids_SHA256']==c['prompt_IDs_SHA256']
    assert all(x['CAPI_returned']and x['CAPI_RC']==0 and (x['finite']is True if stage==1 else x['finite']is None)for x in r['native_calls'])
    sig.append((r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['native_profile']))
   signatures.append(sig);reports.append(q);refs.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert all(s==signatures[0]for s in signatures)
 return dict(reports=refs,actual_owner_count=16,external_requests=2,logical_requests_per_owner=2,
   logical_requests_total_owners=32,Native_CAPI_calls=sum(len(r['native_calls'])for q in reports for r in q['service_requests']),
   expected_node_PIDs=pids,semantic_gold_and_natural_EOS=True,normal16destroy=True,
   public_API_scope_pass=True,performance_comparison=False,online_SLO=False)
