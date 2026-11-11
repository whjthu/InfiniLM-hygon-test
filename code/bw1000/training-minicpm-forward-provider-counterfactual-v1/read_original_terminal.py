from pathlib import Path
import datetime,hashlib,json,subprocess
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-forward-provider-counterfactual-v1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
assert sha(W/'outcome.json')==E['outcome_SHA256'];o=json.loads((W/'outcome.json').read_bytes());intent=json.loads((W/'intent.json').read_bytes());gpu=json.loads((W/'gpu-intent.json').read_bytes());resume=json.loads((W/'resume-once/intent.json').read_bytes());rresult=json.loads((W/'resume-once/result.json').read_bytes())
assert o['claim']==intent['claim']==gpu['claim']==E['claim'] and resume['resume_claim']==rresult['claim']==E['resume_claim'] and resume['request_id']==rresult['request_id']==E['request']
assert o['epoch']==intent['epoch']==gpu['epoch']==resume['original_epoch']==rresult['original_epoch'] and o['deadline']==intent['deadline']==resume['original_deadline']==rresult['original_deadline']==o['epoch']+1200
assert not resume['epoch_reset'] and not rresult['epoch_reset'] and not resume['prior_GPU_launched']
assert rresult['supervisor_rc']==0 and not rresult['supervisor_observation_timeout'] and o['GPU_launched'] and o['native_returncode']==0 and not o['timed_out'] and o['error'] is None
assert o['cleanup']==dict(owned_processes_remaining=0,termination_needed=False) and not o['keeper_passed'] and o['post']['compute_apps']
installed=json.loads((W/'installed.json').read_bytes())
for name,pin in installed['files'].items():assert sha(W/'src'/name)==pin==E['files'][name]['SHA256']
assert sha(W/'native.log')==o['native_log_SHA256']
assert len(o['rank_results'])==8 and not o['rank_failures']
rankrows=[];hashes=[]
for row in sorted(o['rank_results'],key=lambda x:x['rank']):
 rank=row['rank'];root=W/'arms'/('rank-%02d'%rank);raw=(root/'result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['SHA256'];v=json.loads(raw);assert v==row['value']
 assert v['passed'] and v['completed_arms']==['native','torch_chunk'] and v['optimizer_updates']==v['global_step']==0 and v['source_SHA256']==E['plugin_SHA256'] and v['owner']['rank']==rank and v['owner']['deadline']==o['epoch']+1080
 owner=v['owner'];assert o['owners'][str(owner['PID'])]['startticks']==owner['startticks']
 init=json.loads((root/'initialization.json').read_bytes());assert init['named_parameters_SHA256']=='9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16' and init['parameter_tensors']==372 and init['parameter_numel']==10822656 and init['optimizer_state_entries']==0
 providers=json.loads((root/'providers-before.json').read_bytes());assert len(providers['bound_instances'])==18 and providers['native']['module'].startswith('fla.ops.gated_delta_rule') and providers['fallback']['name']=='torch_chunk_gated_delta_rule'
 calls=[]
 for arm in ('native','torch_chunk'):
  p=root/(arm+'-calls.json');x=json.loads(p.read_bytes());assert x['dropout_calls']==186 and x['actual_RNG_restored'] and len(x['chunk_calls'])==18 and all(r['calls']==1 for r in x['chunk_calls']);calls.append(dict(arm=arm,chunk_calls=18,dropout_calls=186,first_call_shapes=x['chunk_calls'][0]['first_call'],SHA256=sha(p)))
  p=root/(arm+'-output.pt');assert p.is_file() and p.stat().st_size<=512*1024**2;hashes.append(dict(rank=rank,arm=arm,bytes=p.stat().st_size,SHA256=sha(p)))
 rankrows.append(dict(rank=rank,owner=owner,source_result_SHA256=row['SHA256'],providers=providers,calls=calls,comparisons=v['comparisons'],exact_native_reproduction=v['exact_native_reproduction']))
alive=[]
for pid,owner in o['owners'].items():
 p=Path('/proc',pid,'stat')
 try:
  st=p.read_text().split(') ',1)[1].split()
  if int(st[19])==owner['startticks'] and st[0]!='Z':alive.append(int(pid))
 except FileNotFoundError:pass
assert not alive
postpids=[int(x.split(',')[1].strip()) for x in o['post']['compute_apps']];unregistered=[pid for pid in postpids if str(pid) not in o['owners']]
current=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
print(json.dumps(dict(independent_original_terminal_verified=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=E['claim'],resume_claim=E['resume_claim'],original_epoch=o['epoch'],original_deadline=o['deadline'],epoch_reset=False,source_files_verified=len(installed['files']),outcome_SHA256=E['outcome_SHA256'],native_log_SHA256=o['native_log_SHA256'],native_returncode=0,original_keeper_passed=False,original_keeper_failure_preserved=True,original_post_compute_apps=o['post']['compute_apps'],original_post_unregistered_PIDs=unregistered,all_original_owned_PIDs_absent=True,current_global_compute_apps=current,rank_results=rankrows,raw_outputs=hashes,planned_rank_arms=16,completed_rank_arms=16,planned_optimizer_updates=0,actual_optimizer_updates=0,exact_native_reproduction_ranks=sum(r['exact_native_reproduction'] for r in rankrows),raw_tensor_transfer_bytes=0,independent_raw_output_SHA_verified=True,independent_tensor_metric_recomputation=False,numerical_acceptance=False,formal_performance=False,Model_GO=False)))
