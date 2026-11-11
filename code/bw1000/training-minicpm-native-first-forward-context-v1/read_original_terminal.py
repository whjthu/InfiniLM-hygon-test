from pathlib import Path
import hashlib,json,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-first-forward-context-v1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
intent=json.loads((W/'intent.json').read_bytes());installed=json.loads((W/'installed.json').read_bytes());assert intent['claim']==installed['claim']==EXPECT['claim'] and intent['request_id']==EXPECT['request_id']
for n,pin in installed['files'].items():assert sha(W/'src'/n)==pin==EXPECT['source_pins'][n]
assert intent['deadline']==intent['epoch']+1200
result=dict(claim=intent['claim'],request_id=intent['request_id'],epoch=intent['epoch'],deadline=intent['deadline'],source_files_verified=len(installed['files']),frames=[],failures=[],raw_outputs=[],GPU_launched=False,Model_GO=False,numerical_acceptance=False,formal_performance=False)
if (W/'prelaunch-result.json').exists() and not (W/'gpu-intent.json').exists():
 raw=(W/'prelaunch-result.json').read_bytes();p=json.loads(raw);assert p['GPU_launched'] is False
 outer=json.loads((W/'outer-result.json').read_bytes());assert outer['supervisor_rc'] is not None and not outer['supervisor_observation_timeout']
 result.update(state='PRELAUNCH_TERMINAL_NO_GPU',independent_terminal_verified=True,prelaunch=p,prelaunch_SHA256=hashlib.sha256(raw).hexdigest(),completed_forwards=0,planned_forwards=8)
elif (W/'outcome.json').exists():
 assert sha(W/'outcome.json')==EXPECT['outcome_SHA256'];o=json.loads((W/'outcome.json').read_bytes());gpu=json.loads((W/'gpu-intent.json').read_bytes());assert gpu['epoch']==o['epoch']==intent['epoch'] and o['deadline']==intent['deadline']
 assert sha(W/'native.log')==o['native_log_SHA256'];config=json.loads((W/'src/CONFIG.json').read_bytes());states=[]
 for rank in range(8):
  root=W/'frames'/('rank-%02d'%rank)
  if (root/'failure.json').exists():result['failures'].append(dict(rank=rank,SHA256=sha(root/'failure.json'),value=json.loads((root/'failure.json').read_bytes())))
  if not (root/'result.json').exists():continue
  v=json.loads((root/'result.json').read_bytes());matching=[r for r in o['rank_results'] if r['rank']==rank];assert len(matching)==1 and matching[0]['value']==v and matching[0]['SHA256']==sha(root/'result.json')
  assert v['passed'] and v['completed_forwards']==1 and v['global_step']==v['optimizer_updates']==0 and v['source_SHA256']==EXPECT['plugin_SHA256']
  owner=v['owner'];assert owner['rank']==rank and owner['epoch']==intent['epoch'] and owner['deadline']==intent['epoch']+1080 and o['owners'][str(owner['PID'])]['startticks']==owner['startticks']
  init=json.loads((root/'initialization.json').read_bytes());assert init['parameter_tensors']==372 and init['parameter_numel']==10822656 and init['optimizer_state_entries']==0 and init['named_parameters_SHA256']==config['initial_named_SHA256']
  calls=json.loads((root/'calls.json').read_bytes());assert calls['dropout_calls']==186 and len(calls['chunk_calls'])==18 and all(x['calls']==1 for x in calls['chunk_calls'])
  begin=json.loads((root/'train-begin-context.json').read_bytes());entered=json.loads((root/'real-root-context.json').read_bytes());inside=json.loads((root/'inside-linear-context.json').read_bytes());layout=json.loads((root/'input-layouts.json').read_bytes())
  def compact(x):return {k:v for k,v in x.items() if k!='module_training'}
  metadata=dict(train_begin_context=compact(begin['context']),root_context=compact(entered['context']),inside_context=compact(inside),template_class=begin['template_class'],template_forward_context=begin['template_forward_context'],template_post_encode=begin['template_post_encode'],DDP_flags={k:x for k,x in entered.items() if k.startswith('DDP_')},layout_changes=layout['changes'],tensor_input_count=len(layout['actual']))
  pins={n:sha(root/n) for n in ('train-begin-context.json','real-root-context.json','inside-linear-context.json','input-layouts.json','initialization.json','calls.json','result.json')}
  p=root/'native-output.pt';assert p.is_file() and p.stat().st_size<=512*1024**2;result['raw_outputs'].append(dict(rank=rank,bytes=p.stat().st_size,SHA256=sha(p)))
  result['frames'].append(dict(rank=rank,value=v,contexts=metadata,metadata_SHA256=pins))
 alive=[]
 for pid,saved in o['owners'].items():
  p=Path('/proc',pid,'stat')
  try:
   st=p.read_text().split(') ',1)[1].split()
   if int(st[19])==saved['startticks'] and st[0]!='Z':alive.append(int(pid))
  except FileNotFoundError:pass
 result.update(state='ACTUAL_NATIVE_PROCESS_TERMINAL',independent_terminal_verified=not alive,GPU_launched=o['GPU_launched'],native_returncode=o['native_returncode'],original_keeper_passed=o['keeper_passed'],original_keeper_failure_preserved=not o['keeper_passed'],original_post=o['post'],all_owned_PIDs_absent=not alive,still_owned_PIDs=alive,cleanup=o['cleanup'],planned_forwards=8,completed_forwards=len(result['frames']),exact_reproduction_ranks=[r['rank'] for r in result['frames'] if r['value']['exact_native_reproduction']],native_phase_wall_seconds=o['native_phase_wall_seconds'],native_phase_timing=o['native_phase_timing'],outcome_SHA256=EXPECT['outcome_SHA256'])
elif not (W/'gpu-intent.json').exists() and (W/'outer-result.json').exists():
 outer=json.loads((W/'outer-result.json').read_bytes());owner=json.loads((W/'supervisor-owner.json').read_bytes());live=False;pidpath=Path('/proc',str(owner['PID']),'stat')
 if pidpath.exists():live=int(pidpath.read_text().split(') ',1)[1].split()[19])==owner['startticks']
 if outer['supervisor_rc'] is not None and not outer['supervisor_observation_timeout'] and not live:
  result.update(state='PRELAUNCH_CPU_FAILED_NO_GPU',independent_terminal_verified=True,supervisor_rc=outer['supervisor_rc'],supervisor_log_SHA256=sha(W/'supervisor.log'),completed_forwards=0,planned_forwards=8)
 else:result.update(state='UNKNOWN_NO_ORIGINAL_TERMINAL',independent_terminal_verified=False,supervisor_owner=owner)
else:
 owner=json.loads((W/'supervisor-owner.json').read_bytes()) if (W/'supervisor-owner.json').exists() else None
 result.update(state='UNKNOWN_NO_ORIGINAL_TERMINAL',independent_terminal_verified=False,supervisor_owner=owner)
result.update(observed_epoch=time.time(),raw_tensor_transfer_bytes=0,raw_tensor_values_recomputed=False)
print(json.dumps(result))
