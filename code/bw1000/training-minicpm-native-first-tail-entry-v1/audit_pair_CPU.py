"""Independent0GPU audit of SHA-bound first-tail entry records and checkpoint20."""
from pathlib import Path
import datetime,hashlib,json,os,signal,sys,time
ROOT=Path(sys.argv[1]);intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];started=time.monotonic();budget=min(120,epoch+1740-time.time());assert budget>0
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('first-tail audit original epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,budget)
deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());sys.path.insert(0,deps['framework']['source_directory'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner['node']=='f11r4n10' and os.environ['SLURM_CPUS_PER_TASK']=='4'
save('independent-pair-audit-owner.json',owner);result=None;rank_rows=[];raw_records=[];stage='original_native_terminal'
try:
 import torch
 from vlm.runtime.resume_entry_pair import compare_forward_entry_pair
 from vlm.runtime.resume_witness import pinned_checkpoint
 torch.set_num_threads(1)
 assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
 native=json.loads((ROOT/'native-outcome.json').read_bytes());assert native['normal_native_exit'] and len(native['arms'])==2 and native['cleanup']==dict(owned_processes_remaining=0,termination_needed=False)
 stage='full_rank_first_tail_entries'
 loss_deltas=[];entry_costs=[]
 for rank in range(8):
  values=[];entries=[]
  for armidx,arm in enumerate(('control','restored')):
   base=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank);path=base/'result.json';raw=path.read_bytes();assert 0<len(raw)<2*1024**2;value=json.loads(raw);entry_path=base/'forward-entry-state.json';entry_raw=entry_path.read_bytes();assert 0<len(entry_raw)<=8*1024**2;entry=json.loads(entry_raw);actual_owner=json.loads((base/'owner.json').read_bytes());expected=native['arms'][armidx]['rank_records'][rank]
   assert value['rank']==actual_owner['rank']==actual_owner['local_rank']==rank and value['world_size']==8 and sha(path)==expected['result_SHA256'] and sha(base/'owner.json')==expected['owner_SHA256']
   assert sha(entry_path)==value['entry_state_SHA256']==expected['entry_SHA256'] and len(entry_raw)==expected['entry_bytes']
   assert actual_owner['epoch']==epoch and actual_owner['deadline']==epoch+1320 and value['native_resume_window_complete'] is False and value['declared_observation_window_complete'] and value['diagnostic_stop_requested'] and value['completed_global_step']==21
   values.append(value);entries.append(entry_raw);raw_records.append(dict(arm=arm,rank=rank,bytes=len(raw),SHA256=sha(path),owner_SHA256=sha(base/'owner.json'),entry_bytes=len(entry_raw),entry_SHA256=sha(entry_path)))
   entry_costs.append(dict(arm=arm,rank=rank,wall_seconds=entry['wall_seconds'],logical_bytes_including_aliases=entry['logical_bytes_including_aliases'],unique_payload_bytes_read=entry['unique_payload_bytes_read'],peak_payload_chunk_bytes=entry['peak_payload_chunk_bytes'],registered_tensors=len(entry['registered_state'])))
  rank_rows.append(compare_forward_entry_pair(values[0],values[1],entries[0],entries[1]))
  loss_deltas.append(dict(rank=rank,first_tail_loss_magnitude_abs_delta=abs(values[0]['outputs'][-4]['loss']['floating_l2']-values[1]['outputs'][0]['loss']['floating_l2']),all_four_loss_magnitude_abs_deltas=[abs(x['loss']['floating_l2']-y['loss']['floating_l2']) for x,y in zip(values[0]['outputs'][-4:],values[1]['outputs'])]))
 stage='generated_checkpoint20_values'
 folder=ROOT/'control/train/checkpoint-20';names=['trainer_state.json','args.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt']+['rng_state_%d.pth'%rank for rank in range(8)]
 pins={name:dict(bytes=(folder/name).stat().st_size,SHA256=sha(folder/name)) for name in names};checkpoint=pinned_checkpoint(folder,pins,8)
 assert checkpoint['state']['global_step']==20 and checkpoint['state']['max_steps']==checkpoint['args']['max_steps']==25 and checkpoint['state']['epoch']==10.0 and checkpoint['args']['resume_only_model'] is False and checkpoint['args']['ignore_data_skip'] is False
 assert checkpoint['adapter']['tensors']==372 and checkpoint['adapter']['elements']==10822656 and len(checkpoint['RNG'])==8
 generated=json.loads((ROOT/'generated-control-checkpoint20-pins.json').read_bytes());assert generated['checkpoint_files']==pins and generated['checkpoint_step']==20
 result=dict(passed=True,completed=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,original_deadline=epoch+1800,audit_deadline=epoch+1740,actual_max_seconds=120,rank_rows=rank_rows,original_rank_records=raw_records,checkpoint_sources=dict(control20=dict(path=str(folder),files=pins)),checkpoint20_value_fingerprints={k:checkpoint[k] for k in ('adapter','optimizer','scheduler','RNG')},compared_ranks=8,planned_tail_forwards=32,input_equal_tail_forwards=sum(r['input_equal_forwards'] for r in rank_rows),exact_loss_tail_forwards=sum(r['exact_loss_forwards'] for r in rank_rows),first_tail_exact_loss_ranks=sum(r['first_tail_loss_exact'] for r in rank_rows),entry_equal_rank_counts={key:sum(r['entry_fields'][key] for r in rank_rows) for key in rank_rows[0]['entry_fields']},registered_state_equal_ranks=sum(r['registered_state_equal'] for r in rank_rows),final_observed_training_state_equal_ranks=sum(r['final_observed_training_state_equal'] for r in rank_rows),entry_observer_costs=entry_costs,loss_magnitude_deltas=loss_deltas,numerical_cause_attributed=False,full25_native_recovery_compared=False,recovery_qualified=False,formal_performance=False,heldout_effects_qualified=False,Model_GO=False,CUDA_initialized=False,GPU_cards=0,raw_tensor_transfer_bytes=0,wall_seconds=time.monotonic()-started,qualification='Independent first-tail diagnostic only; entry/state differences are observations, not an attributed cause or full recovery/quality/performance acceptance.')
except BaseException as error:
 result=dict(passed=False,completed=False,owner=owner,epoch=epoch,stage=stage,error_type=type(error).__name__,message=str(error)[:500],planned_ranks=8,planned_tail_forwards=32,completed_rank_comparisons=len(rank_rows),raw_records_seen=raw_records,wall_seconds=time.monotonic()-started,GPU_cards=0,Model_GO=False)
finally:
 save('independent-pair-audit.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
