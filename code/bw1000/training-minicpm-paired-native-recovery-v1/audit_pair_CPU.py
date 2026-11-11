"""Independent0GPU post-terminal audit of recorded tails and full checkpoints."""
from pathlib import Path
import datetime,hashlib,json,os,signal,sys,time
ROOT=Path(sys.argv[1]);intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];started=time.monotonic();budget=min(120,epoch+1740-time.time());assert budget>0
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('independent pair audit original epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,budget)
deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());sys.path.insert(0,deps['framework']['source_directory'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner['node']=='f11r4n10' and os.environ['SLURM_CPUS_PER_TASK']=='4'
save('independent-pair-audit-owner.json',owner);result=None;rank_rows=[];raw_records=[];stage='original_native_terminal'
try:
 import torch
 from vlm.runtime.resume_pair import compare_resume_pair
 from vlm.runtime.resume_witness import pinned_checkpoint
 assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
 native=json.loads((ROOT/'native-outcome.json').read_bytes());assert native['normal_native_exit'] and len(native['arms'])==2 and native['cleanup']==dict(owned_processes_remaining=0,termination_needed=False)
 stage='full_rank_tail_records'
 for rank in range(8):
  values=[]
  for armidx,arm in enumerate(('control','restored')):
   base=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank);path=base/'result.json';raw=path.read_bytes();assert len(raw)<2*1024**2;value=json.loads(raw);actual_owner=json.loads((base/'owner.json').read_bytes());expected=native['arms'][armidx]['rank_records'][rank]
   assert value['rank']==actual_owner['rank']==actual_owner['local_rank']==rank and value['world_size']==8 and sha(path)==expected['result_SHA256'] and sha(base/'owner.json')==expected['owner_SHA256']
   assert actual_owner['epoch']==epoch and actual_owner['deadline']==epoch+1320 and value['native_resume_window_complete'] is True
   values.append(value);raw_records.append(dict(arm=arm,rank=rank,bytes=len(raw),SHA256=sha(path),owner_SHA256=sha(base/'owner.json')))
  rank_rows.append(compare_resume_pair(*values))
 stage='full_final_checkpoint_values'
 names=['trainer_state.json','args.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt']+['rng_state_%d.pth'%rank for rank in range(8)]
 checkpoints={};source_pins={}
 for key,folder in [('control20',ROOT/'control/train/checkpoint-20'),('control25',ROOT/'control/train/checkpoint-25'),('restored25',ROOT/'restored/train/checkpoint-25')]:
  pins={name:dict(bytes=(folder/name).stat().st_size,SHA256=sha(folder/name)) for name in names};source_pins[key]=dict(path=str(folder),files=pins);checkpoints[key]=pinned_checkpoint(folder,pins,8)
  value=checkpoints[key];step=20 if key=='control20' else 25;assert value['state']['global_step']==step and value['state']['max_steps']==25 and value['state']['epoch']==step/2 and value['args']['max_steps']==25 and value['args']['resume_only_model'] is False and value['args']['ignore_data_skip'] is False
  assert value['adapter']['tensors']==372 and value['adapter']['elements']==10822656
 generated=json.loads((ROOT/'generated-control-checkpoint20-pins.json').read_bytes());assert generated['checkpoint_files']==source_pins['control20']['files'] and generated['checkpoint_step']==20
 a,b=checkpoints['control25'],checkpoints['restored25'];final_equal={key:a[key]['SHA256']==b[key]['SHA256'] for key in ('adapter','optimizer','scheduler')};rng_equal={rank:a['RNG'][rank]['SHA256']==b['RNG'][rank]['SHA256'] for rank in range(8)}
 original=deps['original_checkpoint20_CPU_fingerprints'];at20=checkpoints['control20'];old20={key:at20[key]['SHA256']==original[key]['SHA256'] for key in ('adapter','optimizer','scheduler')};old20['rank_RNG']={rank:at20['RNG'][rank]['SHA256']==original['rank_RNG'][str(rank)]['SHA256'] for rank in range(8)}
 exact=all(row['all_recorded_tail_inputs_equal'] and row['all_recorded_tail_losses_exact'] and row['all_recorded_tail_states_exact'] and row['exact_observed_runtime_cursors']==20 for row in rank_rows) and all(final_equal.values()) and all(rng_equal.values())
 result=dict(passed=True,completed=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,original_deadline=epoch+1800,audit_deadline=epoch+1740,actual_max_seconds=120,rank_rows=rank_rows,original_rank_records=raw_records,checkpoint_sources=source_pins,checkpoint_value_fingerprints={key:{k:value[k] for k in ('adapter','optimizer','scheduler','RNG')} for key,value in checkpoints.items()},final_checkpoint_state_exact=final_equal,final_per_rank_RNG_exact=rng_equal,control20_matches_historical_original20=old20,compared_ranks=8,planned_tail_forwards=160,input_equal_tail_forwards=sum(row['input_equal_forwards'] for row in rank_rows),exact_loss_tail_forwards=sum(row['exact_loss_forwards'] for row in rank_rows),exact_recorded_state_updates=sum(row['exact_full_state_updates'] for row in rank_rows),same_platform_recovery_window_exact=exact,numerical_tolerance_invented=False,full_model_or_cross_platform_alignment_qualified=False,formal_performance=False,heldout_effects_qualified=False,Model_GO=False,CUDA_initialized=False,GPU_cards=0,raw_tensor_transfer_bytes=0,wall_seconds=time.monotonic()-started,qualification='Independent0GPU comparison of complete same-platform native continuation and fresh-process restore windows. Whole-model quality, A100 alignment/tolerances and formal performance are separate.')
except BaseException as error:
 result=dict(passed=False,completed=False,owner=owner,epoch=epoch,stage=stage,error_type=type(error).__name__,message=str(error)[:500],planned_ranks=8,planned_tail_forwards=160,completed_rank_comparisons=len(rank_rows),raw_records_seen=raw_records,wall_seconds=time.monotonic()-started,GPU_cards=0,Model_GO=False)
finally:
 save('independent-pair-audit.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)