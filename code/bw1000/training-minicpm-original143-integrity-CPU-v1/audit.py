"""Read only exact original143 capture bytes and bounded checkpoint metadata."""
from pathlib import Path
import hashlib,json,os,signal,socket,stat,sys,time,traceback
W=Path(__file__).resolve().parent.parent;source=W/'source';evidence=W/'evidence'
epoch=float(os.environ['AUDIT_EPOCH']);deadline=float(os.environ['AUDIT_DEADLINE']);anchor=time.monotonic();initial_left=deadline-time.time()
assert deadline==epoch+240 and epoch<=time.time()<deadline
assert socket.gethostname().split('.')[0]=='f11r4n10' and os.environ['SLURM_JOB_ID']=='1087656'
assert all(os.environ[k]=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
signal.setitimer(signal.ITIMER_REAL,initial_left)
plan=json.loads((source/'ARTIFACT_PLAN.json').read_bytes());root=Path(plan['capture_root']);original=root.parent
assert plan['expected_file_count']==len(plan['files'])==200 and plan['expected_bytes']==19318791784
assert hashlib.sha256((original/'outcome.json').read_bytes()).hexdigest()==plan['original_outcome_SHA256']
owner=dict(node='f11r4n10',step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
checked=[];read_bytes=0;stage='capture_integrity';checkpoints=[]
def check():
 if time.time()>=deadline or time.monotonic()-anchor>=initial_left:raise TimeoutError('original CPU audit deadline exhausted')
def progress(failed=False):
 p=evidence/'progress.tmp';p.write_text(json.dumps(dict(owner=owner,stage=stage,planned_files=200,completed_files=len(checked),verified_capture_bytes=sum(x['bytes'] for x in checked),total_read_bytes=read_bytes,failed=failed)));p.replace(evidence/'progress.json')
def sha(path):
 global read_bytes
 check();before=path.lstat();assert stat.S_ISREG(before.st_mode) and not path.is_symlink()
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while True:
   check();part=stream.read(1024**2)
   if not part:break
   read_bytes+=len(part);assert read_bytes<=plan['total_read_budget'];h.update(part)
 after=path.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 return h.hexdigest()
def small_json(path):
 assert path.stat().st_size<2*1024**2;raw=path.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()
try:
 progress();seen=set()
 for row in plan['files']:
  rank=row['rank'];name=row['name'];assert type(rank) is int and 0<=rank<8 and Path(name).name==name and (rank,name) not in seen;seen.add((rank,name))
  p=root/('rank-%02d'%rank)/name;assert p.resolve().is_relative_to(root.resolve()) and p.stat().st_size==row['bytes']
  assert sha(p)==row['SHA256'],('capture SHA differs',rank,name)
  checked.append(row);progress()
 assert len(checked)==200 and sum(x['bytes'] for x in checked)==plan['expected_bytes']
 stage='baseline_and_checkpoint_inventory';progress();baselines=[]
 for rank in range(8):
  p=original/'train/nvidia-baseline'/('rank-%02d.json'%rank);v,pin=small_json(p)
  assert v['rank']==rank and v['world_size']==8 and v['global_step']==25 and v['trainable_parameters']==10822656 and v['initial_trainable_sha256']=='9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16'
  baselines.append({k:v[k] for k in ('rank','world_size','global_step','initial_trainable_sha256','final_trainable_sha256','trainable_parameters','checkpoint_steps','warmup_steps','measured_steps','full_measurement_seconds','samples_per_second','updates_per_second','peak_allocated_bytes','peak_reserved_bytes')})
  baselines[-1]['record_SHA256']=pin
 checkpoint_root=Path(plan['checkpoint_root']);entries=list(checkpoint_root.iterdir());assert len(entries)<=64
 for step in plan['checkpoint_steps']:
  folder=checkpoint_root/('checkpoint-%d'%step);row=dict(step=step,directory_present=folder.is_dir(),files=[],missing=[])
  expected=['trainer_state.json','optimizer.pt','scheduler.pt','adapter_model.safetensors','adapter_config.json']+['rng_state_%d.pth'%i for i in range(8)]
  if folder.is_dir():
   leaves=list(folder.iterdir());assert len(leaves)<=64;row['direct_entries']=[p.name for p in leaves]
   for name in expected:
    p=folder/name
    if not p.is_file():row['missing'].append(name);continue
    assert p.resolve().is_relative_to(folder.resolve()) and p.stat().st_size<512*1024**2
    row['files'].append(dict(name=name,bytes=p.stat().st_size,SHA256=sha(p)))
   p=folder/'trainer_state.json'
   if p.is_file():
    state,_=small_json(p);row['actual_global_step']=state['global_step'];assert state['global_step']==step
    row['state_fields']=sorted(state);row['log_history_records']=len(state.get('log_history',[]))
  else:row['missing']=expected
  checkpoints.append(row);progress()
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,original_GPU=plan['original_GPU'],original_post=plan['original_post'],verified_capture_files=len(checked),verified_capture_bytes=sum(x['bytes'] for x in checked),total_read_bytes=read_bytes,capture_plan_SHA256=hashlib.sha256((source/'ARTIFACT_PLAN.json').read_bytes()).hexdigest(),baselines=baselines,checkpoint_inventory=checkpoints,checkpoint_expected_files_complete=all(not x['missing'] for x in checkpoints),wall_seconds=time.monotonic()-anchor,torch_imported='torch' in sys.modules,GPU_cards=0,original_GPU_failure_preserved=True,raw_tensor_content_numerically_compared=False,recovery_qualified=False,formal_performance=False,Model_GO=False)
except BaseException as error:
 progress(True);(evidence/'failure.private.txt').write_text(traceback.format_exc())
 result=dict(passed=False,owner=owner,stage=stage,error_type=type(error).__name__,message=str(error)[:300],planned_capture_files=200,verified_capture_files=len(checked),verified_capture_bytes=sum(x['bytes'] for x in checked),total_read_bytes=read_bytes,wall_seconds=time.monotonic()-anchor,GPU_cards=0,Model_GO=False)
finally:
 with (evidence/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
