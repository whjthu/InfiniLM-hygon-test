"""Read original captures in place; emit summaries/hashes, never raw tensors."""
from pathlib import Path
import gc,hashlib,importlib.util,inspect,json,os,resource,signal,socket,sys,time,traceback,unittest
epoch=float(os.environ['VALUES_EPOCH']);deadline=float(os.environ['VALUES_DEADLINE']);assert deadline==epoch+600 and epoch<=time.time()<deadline
def expired(*_):raise TimeoutError('original value-audit deadline exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time());started=time.monotonic();initial_left=deadline-time.time()
source=Path(__file__).resolve().parent;out=source.parent/'evidence';root=Path(json.loads((source/'PLAN.json').read_bytes())['capture_root']);sys.path.insert(0,str(source))
assert len(os.sched_getaffinity(0))<=4
assert all(os.environ[k]=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from vlm.runtime.first_update_audit import fingerprint,summarize_update
from vlm.runtime.initial_trainables import named_parameter_sha256
assert Path(inspect.getsourcefile(fingerprint)).resolve()==(source/'vlm/runtime/first_update_audit.py').resolve()
plan=json.loads((source/'PLAN.json').read_bytes());pins={(x['rank'],x['name']):x for x in plan['files']};assert len(pins)==200
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),CPU_affinity=sorted(os.sched_getaffinity(0)))
seen=set();read_bytes=0;stage='unit_tests';ranks=[];all_files=[]
def check():
 if time.time()>=deadline or time.monotonic()-started>=initial_left:raise TimeoutError('original value audit exhausted')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>8*1024**2:raise MemoryError('8GiB RSS guard exceeded; this is an observed guard, not a Slurm allocation')
def progress(failed=False):
 p=out/'progress.tmp';p.write_text(json.dumps(dict(owner=owner,stage=stage,planned_value_files=136,completed_value_files=len(seen),completed_ranks=len(ranks),planned_ranks=8,verified_source_file_bytes=read_bytes,reused_same_input_mask_identities=64,failed=failed)));p.replace(out/'progress.json')
def load(rank,name):
 global read_bytes
 check();key=(rank,name);assert key not in seen and key in pins;pin=pins[key];p=root/('rank-%02d'%rank)/name;before=p.stat();assert p.is_file() and not p.is_symlink() and before.st_size==pin['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):
   check();read_bytes+=len(raw);assert read_bytes<=plan['source_file_byte_budget'];h.update(raw)
 assert h.hexdigest()==pin['SHA256'];after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 value=torch.load(p,map_location='cpu',weights_only=True,mmap=True);seen.add(key);all_files.append(dict(rank=rank,name=name,SHA256=pin['SHA256'],bytes=pin['bytes']));progress();return value
def require_parameters(values,manifest):
 assert set(values)=={x['name'] for x in manifest}
 for x in manifest:
  t=values[x['name']];assert isinstance(t,torch.Tensor) and t.device.type=='cpu' and list(t.shape)==x['shape'] and str(t.dtype)==x['dtype'] and t.numel()==x['numel']
try:
 progress();assert plan['CPU_unit_tests_reused']['passed'] and plan['CPU_unit_tests_reused']['tests']==4 and plan['CPU_unit_tests_reused']['module_SHA256']==hashlib.sha256((source/'vlm/runtime/first_update_audit.py').read_bytes()).hexdigest()
 for rank in range(8):
  stage='rank_%d_initial'%rank;progress();initial=load(rank,'initial.pt');assert initial['global_step']==0
  manifest=initial['manifest'];assert len(manifest)==372 and sum(x['numel'] for x in manifest)==10822656;require_parameters(initial['parameters'],manifest)
  assert named_parameter_sha256([(x['name'],initial['parameters'][x['name']]) for x in manifest])==plan['initial_named_SHA256']
  initial_fp=fingerprint(initial['parameters']);del initial;gc.collect();forwards=[];updates=[]
  for ordinal in range(1,9):
   stage='rank_%d_forward_%d'%(rank,ordinal);progress();v=load(rank,'forward-%02d-output.pt'%ordinal);assert v['global_step']==(ordinal-1)//4 and v['micro_ordinal']==ordinal
   result=v['result'];loss=result['loss'];assert isinstance(loss,torch.Tensor) and loss.device.type=='cpu' and loss.numel()==1 and torch.isfinite(loss)
   logits=result['logits'];forwards.append(dict(ordinal=ordinal,global_step=v['global_step'],loss=float(loss),result_fingerprint=fingerprint(result),logits_shape=list(logits.shape),logits_dtype=str(logits.dtype)));del v,result,loss,logits;gc.collect()
  previous_parameters=initial_fp['SHA256'];previous_optimizer=None
  for ordinal in (1,2):
   stage='rank_%d_update_%d'%(rank,ordinal);progress()
   before=load(rank,'update-%02d-before-optimizer.pt'%ordinal);after=load(rank,'update-%02d-after-optimizer.pt'%ordinal);pre=load(rank,'update-%02d-preclip.pt'%ordinal);post=load(rank,'update-%02d-postclip.pt'%ordinal)
   require_parameters(before['parameters'],manifest);require_parameters(after['parameters'],manifest)
   summary=summarize_update(before,after,pre,post,ordinal=ordinal);assert summary['parameters_before']['SHA256']==previous_parameters
   if previous_optimizer is not None:assert summary['optimizer_state_before']['SHA256']==previous_optimizer
   assert summary['preclip_gradients']['floating_l2']>0 and summary['optimizer_after_steps']['populated']>0 and summary['optimizer_after_steps']['maximum']==ordinal
   previous_parameters=summary['parameters_after']['SHA256'];previous_optimizer=summary['optimizer_state_after']['SHA256'];updates.append(summary)
   del before,after,pre,post;gc.collect()
  assert any(x['parameter_update']['changed_elements']>0 for x in updates)
  ranks.append(dict(rank=rank,initial_parameters=initial_fp,forwards=forwards,updates=updates,at_least_one_nonzero_saved_parameter_update=True));progress()
 check();assert len(seen)==136 and len(ranks)==8 and not torch.cuda.is_initialized();stage='complete';progress()
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=plan['framework']['commit'],CPU_tests=0,CPU_unit_tests_reused=4,completed_value_files=len(seen),source_file_bytes_verified=read_bytes,reused_same_input_mask_identities=64,rank_summaries=ranks,plan_SHA256=hashlib.sha256((source/'PLAN.json').read_bytes()).hexdigest(),module_SHA256=hashlib.sha256((source/'vlm/runtime/first_update_audit.py').read_bytes()).hexdigest(),wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=str(torch.__version__),GPU_initialized=False,GPU_cards=0,raw_tensor_transfer_bytes=0,Hygon_original143_failure_preserved=True,cross_platform_numerical_accepted=False,fresh_recovery_qualified=False,formal_performance=False,Model_GO=False)
except BaseException as error:
 progress(True);(out/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,owner=owner,stage=stage,error_type=type(error).__name__,message=str(error)[:300],completed_value_files=len(seen),planned_value_files=136,completed_ranks=len(ranks),planned_ranks=8,source_file_bytes_verified=read_bytes,wall_seconds=time.monotonic()-started,GPU_cards=0,raw_tensor_transfer_bytes=0,Model_GO=False)
finally:
 encoded=json.dumps(result,indent=2);assert len(encoded)<1024**2
 with (out/'CPU-result.json').open('x') as f:f.write(encoded)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
