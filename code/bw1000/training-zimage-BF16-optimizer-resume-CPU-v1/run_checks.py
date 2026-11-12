"""One CPU-only actual-BF16-operand optimizer restoration experiment."""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);started=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('BF16optimizer CPU epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):
  raise RuntimeError('pretrainedweights/dataset reads forbidden in optimizer-only CPU experiment')
 if event=='socket.connect':raise RuntimeError('network forbidden in CPU optimizer experiment')
sys.addaudithook(forbid);dump('CPU-owner.json',owner());phase='source'
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==366
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(1)
 def denied(*a,**k):raise RuntimeError('GPU initialization forbidden in CPU optimizer experiment')
 torch.cuda._lazy_init=denied
 binding=json.loads((S/'BINDING.json').read_bytes());tool=F/'tools/hygon_bf16_saved_gradient_resume_probe.py'
 spec=importlib.util.spec_from_file_location('published_actual_BF16_optimizer_probe',tool);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 phase='actual-BF16-values-controlled-CPU-optimizer-continuation'
 probe=module.run(binding,W/'probe',epoch=epoch,deadline=deadline)
 assert not torch.cuda.is_initialized()
 result=dict(passed=probe['passed'],owner=owner(),epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=366,
  probe=probe,probe_result_SHA256=sha(W/'probe/result.json'),wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  CPU_only=True,GPU_cards=0,GPU_initialized=False,native_model_loaded=False,native_model_training_updates=0,
  pretrained_weight_or_dataset_bodies_read=False,private_operands_exported=False,actual_CPU_optimizer_steps=probe['actual_optimizer_steps'],
  original_native_resume_FAIL_not_reclassified=True,CUDA_cause_qualified=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-9000:]))
 result=dict(passed=False,owner=owner(),epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,wall_seconds=time.monotonic()-started,GPU_cards=0,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
