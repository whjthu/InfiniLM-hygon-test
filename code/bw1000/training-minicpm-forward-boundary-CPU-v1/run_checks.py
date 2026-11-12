"""Run affected real tinyCPU hook/resume tests once; never open model assets."""
from pathlib import Path
import datetime,hashlib,importlib.metadata as md,io,json,os,resource,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework';sys.path.insert(0,str(F))
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();assert 0<deadline-time.time()<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU boundary epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
attempts=[]
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0])
  if path.startswith('/mnt/data/nvidia-baseline-20261010/inputs/') or '/train/resume-witness/' in path:
   attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('real model/data/capture access forbidden in tinyCPU scope')
sys.addaudithook(forbid);dump('CPU-owner.json',owner());phase='source'
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==353
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(1)
 def forbidden_device(*a,**k):raise RuntimeError('GPU init forbidden in CPU tests')
 torch.cuda._lazy_init=forbidden_device
 phase='real-tinyCPU-hooks-and-affected-resume'
 log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_forward_boundary_trace','vlm.tests.test_resume_witness']);assert suite.countTestCases()==26
 tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==26
 assert not torch.cuda.is_initialized() and not attempts
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=353,actual_tests=26,new_hook_tests=8,affected_resume_tests=18,new_resume_integration_tests=2,tests_SHA256=sha(W/'tests.private.txt'),versions={n:md.version(n) for n in ('torch','transformers','ms-swift','peft','safetensors')},wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,real_model_asset_open_attempts=attempts,real_model_loaded=False,tiny_synthetic_checkpoint_IO=True,Torch_imported=True,GPU_initialized=False,GPU_cards=0,HIP_target_qualified=False,GPU_GO=False,Model_GO=False,
  qualification='Real PyTorch CPU output/gradient/RNG/hooks/limits/partial-denominator and tiny full-state witness integration only;not actual52nativeMiniCPM resolver,realmodel/device,firstdivergence,recovery,numerics,performance or effects.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 if 'log' in globals() and not (W/'tests.private.txt').exists():(W/'tests.private.txt').write_text(log.getvalue())
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
