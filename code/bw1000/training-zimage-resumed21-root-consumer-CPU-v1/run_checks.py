"""CPU-only NumPy comparison regression and consumption of saved native arms."""
from pathlib import Path
import contextlib,hashlib,io,json,os,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);anchor=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('originalCPU consumer epoch expired')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
f=Path('/proc/self/stat').read_text().split(') ',1)[1].split();owner=dict(PID=os.getpid(),PGID=int(f[2]),startticks=int(f[19]));phase='source';tests=None
def audit(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):raise RuntimeError('pretrainedweights/dataset forbidden in CPU consumer')
 if event=='socket.connect':raise RuntimeError('network forbidden inCPUconsumer')
sys.addaudithook(audit);dump('CPU-owner.json',owner)
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==371
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(1)
 def denied(*a,**k):raise RuntimeError('GPU initialization forbidden inCPUconsumer')
 torch.cuda._lazy_init=denied;sys.path.insert(0,str(F/'generation/src'));sys.path.insert(0,str(F/'generation/tests'))
 phase='actualCPU-NumPy-RNG-regression'
 suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_checkpoint_compare.py');assert suite.countTestCases()==3
 stream=io.StringIO();report=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(W/'CPU-tests.private.log').write_text(stream.getvalue())
 tests=dict(tests=report.testsRun,failures=len(report.failures),errors=len(report.errors),skipped=len(report.skipped),passed=report.wasSuccessful())
 assert report.wasSuccessful() and not report.skipped and not torch.cuda.is_initialized()
 phase='consume-original-saved-native-arms-no-GPU'
 b=json.loads((S/'BINDING.json').read_bytes());scope=dict(CLAIM=b['original_claim'],REQUEST=b['original_request'],EXPECTED_FILES=b['original_remote_source_pins'],FRAMEWORK_ROOT=F,RECOVERY_ROOT=W,__name__='__main__')
 text=io.StringIO()
 with contextlib.redirect_stdout(text):exec(compile((S/'read_original_and_compare.py').read_bytes(),'independent_original_native_CPU_consumer','exec'),scope)
 value=json.loads(text.getvalue());assert value['passed'] and value['original_overall_passed'] is False and not torch.cuda.is_initialized()
 dump('ORIGINAL-CONSUME.json',value)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=371,CPU_tests=tests,
  original_consume_SHA256=sha(W/'ORIGINAL-CONSUME.json'),recovered_comparison=value['recovered_comparison'],
  reader_private_tensor_bytes_read=value['reader_private_tensor_bytes_read'],original_GPU_overall_passed=False,original_model_updates_reused=3,new_model_loads=0,new_GPU_runs=0,
  GPU_initialized=False,private_values_exported=False,wall_seconds=time.monotonic()-anchor,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-9000:]))
 result=dict(passed=False,owner=owner,epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,CPU_tests=tests,wall_seconds=time.monotonic()-anchor,new_GPU_runs=0,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),new_GPU_runs=0)))
if not result['passed']:raise SystemExit(1)
