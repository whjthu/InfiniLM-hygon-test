"""Real CPU recording regression with the exact original loss source;0GPU."""
from pathlib import Path
import datetime,hashlib,io,json,os,resource,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE'])
start=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU recorder epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):
  raise RuntimeError('model/data bodies forbidden in recorder CPU fixture')
 if event=='socket.connect':raise RuntimeError('network forbidden in recorder CPU fixture')
sys.addaudithook(forbid);dump('CPU-owner.json',owner());phase='source'
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==362
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 binding=json.loads((S/'BINDING.json').read_bytes());original=Path(binding['original_loss_path'])
 assert original.stat().st_size==6673 and sha(original)==binding['original_loss_SHA256']
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(1)
 def denied(*args,**kwargs):raise RuntimeError('GPU initialization forbidden in CPU recorder fixture')
 torch.cuda._lazy_init=denied
 os.environ['ZIMAGE_ORIGINAL_LOSS_SOURCE']=str(original)
 sys.path.insert(0,str(F/'generation/src'));sys.path.insert(0,str(F/'generation/tests'))
 proof=json.loads((S/'REUSE_PROOF.json').read_bytes());oldroot=Path(proof['previous_CPU_namespace'])
 assert sha(oldroot/'CPU-result.json')==proof['prior_CPU_result']['SHA256'] and sha(oldroot/'CPU-tests.private.log')==proof['prior_tests_log']['SHA256']
 prior=json.loads((oldroot/'CPU-result.json').read_bytes());assert prior['passed'] is False and prior['real_CPU_tests']==3 and prior['errors']==1 and prior['failures']==0
 oldmf=json.loads((oldroot/'src/FRAMEWORK_FILES.json').read_bytes())
 changed=[n for n in fm['files'] if fm['files'][n]!=oldmf['files'][n]];assert changed==['generation/tests/test_zimage_first_update_capture.py']
 def methods(path):
  import ast
  cls=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='RealCPUCapture')
  return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
 oldmethods=methods(oldroot/'framework/generation/tests/test_zimage_first_update_capture.py');newmethods=methods(F/'generation/tests/test_zimage_first_update_capture.py')
 for name,expected in proof['unchanged_method_AST_SHA256'].items():assert oldmethods[name]==newmethods[name] and hashlib.sha256(oldmethods[name].encode()).hexdigest()==expected
 phase='real-CPU-corrected-serializer-failure-only'
 suite=unittest.defaultTestLoader.loadTestsFromName('test_zimage_first_update_capture.RealCPUCapture.test_file_budget_failure_restores_loss_and_keeps_initial_progress')
 assert suite.countTestCases()==1
 stream=io.StringIO();report=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (W/'CPU-tests.private.log').write_text(stream.getvalue())
 assert not torch.cuda.is_initialized()
 result=dict(passed=report.wasSuccessful(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_files=362,real_CPU_tests=report.testsRun,
  failures=len(report.failures),errors=len(report.errors),skipped=len(report.skipped),
  native_loss_SHA256=sha(original),tests_log_SHA256=sha(W/'CPU-tests.private.log'),
  baseline_updates=0,recorded_updates=0,reused_prior_passed_cases=2,combined_cases_qualified=3,planned_first_capture_files=12,
  weight_or_dataset_bodies_read=False,GPU_initialized=False,GPU_cards=0,GPU_GO=False,Model_GO=False,
  wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  qualification='CPU fixture only:one corrected bytebudget negative test. Two prior native-loss/update fixturePASS cases reused via identical production sources and unchanged testmethodAST;prior overallFAIL retained. No repeated25updatefixture. Not original model/data training,CUDA/HIP,numerical/performance/recovery/effect qualification.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,owner=owner(),epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
