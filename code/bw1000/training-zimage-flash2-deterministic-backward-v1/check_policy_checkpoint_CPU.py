"""Real CPU checkpoint policy-identity checks before any SDK/native GPU work."""
from pathlib import Path
import hashlib,io,json,os,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework';epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=min(epoch+150,float(os.environ['CONTRACT_GPU_DEADLINE']))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('policy checkpoint precheck epoch expired')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def audit(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):raise RuntimeError('model/dataset bodies forbidden inCPUprecheck')
 if event=='socket.connect':raise RuntimeError('network forbidden inCPUprecheck')
sys.addaudithook(audit)
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
def denied(*a,**k):raise RuntimeError('GPU forbidden inCPUpolicychecks')
torch.cuda._lazy_init=denied;sys.path.insert(0,str(F/'generation/src'));sys.path.insert(0,str(F/'generation/tests'))
suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_zimage_attention_policy_checkpoint.py');assert suite.countTestCases()==2
stream=io.StringIO();report=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(W/'CPU-policy-checks.private.log').write_text(stream.getvalue())
r=dict(passed=report.wasSuccessful(),tests=report.testsRun,failures=len(report.failures),errors=len(report.errors),skipped=len(report.skipped),GPU_initialized=torch.cuda.is_initialized(),epoch=epoch,deadline=deadline,
 log_SHA256=hashlib.sha256(stream.getvalue().encode()).hexdigest(),native_models_loaded=0,GPU_cards=0)
with (W/'CPU-policy-checks.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
if not report.wasSuccessful() or report.skipped or torch.cuda.is_initialized():raise SystemExit(1)
