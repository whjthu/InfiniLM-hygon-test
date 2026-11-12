"""Three realCPU restored20 recording regressions before any new native load."""
from pathlib import Path
import hashlib,io,json,os,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=min(epoch+180,float(os.environ['CONTRACT_GPU_DEADLINE']))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('originalCPU recorder precheck expired')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):raise RuntimeError('model/data bodies forbidden inCPUprecheck')
 if event=='socket.connect':raise RuntimeError('network forbidden inCPUprecheck')
sys.addaudithook(forbid)
binding=json.loads((S/'BINDING.json').read_bytes());original=Path(binding['diagnostic_original_loss']['path']);raw=original.read_bytes()
assert len(raw)==6673 and hashlib.sha256(raw).hexdigest()==binding['diagnostic_original_loss']['SHA256']
os.environ['ZIMAGE_ORIGINAL_LOSS_SOURCE']=str(original)
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
def denied(*a,**k):raise RuntimeError('GPU forbidden inCPUrecorder precheck')
torch.cuda._lazy_init=denied
sys.path.insert(0,str(F/'generation/src'));sys.path.insert(0,str(F/'generation/tests'))
suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_zimage_resumed_update_capture.py');assert suite.countTestCases()==3
stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(W/'CPU-recorder.private.log').write_text(stream.getvalue())
result=dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),GPU_initialized=torch.cuda.is_initialized(),GPU_cards=0,
 epoch=epoch,deadline=deadline,model_or_dataset_loaded=False,original_loss_SHA256=hashlib.sha256(raw).hexdigest(),log_SHA256=hashlib.sha256(stream.getvalue().encode()).hexdigest())
with (W/'CPU-recorder-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
if not r.wasSuccessful() or r.skipped or torch.cuda.is_initialized():raise SystemExit(1)
