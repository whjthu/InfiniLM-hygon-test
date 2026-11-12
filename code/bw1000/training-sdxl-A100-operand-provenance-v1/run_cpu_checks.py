from pathlib import Path
import hashlib,io,json,os,resource,sys,time,traceback,unittest
W=Path(__file__).resolve().parents[1];F=W/'source/framework';epoch=float(os.environ['SOURCE_EPOCH']);deadline=float(os.environ['SOURCE_CPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==120
sys.path[:0]=[str(F/'generation/src'),str(F/'tools')];result=None
try:
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_k):raise RuntimeError('GPU context forbidden in source CPU checks')
 torch.cuda._lazy_init=forbid;log=io.StringIO();suite=unittest.TestSuite()
 for root,pattern in [(F/'generation/tests','test_lora_backward_audit.py'),(F/'tests','test_sdxl_fixed_replay_platform.py')]:suite.addTests(unittest.defaultTestLoader.discover(str(root),pattern=pattern,top_level_dir=str(root)))
 tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'CPU-tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==14 and not torch.cuda.is_initialized();result=dict(passed=True,actual_CPU_tests=14,CUDA_initialized=False,GPU_cards=0,model_weight_bytes_read=0,reference_body_bytes_read=0)
except BaseException as error:
 (W/'CPU-failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,error_type=type(error).__name__,GPU_cards=0)
result.update(epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])));(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
