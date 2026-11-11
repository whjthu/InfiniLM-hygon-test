from pathlib import Path
import hashlib,json,os,sys,time,unittest
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'framework';sys.path.insert(0,str(F));start=time.monotonic()
import torch
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
torch.set_num_threads(2)
suite=unittest.TestSuite()
for name in ('test_initial_trainables.py','test_native_initial_trainables.py'):
 suite.addTests(unittest.defaultTestLoader.discover(str(F/'vlm/tests'),pattern=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
value=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),torch=torch.__version__,GPU_context_initialized=torch.cuda.is_initialized(),GPU_devices_visible=torch.cuda.device_count(),wall_seconds=time.monotonic()-start,scope='Real PEFT/AdamW CPU trainable-copy and callback orchestration; tiny synthetic values; native Gemma model-contract validator is stubbed in callback fixture. No production initial-file read, full-model load, GPU, RNG replay, cross-platform numerical, recovery or performance qualification.')
assert not value['GPU_context_initialized']
(ROOT/'CPU-result.json').write_text(json.dumps(value,indent=2)+'\n')
if not value['passed']:raise SystemExit(1)
