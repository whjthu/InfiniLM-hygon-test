"""Run the exact framework dropout tests with all accelerators masked."""
import datetime,hashlib,importlib.metadata as metadata,io,json,os,runpy,signal,time,unittest
from pathlib import Path
ROOT=Path('/replay');epoch=float(os.environ['CPU_EPOCH']);deadline=float(os.environ['CPU_DEADLINE'])
assert deadline-epoch==180 and epoch<=time.time()<deadline
signal.alarm(max(1,min(120,int(deadline-time.time()-5))))
assert all(os.environ[n]=='-1' for n in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
import torch
assert not torch.cuda.is_initialized()
runtime={'torch':str(torch.__version__),'torch_distribution':metadata.version('torch'),'torch_file':torch.__file__,'hip':torch.version.hip,'device_initialized':torch.cuda.is_initialized()}
with (ROOT/'runtime-inventory.json').open('x') as f:json.dump(runtime,f,indent=2)
assert metadata.version('torch')=='2.10.0+das.opt1.dtk2604.2606161421.g91a580'
assert str(torch.__version__)=='2.10.0'
torch.set_num_threads(2)
namespace=runpy.run_path(str(ROOT/'framework/vlm/tests/test_dropout_capture.py'))
log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(namespace['DropoutCaptureTest'])
started=time.monotonic();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
pins={}
for name in ('vlm/runtime/dropout_capture.py','vlm/tests/test_dropout_capture.py'):
    pins[name]=hashlib.sha256((ROOT/'framework'/name).read_bytes()).hexdigest()
out={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epoch':epoch,'deadline':deadline,'tests':result.testsRun,
    'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-started,
    'torch':str(torch.__version__),'torch_distribution':metadata.version('torch'),'torch_file':torch.__file__,'hip':torch.version.hip,'device_initialized':False,
    'source_SHA256':pins,'log':log.getvalue(),'GPU_kernel_qualification':False,'training_qualification':False}
with (ROOT/'CPU-result.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
raise SystemExit(0 if result.wasSuccessful() else 1)
