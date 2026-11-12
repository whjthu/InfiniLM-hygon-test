from pathlib import Path
import hashlib,importlib.metadata as metadata,inspect,json,os,resource,sys,time,traceback
W=Path(__file__).resolve().parents[1];S=W/'source';F=S/'framework';epoch=float(os.environ['SOURCE_EPOCH']);deadline=float(os.environ['SOURCE_GPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==660;sys.path[:0]=[str(F/'tools'),str(F/'generation/src')];result=None
try:
 import torch
 deps=json.loads((S/'DEPENDENCIES.json').read_bytes());versions={n:metadata.version(n) for n in deps['versions']};assert versions==deps['versions'];assert torch.version.hip is None and torch.version.cuda is not None and torch.cuda.device_count()==1;torch.set_num_threads(4)
 import hygon_sdxl_backward_audit as entry
 sys.argv=[str(F/'tools/hygon_sdxl_backward_audit.py'),'--reference',deps['reference'],'--policy',str(S/'policy.json'),'--weights',deps['weights'],'--output',str(W/'audit'),'--accelerator','cuda'];entry.main();report=json.loads((W/'audit/backward-audit.json').read_bytes());summary=json.loads((W/'audit/replay/summary.json').read_bytes());assert report['passed'] and summary['runtime']['accelerator']=='cuda'
 from diffusers import UNet2DConditionModel
 import peft
 providers={}
 for n,p in [('torch',Path(torch.__file__)),('torch_extension',Path(torch._C.__file__)),('peft',Path(peft.__file__)),('diffusers_UNet',Path(inspect.getfile(UNet2DConditionModel)))]:raw=p.read_bytes();providers[n]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 result=dict(passed=True,actual_optimizer_updates=1,versions=versions,torch_version=torch.__version__,CUDA=torch.version.cuda,runtime=summary['runtime'],peak_reserved_bytes=summary['peak_reserved_bytes'],providers=providers,flags=dict(BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32),diagnostic=report['diagnostic'],private_capture=report['private_capture'],observed_SHA256=summary['observed_sha256'],raw_capture_cross_host_transfer_bytes=0,formal_performance=False,Model_GO=False)
except BaseException as error:
 (W/'GPU-failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,error_type=type(error).__name__,raw_capture_cross_host_transfer_bytes=0,Model_GO=False)
result.update(epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])));(W/'GPU-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result.get(k) for k in ['passed','actual_optimizer_updates','error_type','Model_GO']}))
if not result['passed']:raise SystemExit(1)
