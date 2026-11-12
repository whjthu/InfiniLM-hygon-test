"""Inspect actual HIP training dependencies with all device initialization forbidden."""
from pathlib import Path
import datetime,hashlib,importlib,importlib.metadata as md,importlib.util,json,os,resource,signal,sys,time,traceback
W=Path('/evidence');S=Path('/input-stage');start=time.monotonic()
epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE'])
assert deadline-epoch==220 and epoch<=time.time()<deadline
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('single target CPU epoch exhausted')))
signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner['node']=='f11r4n13' and os.environ['SLURM_CPUS_PER_TASK']=='4'
assert all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
attempts=[];versions={};imports={};phase='source';torch=None
def forbidden(event,args):
 if event=='socket.connect':raise RuntimeError('network access forbidden in target CPU inspection')
 if event=='open' and args and isinstance(args[0],(bytes,str)):
  p=os.fsdecode(args[0])
  if p.endswith(('.safetensors','.pt','.pth','.ckpt','.bin')):
   attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('tensor and weight file access forbidden')
sys.addaudithook(forbidden)
def pin(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();size=p.stat().st_size
 assert size<8*1024**2
 return dict(path=str(p),bytes=size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
try:
 manifest=json.loads((S/'SOURCE_FILES.json').read_bytes())
 for n,v in manifest['files'].items():
  p=S/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['SHA256']
 phase='actual-HIP-torch-import';import torch
 assert torch.version.hip=='6.3.26113' and not torch.cuda.is_initialized()
 torch.set_num_threads(1)
 def deny_GPU(*a,**k):raise RuntimeError('device initialization forbidden in CPU dependency inspection')
 torch.cuda._lazy_init=deny_GPU
 providers={'torch':pin(torch.__file__),'torch_extension':pin(torch._C.__file__)}
 phase='actual-dependent-imports'
 distributions=['torch','numpy','transformers','diffusers','peft','accelerate','safetensors','einops','Pillow','torchvision','flash-attn','xformers']
 for name in distributions:
  try:versions[name]=dict(installed=True,version=md.version(name))
  except md.PackageNotFoundError:versions[name]=dict(installed=False)
 for name in ['numpy','transformers','diffusers','peft','accelerate','safetensors','einops','PIL','torchvision']:
  try:
   module=importlib.import_module(name);imports[name]=dict(passed=True,provider=pin(module.__file__))
  except Exception as e:
   (W/(name+'-import-error.private.txt')).write_text(traceback.format_exc())
   imports[name]=dict(passed=False,error_type=type(e).__name__,error_message_SHA256=hashlib.sha256(str(e).encode()).hexdigest(),missing_module=getattr(e,'name',None))
  assert not torch.cuda.is_initialized()
 spec=importlib.util.find_spec('diffsynth')
 diffsynth=dict(found=spec is not None,origin=spec.origin if spec else None,import_executed=False)
 missing=[n for n,v in imports.items() if not v['passed']]
 result=dict(passed=True,state='TARGET_HIP_CPU_DEPENDENCY_INSPECTION_COMPLETE',versions=versions,imports=imports,providers=providers,diffsynth=diffsynth,missing_required_imports=missing,native_source_import_GO=bool(not missing and spec is not None),native_model_import_executed=False,CUDA_initialized=False,HIP=torch.version.hip,weights_or_data_read_bytes=0,all17_final_accepted=0,Model_GO=False,GPU_GO=False)
except BaseException as e:
 (W/'failure.private.txt').write_text(traceback.format_exc())
 result=dict(passed=False,phase=phase,error_type=type(e).__name__,versions=versions,imports=imports,CUDA_initialized=torch.cuda.is_initialized() if torch else None,Model_GO=False,GPU_GO=False)
result.update(owner=owner,epoch=epoch,deadline=deadline,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,forbidden_file_attempts=len(attempts),GPU_cards=0)
(W/'CPU-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
