"""Actual target CPU F64 reference and passive SDK binary resolution; no GPU."""
from pathlib import Path
import ast,hashlib,importlib.metadata as md,importlib.util,json,os,resource,signal,subprocess,sys,time,traceback
S=Path('/input-stage');F=Path('/input-framework');W=Path('/evidence');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic()
assert deadline-epoch==220 and epoch<=time.time()<deadline
def check():
 if time.time()>=deadline:raise TimeoutError('original providerCPU epoch exhausted')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original providerCPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n13' and os.environ['SLURM_CPUS_PER_TASK']=='4'
attempts=[];phase='source';torch=None;cases=[];artifacts={};provider=None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024**2),b''):check();h.update(b)
 return h.hexdigest()
def forbid(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden in providerCPU scope')
 if event=='open' and args and isinstance(args[0],(bytes,str)):
  name=os.fsdecode(args[0])
  if name.endswith(('.safetensors','.ckpt','.bin')) or (name.endswith(('.pt','.pth')) and not name.startswith('/evidence/')):
   attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('original model/data tensor files forbidden')
sys.addaudithook(forbid)
def save(n,tensors):
 p=W/n;assert not p.exists();torch.save(tensors,p);artifacts[n]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 assert sum(v['bytes'] for v in artifacts.values())<=128*1024**2
try:
 for n,pin in json.loads((S/'SOURCE_FILES.json').read_bytes())['files'].items():
  p=S/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==382
 for n,pin in fm['files'].items():
  p=F/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 spec=json.loads((S/'SDK_SPEC.json').read_bytes());assert len(spec['cases'])==3
 phase='actual-CPU-controlled-input-and-independent-reference'
 import torch
 assert torch.version.hip==spec['expected_HIP'] and md.version('torch')==spec['expected_Torch'] and not torch.cuda.is_initialized()
 def deny_GPU(*a,**k):raise RuntimeError('GPU initialization forbidden in CPU reference scope')
 torch.cuda._lazy_init=deny_GPU;torch.set_num_threads(4)
 sys.path.insert(0,str(F/'generation/src'));from gencuda.zimage_hip_attention_probe import cpu_reference,compare_reference
 generator=torch.Generator(device='cpu');generator.manual_seed(20261113)
 for row in spec['cases']:
  check();inputs=[(torch.randn(tuple(row['shape']),generator=generator)*.1).to(torch.bfloat16) for _ in range(4)]
  hashes=[hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for t in inputs]
  assert hashes==row['input_SHA256'] and all(list(t.stride())==row['stride'] for t in inputs)
  reference=cpu_reference(inputs,torch)
  live=[t.permute(0,2,1,3).to(torch.float64).requires_grad_() for t in inputs[:3]]
  y=torch.nn.functional.scaled_dot_product_attention(*live,dropout_p=0.,is_causal=False);y.backward(inputs[3].permute(0,2,1,3).to(torch.float64))
  actual=[y.detach().permute(0,2,1,3).contiguous()]+[t.grad.detach().permute(0,2,1,3).contiguous() for t in live]
  metrics=compare_reference(actual,reference,torch);passed=all(v['max_abs']<=spec['reference_max_abs'] and v['relative_l2']<=spec['reference_relative_l2'] for v in metrics)
  seq=row['shape'][1];save('inputs-%d.private.pt'%seq,inputs);save('reference-%d.private.pt'%seq,reference)
  cases.append(dict(shape=row['shape'],stride=row['stride'],input_SHA256=hashes,CPU_F64_metrics=metrics,passed=passed));assert passed
 phase='actual-target-interpreter-extension-resolution'
 extension=importlib.util.find_spec(spec['extension_module']);assert extension is not None
 p=Path(extension.origin);assert p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=spec['binary_hash_limit_bytes']
 before=p.stat();provider=dict(path=str(p),bytes=before.st_size,SHA256=sha(p),module=spec['extension_module'],dlopen=False)
 after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert md.version('flash-attn')==spec['expected_flash_version'] and 'flash_attn_2_cuda' not in sys.modules
 dynamic=subprocess.run(['readelf','-d',str(p)],capture_output=True,text=True,timeout=min(12,deadline-time.time()-3));assert dynamic.returncode==0
 needed=[]
 for line in dynamic.stdout.splitlines():
  if '(NEEDED)' in line:needed.append(line.split('[',1)[1].split(']',1)[0])
 provider['DT_NEEDED']=needed;provider['dynamic_metadata_SHA256']=hashlib.sha256(dynamic.stdout.encode()).hexdigest()
 assert not torch.cuda.is_initialized() and not attempts
 value=dict(passed=True,state='TARGET_CPU_REFERENCE_INPUTS_AND_PASSIVE_SDK_PROVIDER_QUALIFIED',cases=cases,artifacts=artifacts,provider=provider,framework_commit=fm['framework_commit'],framework_files=382,original_A100_controlled_input_bytes_matched=True,CPU_F64_reference_qualified=True,actual_SDK_GPU_calls=0,CUDA_initialized=False,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
except BaseException as e:
 (W/'failure.private.txt').write_text(traceback.format_exc());value=dict(passed=False,phase=phase,error_type=type(e).__name__,cases=cases,artifacts=artifacts,provider=provider,CUDA_initialized=torch.cuda.is_initialized() if torch else None,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
value.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,forbidden_file_attempts=len(attempts))
(W/'CPU-result.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value))
if not value['passed']:raise SystemExit(1)
