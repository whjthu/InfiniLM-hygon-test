"""Actual CPU tests for guarded storage and passive SDK link dependencies."""
from pathlib import Path
import ast,hashlib,importlib.metadata as md,importlib.util,json,os,re,resource,signal,subprocess,sys,time,traceback
S=Path('/input-stage');F=Path('/input-framework');W=Path('/evidence');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic();assert deadline-epoch==220 and epoch<=time.time()<deadline
def check():
 if time.time()>=deadline:raise TimeoutError('original preflightCPU epoch exhausted')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original preflightCPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n13' and os.environ['SLURM_CPUS_PER_TASK']=='4'
attempts=[];phase='source';torch=None;guard_cases=[];libraries={};hashed_bytes=0
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024**2),b''):check();h.update(b)
 return h.hexdigest()
def forbid(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden in SDKpreflightCPU scope')
 if event=='open' and args and isinstance(args[0],(bytes,str)):
  name=os.fsdecode(args[0])
  if name.endswith(('.safetensors','.ckpt','.bin','.pt','.pth')):
   attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('model/data/reference tensor files forbidden in SDKpreflightCPU scope')
sys.addaudithook(forbid)
try:
 for n,pin in json.loads((S/'SOURCE_FILES.json').read_bytes())['files'].items():
  p=S/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==382
 for n,pin in fm['files'].items():
  p=F/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 binding=json.loads((S/'REFERENCE_BINDING.json').read_bytes());phase='actual-CPU-guard-storage'
 import torch
 assert torch.version.hip==binding['versions']['HIP'] and md.version('torch')==binding['versions']['torch'] and not torch.cuda.is_initialized()
 def deny_GPU(*a,**k):raise RuntimeError('GPU initialization forbidden in preflightCPU scope')
 torch.cuda._lazy_init=deny_GPU;torch.set_num_threads(4)
 sys.path.insert(0,str(F/'generation/src'));from gencuda.zimage_hip_attention_probe import guarded_input,audit_inputs
 for shape in [(1,2,3,4),(1,288,30,128)]:
  count=1
  for n in shape:count*=n
  fixture=(torch.arange(count,dtype=torch.float32).remainder(97).sub(48).mul(1/512)).to(torch.bfloat16).reshape(shape)
  positive=guarded_input(fixture,torch,device='cpu');assert positive['flat'].data_ptr()!=positive['host'].data_ptr() and positive['tensor'].is_leaf and torch.equal(positive['tensor'],fixture)
  checked=audit_inputs([positive],torch);assert checked==fixture.numel()*2+512
  negatives=[]
  for location,index in [('prefix',0),('payload',256),('suffix',-1)]:
   bad=guarded_input(fixture,torch,device='cpu');bad['flat'][index]=(int(bad['flat'][index])+1)%256
   try:audit_inputs([bad],torch)
   except ValueError:negatives.append(location)
   else:raise AssertionError('mutated actual shared guard storage was not rejected')
  guard_cases.append(dict(shape=list(shape),positive=True,independent_host_storage=True,guard_and_payload_bytes_checked=checked,rejected_corruptions=negatives))
 phase='passive-actual-SDK-resolved-link-map'
 extension=importlib.util.find_spec(binding['provider']['module']);assert extension is not None
 p=Path(extension.origin);assert p.resolve()==Path(binding['provider']['path']).resolve() and p.stat().st_size==binding['provider']['bytes'] and sha(p)==binding['provider']['SHA256']
 torch_lib=str(Path(torch.__file__).resolve().parent/'lib');original_ld=os.environ.get('LD_LIBRARY_PATH','');os.environ['LD_LIBRARY_PATH']=torch_lib+(':'+original_ld if original_ld else '')
 environment={k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','PATH','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']};environment['Torch_lib_prefix']=torch_lib
 env_path=W/'loader-environment.private.json';env_path.write_text(json.dumps(environment,indent=2)+'\n')
 dynamic=subprocess.run(['ldd',str(p)],capture_output=True,text=True,timeout=min(15,deadline-time.time()-5));(W/'ldd.private.txt').write_text(dynamic.stdout+dynamic.stderr);assert dynamic.returncode==0 and 'not found' not in dynamic.stdout
 resolved={};virtual=[]
 for line in dynamic.stdout.splitlines():
  words=line.strip().split()
  if not words:continue
  if words[0].startswith('linux-vdso'):virtual.append(words[0]);continue
  if '=>' in words:
   assert len(words)>=3 and words[2].startswith('/');resolved[words[0]]=words[2]
  else:
   assert words[0].startswith('/');resolved[Path(words[0]).name]=words[0]
 assert set(binding['provider']['DT_NEEDED']).issubset(resolved) and 'libflash_attention.so' in resolved
 for name,path in resolved.items():
  check();q=Path(path).resolve(strict=True);before=q.stat();assert q.is_file() and not q.is_symlink() and before.st_size<=4*1024**3
  key=str(q)
  if key not in libraries:
   hashed_bytes+=before.st_size;assert hashed_bytes<=16*1024**3;libraries[key]=dict(path=key,bytes=before.st_size,SHA256=sha(q),requested_names=[],requested_paths=[])
  libraries[key]['requested_names'].append(name);libraries[key]['requested_paths'].append(path)
  after=q.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert 'flash_attn_2_cuda' not in sys.modules and not torch.cuda.is_initialized() and not attempts
 value=dict(passed=True,state='TARGET_CPU_GUARD_AND_PASSIVE_RESOLVED_SDK_LINK_MAP_QUALIFIED',guard_cases=guard_cases,guard_positive=2,guard_corruptions_rejected=6,resolved_libraries=libraries,resolved_names=resolved,virtual_libraries=virtual,total_dependency_bytes_hashed=hashed_bytes,loader_environment_pin=dict(bytes=env_path.stat().st_size,SHA256=sha(env_path)),loader_environment_recipe='Prefix actual Torch lib directory to complete inherited DTK LD_LIBRARY_PATH;preserve full tail. Future device scope must use identical bytes.',passive_ldd_SHA256=sha(W/'ldd.private.txt'),ldd_child_device_effects_observed=False,explicit_SDK_GPU_calls=0,CUDA_initialized=False,extension_dlopen=False,framework_commit=fm['framework_commit'],framework_files=382,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
except BaseException as e:
 (W/'failure.private.txt').write_text(traceback.format_exc());value=dict(passed=False,phase=phase,error_type=type(e).__name__,guard_cases=guard_cases,resolved_libraries=libraries,total_dependency_bytes_hashed=hashed_bytes,CUDA_initialized=torch.cuda.is_initialized() if torch else None,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
value.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,forbidden_file_attempts=len(attempts))
(W/'CPU-result.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value))
if not value['passed']:raise SystemExit(1)
