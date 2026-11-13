"""Capture exact passive ldd mappings before assertions; no SDK/device execution."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,resource,signal,subprocess,sys,time,traceback
W=Path('/run-scope');S=W/'src';E=W/'evidence';intent=json.loads((W/'intent.json').read_bytes());epoch=intent['epoch'];deadline=epoch+90;start=time.monotonic();assert epoch<=time.time()<deadline
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original passive CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n13'
phase='source';torch=None;names={};observed={};total=0;file_attempts=[]
def check():
 if min(deadline-time.time(),90-(time.monotonic()-start))<=0:raise TimeoutError('original passive CPU epoch exhausted')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(4*1024**2),b''):check();h.update(raw)
 return h.hexdigest()
def audit(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden')
 if event=='open' and args and isinstance(args[0],(bytes,str)) and os.fsdecode(args[0]).endswith(('.safetensors','.pt','.pth','.ckpt','.bin')):
  file_attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('model/data/tensor files forbidden')
sys.addaudithook(audit)
try:
 for n,pin in json.loads((S/'SOURCE_FILES.json').read_bytes())['files'].items():
  p=S/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 binding=json.loads((S/'LIBRARY_BINDING.json').read_bytes());sdk=json.loads((S/'SDK_BINDING.json').read_bytes());envraw=Path('/qualified-preflight/evidence/loader-environment.private.json').read_bytes();assert hashlib.sha256(envraw).hexdigest()==binding['loader_environment_pin']['SHA256'];env=json.loads(envraw);assert os.environ['LD_LIBRARY_PATH']==env['LD_LIBRARY_PATH']
 phase='masked-Torch-and-primary-provider'
 import torch
 assert not torch.cuda.is_initialized() and torch.version.hip==sdk['HIP']
 def deny(*a,**k):raise RuntimeError('GPU initialization forbidden')
 torch.cuda._lazy_init=deny
 assert os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']==os.environ['CUDA_VISIBLE_DEVICES']=='-1'
 for pin in sdk['provider_files'].values():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='actual-passive-map'
 r=subprocess.run(['ldd',sdk['provider_files']['extension']['path']],capture_output=True,text=True,timeout=min(15,deadline-time.time()-5));(E/'ldd.private.txt').write_text(r.stdout+r.stderr);assert r.returncode==0 and 'not found' not in r.stdout
 from ldd_map import parse_ldd,classify
 names=parse_ldd(r.stdout)
 (E/'observed-name-paths.json').write_text(json.dumps(names,indent=2)+'\n')
 expected_names={name:str(Path(path).resolve(strict=True)) for name,path in binding['resolved_names'].items()}
 unique={}
 for name,path in names.items():
  p=Path(path).resolve(strict=True);s=p.stat();assert p.is_file() and not p.is_symlink() and 0<s.st_size<=4*1024**3
  if str(p) not in unique:
   total+=s.st_size;assert total<=16*1024**3;unique[str(p)]=dict(realpath=str(p),bytes=s.st_size,SHA256=sha(p))
  after=p.stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns]
  observed[name]=dict(unique[str(p)],requested_path=path)
 comparison=classify(expected_names,binding['resolved_libraries'],observed)
 differences=[dict(name=n,expected_path=binding['resolved_names'].get(n),observed_path=names.get(n),expected_realpath=expected_names.get(n),observed_realpath=observed.get(n,{}).get('realpath')) for n in sorted(set(names)|set(binding['resolved_names'])) if names.get(n)!=binding['resolved_names'].get(n)]
 assert not torch.cuda.is_initialized() and 'flash_attn_2_cuda' not in sys.modules and not file_attempts
 value=dict(passed=True,state='COMPUTE_CPU_PASSIVE_LOADER_MAP_CAPTURED',comparison=comparison,literal_path_differences=differences,literal_name_path_maps_equal=names==binding['resolved_names'],expected_names=binding['resolved_names'],observed_names=names,observed_libraries=observed,unique_library_count=len(unique),dependency_bytes_hashed=total,ldd_SHA256=sha(E/'ldd.private.txt'),observed_name_paths_SHA256=sha(E/'observed-name-paths.json'),loader_environment_pin=binding['loader_environment_pin'],loader_environment_applied_before_interpreter_start=True,ldd_child_device_effects_observed=False,extension_dlopen=False,CUDA_initialized=False,explicit_SDK_GPU_calls=0)
except BaseException as e:
 (E/'failure.private.txt').write_text(traceback.format_exc());value=dict(passed=False,phase=phase,error_type=type(e).__name__,observed_names=names,observed_libraries=observed,CUDA_initialized=torch.cuda.is_initialized() if torch else None)
value.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,forbidden_file_attempts=len(file_attempts),GPU_cards=0,weights_or_data_read_bytes=0,GPU_GO=False,Model_GO=False)
(E/'CPU-result.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value))
if not value['passed']:raise SystemExit(1)
