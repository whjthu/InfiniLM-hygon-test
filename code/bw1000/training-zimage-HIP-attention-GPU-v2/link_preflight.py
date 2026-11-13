"""Revalidate the actual pinned SDK libraries before fresh device admission."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,signal,subprocess,time
W=Path('/run-scope');i=json.loads((W/'intent.json').read_bytes());deadline=i['epoch']+120
assert i['epoch']<=time.time()<deadline
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('pre-device link deadline')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
b=json.loads((W/'src/LIBRARY_BINDING.json').read_bytes());sdk=json.loads((W/'src/SDK_BINDING.json').read_bytes());states={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(4*1024**2),b''):
   if time.time()>=deadline:raise TimeoutError('pre-device link deadline')
   h.update(raw)
 return h.hexdigest()
import torch
assert not torch.cuda.is_initialized() and torch.version.hip==sdk['HIP']
def deny(*args,**kwargs):raise RuntimeError('masked link preflight may not create GPU context')
torch.cuda._lazy_init=deny
assert os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']==os.environ['CUDA_VISIBLE_DEVICES']=='-1'
for pin in sdk['provider_files'].values():
 p=Path(pin['path']);assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
r=subprocess.run(['ldd',sdk['provider_files']['extension']['path']],capture_output=True,text=True,timeout=min(15,deadline-time.time()-5));(W/'link-ldd.private.txt').write_text(r.stdout+r.stderr);assert r.returncode==0 and 'not found' not in r.stdout
names={}
for line in r.stdout.splitlines():
 row=line.split()
 if not row or row[0].startswith('linux-vdso'):continue
 if '=>' in row:names[row[0]]=row[2]
 else:names[Path(row[0]).name]=row[0]
with (W/'link-observed-names.json').open('x') as f:json.dump(names,f,indent=2)
assert names==b['resolved_names']
for name,pin in b['resolved_libraries'].items():
 p=Path(name);s=p.stat();assert p.resolve()==p and s.st_size==pin['bytes'] and sha(p)==pin['SHA256']
 after=p.stat();state=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns];assert state==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns];states[name]=state
assert not torch.cuda.is_initialized()
value=dict(passed=True,epoch=i['epoch'],deadline=deadline,CUDA_initialized=False,explicit_GPU_calls=0,library_count=len(states),library_states=states,library_binding_SHA256=sha(W/'src/LIBRARY_BINDING.json'),loader_environment_pin=b['loader_environment_pin'],loader_environment_applied_before_interpreter_start=True,ldd_child_effects_observed=False,passive_ldd_SHA256=sha(W/'link-ldd.private.txt'),observed_names_SHA256=sha(W/'link-observed-names.json'))
with (W/'link-preflight.json').open('x') as f:json.dump(value,f,indent=2)
