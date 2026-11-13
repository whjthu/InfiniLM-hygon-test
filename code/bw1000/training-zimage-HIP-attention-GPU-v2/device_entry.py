"""Bind the actual Slurm device before running the shared observed-shape SDK probe."""
from pathlib import Path
import ctypes as C,hashlib,json,os,re,signal,socket,sys,time
W=Path('/run-scope');i=json.loads((W/'intent.json').read_bytes());deadline=i['epoch']+420
assert i['epoch']<=time.time()<deadline
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original SDK inside epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
from snapshot import drm_inventory
before=json.loads((W/'before.json').read_bytes());gate=json.loads((W/'lease-before.json').read_bytes());step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
assert gate['step']==step and re.fullmatch(r'1087656\.\d+',step) and gate['node']==socket.gethostname().split('.')[0]=='f11r4n13'
assert gate['host_UID']==os.getuid() and gate['kfd_holder_count']==0 and gate['boot_id']==before['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert gate['pci']==before['pci']==drm_inventory() and 0<=time.monotonic_ns()-gate['monotonic_ns']<=45_000_000_000
assert gate['before_SHA256']==hashlib.sha256((W/'before.json').read_bytes()).hexdigest() and gate['link_preflight_SHA256']==hashlib.sha256((W/'link-preflight.json').read_bytes()).hexdigest()
link=json.loads((W/'link-preflight.json').read_bytes());assert link['passed'] and link['epoch']==i['epoch'] and not link['CUDA_initialized']
for name,state in link['library_states'].items():
 s=Path(name).stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==state
import torch
assert not torch.cuda.is_initialized() and torch.version.hip and torch.cuda.device_count()==1
runtime=C.CDLL('libamdhip64.so');count=C.c_int();assert runtime.hipGetDeviceCount(C.byref(count))==0 and count.value==1
pci=C.create_string_buffer(32);assert runtime.hipDeviceGetPCIBusId(pci,len(pci),0)==0;name=pci.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name):name='0000:'+name
assert name in before['pci']
from assigned_kfd import assigned_kfd
assigned=dict(node='f11r4n13',step=step,pci=name,KFD=assigned_kfd(name),visible_cards=1,GPU_name=torch.cuda.get_device_name(0),HIP=torch.version.hip,PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),boot_id=gate['boot_id'],epoch=i['epoch'],deadline=deadline)
with (W/'assigned.json').open('x') as f:json.dump(assigned,f,indent=2)
sys.path.insert(0,str(W/'framework/generation/src'))
from gencuda.zimage_hip_attention_probe import run
binding=json.loads((W/'src/SDK_BINDING.json').read_bytes());assert assigned['GPU_name']==binding['GPU_name']
result=run(W/'diagnostic',binding,epoch=i['epoch'],deadline=deadline)
torch.cuda.synchronize();assert runtime.hipDeviceGetPCIBusId(pci,len(pci),0)==0
after=pci.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',after):after='0000:'+after
assert after==name
if not result['passed']:raise SystemExit(1)
