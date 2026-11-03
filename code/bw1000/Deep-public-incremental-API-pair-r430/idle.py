"""One CPU-only passive KFD/PID/43882 observation; never connects or kills."""
from pathlib import Path
import datetime,json,os,socket,time,re
node=socket.gethostname().split('.')[0];assert node in ['a02r4n01'];started=time.time()
PCI=[];PCI_error=None
try:PCI=sorted({p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')})
except OSError as error:PCI_error=repr(error)
print('IDLE_CPU_IDENTITY '+json.dumps(dict(node=node,CPU_only=True,GPU_initialized=False,physical_PCI=PCI,physical_PCI_error=PCI_error)),flush=True)
assert PCI_error is None and len(PCI)==8 and all(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}[.][0-7]',p)for p in PCI)
limit=float(os.environ.get('DEEP_R430_POST_DEADLINE_UNIX','0'));expected=json.loads(os.environ.get('DEEP_R430_EXPECTED_PIDS','{}')).get(node,[])
while True:
 kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();pids=sorted(int(p.name)for p in kfd.iterdir()if p.name.isdigit());matching=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
  try:args=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any((b'public_deep_native'in a or b'public_deep_text'in a) or b'/Deep-public-incremental-API-pair-r430/client.py'in a for a in args):matching.append(int(proc.name))
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',43882));available=True
  except OSError:available=False
 alive=[p for p in expected if Path('/proc/'+str(p)).exists()]
 if available or pids or matching or alive or time.time()>=min(limit,started+90):break
 time.sleep(.5) # same original cleanup deadline; no connection/restart
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=pids,matching_component_processes=matching,expected_PIDs_still_exist=alive,ports_bindable={'43882':available},TCP_connections_made=0,passive_port_wait_s=time.time()-started,physical_PCI=PCI,physical_identity_scope='CPU-only DRM renderdevice realpath PCI addresses; not Runtime busytime or an independent CUDAindex-to-PCI mapping')
print(json.dumps(q));assert not pids and not matching and not alive and available
