"""CPU-only exact-node KFD/PID/port/physicalPCI pre/post; never signals."""
from pathlib import Path
import os,socket,json,re,time
NODES=['a02r3n15','a02r3n16'];node=socket.gethostname().split('.')[0];assert node in NODES
pci=sorted({p.resolve().name.lower()for p in Path('/sys/class/drm').glob('renderD*/device')});assert len(pci)==8;kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();kp=sorted(int(p.name)for p in kfd.iterdir()if p.name.isdigit());matching=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if any(b'GLM-vocab-parallel-embedding-r471/full-model-entry-v1/owner.py'in x or b'GLM-vocab-parallel-embedding-r471/full-model-entry-v1/worker.py'in x or b'GLM-vocab-parallel-embedding-r471/full-model-entry-v1/control.py'in x for x in a):matching.append(int(p.name))
alive=[];expected=json.loads(os.environ.get('GLM470_EXPECTED_PIDS','{}')).get(node,[])
for a in expected:
 p=Path('/proc/'+str(a['PID']))
 try:
  ticks=(p/'stat').read_text().split()[21]
  if ticks==a['startticks']:alive.append(a)
 except FileNotFoundError:pass
old_inodes={v for a in expected for v in a.get('socket_inodes',[])};uid_listeners=[]
for table in ['/proc/net/tcp','/proc/net/tcp6']:
 for line in Path(table).read_text().splitlines()[1:]:
  fields=line.split()
  if len(fields)>9 and fields[3]=='0A'and fields[9]in old_inodes:uid_listeners.append(dict(table=table,inode=fields[9],local=fields[1]))
ports={}
for port in [43843,43844]:
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',port));ports[str(port)]=True
  except OSError:ports[str(port)]=False
q=dict(node=node,PID=os.getpid(),UID=os.getuid(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],physical_PCI=pci,KFD_available=True,kfd_pids=kp,matching_Model_processes=matching,expected_owned_PIDs_still_live=alive,expected_UIDsocket_listeners_still_live=uid_listeners,ports_bindable=ports,CPU_only=True,CUDA_initialized=False,epoch=time.time());print(json.dumps(q),flush=True);assert not kp and not matching and not alive and not uid_listeners and all(ports.values())
