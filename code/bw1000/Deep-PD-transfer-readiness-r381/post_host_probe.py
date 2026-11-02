"""Read-only two-role cleanup evidence; no connecting to listeners or GPU imports."""
from pathlib import Path
import os,socket,json
k=Path('/sys/class/kfd/kfd/proc');pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(a.endswith('/Deep-PD-transfer-readiness-r381-host-v1/worker.py')for a in args):pids.append(int(p.name))
 except (PermissionError,OSError):pass
entries=[]
for name in ['tcp','tcp6']:
 for row in (Path('/proc/net')/name).read_text().splitlines()[1:]:
  f=row.split()
  if int(f[1].rsplit(':',1)[1],16)==49181 or int(f[2].rsplit(':',1)[1],16)==49181:
   entries.append(dict(table=name,local=f[1],peer=f[2],state=f[3],LISTEN=f[3]=='0A',TIME_WAIT=f[3]=='06',inode=f[9]))
s=socket.socket()
try:s.bind(('',49181));bindable=True
except OSError:bindable=False
finally:s.close()
print(json.dumps(dict(node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,KFD_available=k.exists(),kfd_pids=[int(p.name)for p in k.iterdir()if p.name.isdigit()]if k.exists()else None,matching_worker_PIDs=pids,port49181_bindable=bindable,TCP_entries=entries,interpretation='TIME_WAIT without live listener/PID is reported separately; no TCPconnect was made')))
