"""Fresh/post CPU-only exact-node process/KFD/listener probe; never TCP-connect."""
from pathlib import Path
import os,socket,json,time,sys
node=socket.gethostname();assert node in ['a02r4n12','a02r4n13'];ports=[49282]if node=='a02r4n12'else[49283,49181]
k=Path('/sys/class/kfd/kfd/proc');pids=[int(p.name)for p in k.iterdir()if p.name.isdigit()]if k.exists()else None
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(s.endswith('/Deep-PD-disaggregation-r382/execution/bootstrap.py')for s in a):processes.append(int(p.name))
 except OSError:pass
def observe_ports():
 bound={}
 for port in ports:
  s=socket.socket()
  try:s.bind(('',port));bound[str(port)]=True
  except OSError:bound[str(port)]=False
  finally:s.close()
 entries=[]
 for name in ['tcp','tcp6']:
  for line in (Path('/proc/net')/name).read_text().splitlines()[1:]:
   f=line.split()
   if int(f[1].split(':')[-1],16)in ports:entries.append(dict(table=name,local=f[1],peer=f[2],state=f[3],LISTEN=f[3]=='0A',TIME_WAIT=f[3]=='06'))
 return bound,entries
bound,entries=observe_ports();first=dict(ports_bindable=bound,TCP_entries=entries);start=time.monotonic()
if '--wait-port-release'in sys.argv and not pids and not processes and entries and all(e['TIME_WAIT']for e in entries):
 # One original CPU post step, passive kernel TIME_WAIT visibility only. No
 # repeated model/network transfer, no listener connection and no hidden first
 # observation. Live PIDs/KFD/LISTEN never enter this waiting branch.
 while not all(bound.values())and time.monotonic()-start<75:
  time.sleep(1);pids=[int(p.name)for p in k.iterdir()if p.name.isdigit()];bound,entries=observe_ports()
  if pids or any(e['LISTEN']for e in entries):break
print(json.dumps(dict(node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,KFD_available=k.exists(),kfd_pids=pids,owned_process_PIDs=processes,ports_bindable=bound,TCP_entries=entries,first_port_observation=first,passive_port_release_wait_s=time.monotonic()-start)))
