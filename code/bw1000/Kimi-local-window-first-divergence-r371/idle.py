"""CPU-only process/KFD/port probe; never connects a sole-accept listener."""
import pathlib,json,socket,os
k=pathlib.Path('/sys/class/kfd/kfd/proc');pids=[int(x.name)for x in k.iterdir()if x.name.isdigit()]if k.exists()else[]
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(a.endswith('/Kimi-local-window-first-divergence-r371/GPU_driver.py')or a.endswith('/Kimi-long-draft-memory-floor-r362b/worker.py')for a in args):rows.append(int(p.name))
 except (OSError,PermissionError):pass
ports={}
for port in [46947,46962]:
 s=socket.socket()
 try:s.bind(('',port));ports[str(port)]=True
 except OSError:ports[str(port)]=False
 finally:s.close()
print(json.dumps(dict(node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_available=k.exists(),kfd_pids=pids,matching_component_processes=rows,ports_bindable=ports,GPU_initialized=False)))
