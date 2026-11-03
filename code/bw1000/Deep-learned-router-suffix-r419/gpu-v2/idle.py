"""Read-only process/KFD probe, no GPU libraries."""
import pathlib,json,os,socket
k=pathlib.Path('/sys/class/kfd/kfd/proc');pids=[int(x.name)for x in k.iterdir()if x.name.isdigit()]if k.exists()else[]
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(a.endswith('/incoming/Deep-learned-router-suffix-r419/gpu-v2/GPU_driver.py')for a in args):processes.append(int(p.name))
 except (OSError,PermissionError):pass
import socket
with socket.socket()as probe:port_open=probe.connect_ex(('127.0.0.1',43882))==0
print(json.dumps(dict(node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_available=k.exists(),kfd_pids=pids,matching_component_processes=processes,GPU_initialized=False,port43882_listening=port_open)))
