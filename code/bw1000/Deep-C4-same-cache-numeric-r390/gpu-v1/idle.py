"""Read-only process/KFD probe, no GPU libraries."""
import pathlib,json,os,socket
k=pathlib.Path('/sys/class/kfd/kfd/proc');pids=[int(x.name)for x in k.iterdir()if x.name.isdigit()]if k.exists()else[]
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(a.endswith('/incoming/Deep-C4-same-cache-numeric-r390/gpu-v1/numeric_driver.py')for a in args):processes.append(int(p.name))
 except (OSError,PermissionError):pass
print(json.dumps(dict(node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_available=k.exists(),kfd_pids=pids,matching_component_processes=processes,GPU_initialized=False)))
