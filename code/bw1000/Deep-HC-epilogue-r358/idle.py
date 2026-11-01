"""Read-only KFD probe: no CUDA/HIP/torch imports."""
import pathlib,json,os,socket
p=pathlib.Path('/sys/class/kfd/kfd/proc');pids=[int(x.name)for x in p.iterdir()if x.name.isdigit()]if p.exists()else[]
print(json.dumps(dict(node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_available=p.exists(),kfd_pids=pids,GPU_initialized=False)))
