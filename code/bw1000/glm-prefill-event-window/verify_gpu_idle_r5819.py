import pathlib,json,socket,datetime,os
p=pathlib.Path('/sys/class/kfd/kfd/proc');d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'kfd_proc_exists':p.exists(),'kfd_pids':sorted(x.name for x in p.iterdir()if x.name.isdigit())if p.exists()else None,'CUDA_not_initialized_by_probe':True}
assert d['kfd_proc_exists'] and d['kfd_pids']==[],d
print(json.dumps(d))
