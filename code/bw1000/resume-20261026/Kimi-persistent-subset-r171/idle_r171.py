import pathlib,os,json,socket
assert socket.gethostname()=='a02r4n01';p=pathlib.Path('/sys/class/kfd/kfd/proc');assert p.exists();pids=sorted(x.name for x in p.iterdir()if x.name.isdigit());assert not pids,pids
print(json.dumps({'passed':True,'node':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'kfd_pids':pids,'CUDA_initialized':False}))
