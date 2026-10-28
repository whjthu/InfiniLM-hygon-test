import pathlib,os,json,socket,datetime
p=pathlib.Path('/sys/class/kfd/kfd/proc');assert p.exists();pids=sorted(x.name for x in p.iterdir()if x.name.isdigit());assert not pids,pids
assert socket.gethostname()=='a02r4n01'
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'kfd_pids':pids,'passed':True,'CUDA_initialized':False,'IB_CCL_not_used':True}))
