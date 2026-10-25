import json,subprocess,os
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_'))};q={}
for name,args in [('sstat',['sstat','-j','910380.407','--noheader','--parsable2','--format=JobID,NTasks,NodeList,AveCPU,AveRSS,MaxRSS,AveDiskRead,MaxDiskRead,AveDiskWrite,MaxDiskWrite']),('sacct',['sacct','-j','910380.407','-n','-P','-o','JobID,JobName%100,State,Elapsed,MaxRSS,MaxDiskRead,MaxDiskWrite,NodeList%100'])]:
 try:p=subprocess.run(args,capture_output=True,text=True,timeout=25,env=env);q[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scope':'OwnStep407 wholelifetime current servingphase; not load-only processIO'}
 except Exception as e:q[name]={'error':repr(e)}
print(json.dumps(q))
