import subprocess,json,os,datetime
out={'observedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parents':{}};env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM','PMI','CUDA','HIP','ROCR'))}
for parent in ['910379','910375','910381','910371']:
 z=subprocess.run(['scontrol','show','job',parent,'-o'],env=env,text=True,capture_output=True,timeout=25);fields={a:b for a,b in (s.split('=',1)for s in z.stdout.split()if'='in s)};out['parents'][parent]={k:fields.get(k)for k in ['JobId','JobState','UserId','StartTime','EndTime','TimeLimit','NodeList','NumNodes','NumCPUs','TRES','AllocTRES']};nodes=fields.get('NodeList')
 if nodes:out['parents'][parent]['actual_nodes']=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=25).splitlines()
print(json.dumps(out))
