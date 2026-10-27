import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for key,n in [('watch','GLM-M15-once-r123-watch.json'),('launch','GLM-M15-once-r123-launch.json')]:
 p=r/n;out[key+'_exists']=p.exists()
 if p.exists():out[key]=json.loads(p.read_bytes())
h=out.get('watch',{}).get('actual_handle')
if h:
 p=r/'results'/('GLM-M15-component-r121b-'+h+'.json');out['result_exists']=p.exists()
 if p.exists():out['result']=json.loads(p.read_bytes());out['result_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
rows=subprocess.check_output(['sacct','-S','2026-10-26','-j','910376,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[x.split('|')for x in rows if 'GLM-M15-once'in x]
print(json.dumps(out))
