import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
rows=subprocess.check_output(['sacct','-S','2026-10-26','-j','910407,910379,910371,910406','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines()
out['Slurm']=[x.split('|')for x in rows if 'GLM-M15-CPU-r121' in x or x.split('|')[0] in ['910379.474','910371.586','910406.277']]
p=r/'candidates/GLM-M15-component-r121/CPU-build.json';out['CPU_receipt_exists']=p.exists()
if p.exists():out['CPU_receipt']=json.loads(p.read_bytes());out['CPU_receipt_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
hp=r/'results/Kimi-cap8micro1-A-model-r9293-kimi-cap8-micro1-r9293-5efe17cbd6.json'
if hp.exists():
 h=json.loads(hp.read_bytes());out['Kimi_health']={k:h.get(k)for k in ['heartbeat_UTC','completed','passed','audit','post_KFD_passed']};out['Kimi_health']['error_present']=bool(h.get('error'))
print(json.dumps(out))
