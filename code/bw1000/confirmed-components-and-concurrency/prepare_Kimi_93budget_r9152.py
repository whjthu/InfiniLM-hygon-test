from pathlib import Path
import json,hashlib,base64
r=Path.cwd();d=r/'Kimi-93layer-budget-private-r9152';d.mkdir(exist_ok=False);a=json.loads((r/'Kimi-actual93-header-r9151.json').read_bytes());s=(r/'preflight_kimi_resident_r2493.py').read_text();s=s.replace("ranges={1:[[0,93]],2:[[0,46],[46,93]],3:[[0,30],[30,62],[62,93]],4:[[0,24],[24,47],[47,71],[71,93]]}","ranges={pp:[[max(0,min(93,95*p//pp-1)),max(0,min(93,95*(p+1)//pp-1))]for p in range(pp)]for pp in [2,4]}");s=s.replace('for tp,pp in [(8,4),(16,2),(32,1),(32,2),(48,1)]:','for tp,pp in [(16,2),(16,4)]:');s=s.replace("'layers':[begin,end],","'layers':[begin,end],'planning_reserve_bytes':6*1024**3,'parameter_plus_reserve_fits':total+6*1024**3<64*1024**3,");s=s.replace("p=r/'results/kimi-resident-r2493-preflight.json'","d.update(served_layers=93,config_SHA256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),index_SHA256=hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),partition_header_SHA256=PARTITION_SHA256,PP_ranges_by_size={str(k):v for k,v in ranges.items()},header_only_not_GPU_capacity=True,model_host_view=str(model),source_commit='5f91ad738583db8f5617aeae53b76523f85d1ec6');assert d['config_SHA256']==CONFIG_SHA256 and d['index_SHA256']==INDEX_SHA256; p=r/'results/Kimi-93layer-static-budget-r9152.json'")
assert "[(16,2),(16,4)]"in s
s='CONFIG_SHA256='+repr(a['config_SHA256'])+'\nINDEX_SHA256='+repr(a['index_SHA256'])+'\nPARTITION_SHA256='+repr(a['source_header_SHA256'])+'\n'+s
(d/'budget_headers_r9152.py').write_bytes(s.encode());a_src=("#!/bin/bash\nset -euo pipefail\npython3 /public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-93layer-budget-r9152/budget_headers_r9152.py\n");(d/'CPU_inside.sh').write_bytes(a_src.encode());files={p.name:base64.b64encode(p.read_bytes()).decode()for p in d.iterdir()if p.is_file()}
remote='''import pathlib,json,base64,hashlib,subprocess,datetime,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/Kimi-93layer-budget-r9152';d.mkdir(exist_ok=True);payload=PAYLOAD
for n,v in payload.items():
 p=d/n;b=base64.b64decode(v)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
record=r/'results/Kimi-93layer-budget-r9152-launch.json';assert not record.exists();args=['srun','--jobid=910406','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n19','--gres=none','--mem=8192M','-t','00:05:00','--job-name=Kimi-93layer-budget-r9152','--output='+str(r/'logs/Kimi-93layer-budget-r9152-%J.log'),'bash',str(d/'CPU_inside.sh')]
with record.open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'GPU':False,'CPU_only':True},f,indent=2)
with(r/'logs/Kimi-93layer-budget-r9152-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'controller_PID':p.pid,'record':str(record),'CPU_only':True,'purpose':'Fresh93layerheaderbudgetactualPP4partition'}))
'''.replace('PAYLOAD',repr(files));(r/'stage_Kimi_93budget_r9152_remote.py').write_bytes(remote.encode());print('SOURCE_READY')
