from pathlib import Path
import json,subprocess,hashlib
base=Path(__file__).resolve().parent
a=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
remote=r'''
import json,os,shutil,socket,subprocess
from pathlib import Path
import datetime
home=Path('/home/whj/nvidia-baseline')
def query(*args):
    return [list(map(str.strip,r.split(','))) for r in subprocess.check_output(
        ['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=10).splitlines() if r.strip()]
cards=query('--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu')
apps=query('--query-compute-apps=gpu_uuid,pid,used_memory')
runtime={p.name:p.joinpath('bin/python').is_file() for p in home.glob('env-*') if 'hunyuan' in p.name.lower()}
mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
print(json.dumps(dict(host=socket.gethostname(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cards=cards,apps=apps,
    env_candidates=runtime,available_memory_bytes=mem,free_disk=shutil.disk_usage(home).free,
    namespace_exists=(home/'alignment-assets/hygon-hunyuan-sft-first-update-v1').exists())))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(a/'ssh_config'),'a100-26','timeout 30 python3 -B -u -'],input=remote.encode(),capture_output=True,timeout=45)
v=dict(rc=r.returncode,stderr_sha=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:v['data']=json.loads(r.stdout)
with (base/'hunyuan-first-update-a100-admission-v1.json').open('x') as f:json.dump(v,f,indent=2)
d=v.get('data',{})
print(json.dumps(dict(rc=r.returncode,cards=[[x[0],*x[3:]] for x in d.get('cards',[])],active_compute=len(d.get('apps',[])),environments=d.get('env_candidates'),free_memory=d.get('available_memory_bytes'),free_disk=d.get('free_disk'),namespace_exists=d.get('namespace_exists'))))
