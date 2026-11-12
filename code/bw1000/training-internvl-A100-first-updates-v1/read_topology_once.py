"""Read originalA100-27 intra-node GPU topology without launching compute."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
import datetime,hashlib,json,subprocess
r=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,timeout=15)
assert len(r.stdout)<=65536 and len(r.stderr)<=65536
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,
 stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),
 topology=r.stdout.decode(errors='replace').splitlines() if r.returncode==0 else None,
 GPU_compute_launched=False,scope='Intra-node originalA100-27 topology only,not Hygon40node connectivity or measured bandwidth.')))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
for name,raw in [('topology.stdout.private.json',r.stdout),('topology.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'topology-result.json').open('x') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,topology_returncode=v.get('returncode'),topology_lines=len(v.get('topology') or []),GPU_compute_launched=False)))
