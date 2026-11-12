"""Read exact originalA100-27 GPU identities and prelaunch facts, no compute."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import datetime,importlib.metadata as md,json,shutil,socket,subprocess,sys
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
cards=[[v.strip() for v in row.split(',')] for row in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
assert len(cards)==8 and {r[0] for r in cards}=={str(i) for i in range(8)} and all(r[3]=='NVIDIA A100-SXM4-40GB' for r in cards)
assert len({r[1] for r in cards})==len({r[2] for r in cards})==8
try:
 with socket.socket() as s:s.bind(('127.0.0.1',29901))
 port=True
except OSError:port=False
namespace=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1')
assert 'torch' not in sys.modules
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_cards=8,cards=cards,compute_apps=apps,
 UUID_by_index={r[0]:r[1] for r in cards},PCI_by_index={r[0]:r[2] for r in cards},all_idle=not apps and all(int(r[5])<=16 and int(r[6])==0 for r in cards),
 planned_master_port=29901,master_port_bindable=port,new_namespace_exists=namespace.exists(),
 free_disk_bytes=shutil.disk_usage(namespace.parent).free,versions={n:md.version(n) for n in ('torch','torchvision','transformers','ms-swift','peft','accelerate','Pillow','tokenizers')},
 Torch_imported=False,GPU_compute_launched=False,nvidia_smi_read_only_queries=True,fresh_admission_required_again_at_actual_launch=True)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('gpu-binding.stdout.private.json',r.stdout),('gpu-binding.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'gpu-binding-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,observed_UTC=v.get('observed_UTC'),GPU_cards=v.get('GPU_cards'),all_idle=v.get('all_idle'),compute_apps=len(v.get('compute_apps',[])),new_namespace_exists=v.get('new_namespace_exists'),port_bindable=v.get('master_port_bindable'),free_disk_bytes=v.get('free_disk_bytes'),GPU_compute_launched=False)))
