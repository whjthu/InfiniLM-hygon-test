"""Read size and identity metadata for the four published Qwen weights at the original host."""
from pathlib import Path
import hashlib
import json
import subprocess

base = Path(__file__).resolve().parent
repo = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
identity = repo / 'coordination/training/qwen-official-public-original-model-identity-v1.json'
source = repo / 'coordination/training/qwen-original-model-files-v1.json'
official = json.loads(identity.read_bytes())['official_weight_pointers']['files']
original = json.loads(source.read_bytes())
pins = {name: original['files'][name] for name in official}
for name, pin in pins.items():
    assert pin['bytes']==official[name]['weight_bytes'] and pin['SHA256']==official[name]['weight_SHA256']
cfg = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
program = r'''
import datetime,json,stat
from pathlib import Path
base=Path(ROOT)
rows={}
for name,pin in PINS.items():
    p=base/name
    s=p.lstat()
    row=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,
             regular=stat.S_ISREG(s.st_mode))
    row['size_matches']=row['regular'] and s.st_size==pin['bytes']
    companion=p.with_name(name+'.source-sha256')
    row['companion_matches']=None
    if companion.is_file() and not companion.is_symlink() and companion.stat().st_size<=128:
        row['companion_matches']=companion.read_text().strip()==pin['SHA256']
    rows[name]=row
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),weights=rows,
    source_weights_read=0,private_input_bytes=0,GPU_cards=0)))
'''
prefix=f"ROOT={original['model_root']!r}\nPINS={pins!r}\n"
argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(cfg),'a100-26','python3 -B -u -']
r=subprocess.run(argv,input=(prefix+program).encode(),capture_output=True,timeout=40)
v=dict(transport_rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
    v['value']=json.loads(r.stdout)
else:
    v['failure_types']=[s for s in ('Connection closed','Permission denied','FileNotFoundError','timed out') if s.encode() in r.stderr]
with (base/'qwen-public-weights-original-host-stat-v1.json').open('x',encoding='utf8') as f:
    json.dump(v,f,indent=2)
print(json.dumps(v))
