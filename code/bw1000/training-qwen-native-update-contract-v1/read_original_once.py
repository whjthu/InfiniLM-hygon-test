"""Read exact original Qwen configuration/source metadata; no model imports."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess

D = Path(__file__).resolve().parent
A = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
REMOTE = r'''
from pathlib import Path
import ast,base64,datetime,hashlib,json,sysconfig
root=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/Qwen3-VL-8B-Instruct')
site=Path(sysconfig.get_paths()['purelib'])
files={}
for tag,p,limit in (
 ('config',root/'config.json',100000),
 ('index',root/'model.safetensors.index.json',5000000),
 ('model_source',site/'transformers/models/qwen3_vl/modeling_qwen3_vl.py',300000),
 ('configuration_source',site/'transformers/models/qwen3_vl/configuration_qwen3_vl.py',100000),
 ('swift_qwen_source',site/'swift/model/models/qwen.py',500000)):
 raw=p.read_bytes();assert 0<len(raw)<=limit
 files[tag]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
index=json.loads(base64.b64decode(files['index']['base64']))
shards=[]
for name in sorted(set(index['weight_map'].values())):
 assert Path(name).name==name and name.endswith('.safetensors')
 p=root/name;s=p.stat();shards.append(dict(name=name,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,shards=shards,
 tensors=len(index['weight_map']),weight_values_read=False,Torch_imported=False,GPU_calls=0)))
'''
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26',
     '/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
with (D/'read-intent.json').open('x',encoding='utf8') as f:
 json.dump(dict(scope='original Qwen exact metadata and three installed source files; read only',
                payload_SHA256=hashlib.sha256(REMOTE.encode()).hexdigest(),weight_values_read=False),f,indent=2)
r=subprocess.run(ssh,input=REMOTE.encode(),capture_output=True,timeout=60)
(D/'original-stdout.private.json').write_bytes(r.stdout)
(D/'original-stderr.private.txt').write_bytes(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 row=json.loads(r.stdout)
 for tag,v in row['files'].items():
  raw=base64.b64decode(v.pop('base64'),validate=True)
  assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['SHA256']
  (D/(tag+'.original')).write_bytes(raw)
 receipt.update(row)
(D/'original-metadata-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(transport_rc=r.returncode,files=len(receipt.get('files',{})),tensors=receipt.get('tensors'),
                     shards=len(receipt.get('shards',[])),weight_values_read=False,GPU_calls=0)))
