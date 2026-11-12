"""Repeat only the failed, side-effect-free original metadata observation."""
from pathlib import Path
import ast,base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent
tree=ast.parse((D/'read_original_once.py').read_bytes())
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
assert hashlib.sha256(remote.encode()).hexdigest()==json.loads((D/'read-intent.json').read_bytes())['payload_SHA256']
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote.encode(),capture_output=True,timeout=60)
with (D/'retry-stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'retry-stderr.private.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 row=json.loads(r.stdout)
 for tag,v in row['files'].items():
  raw=base64.b64decode(v.pop('base64'),validate=True);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['SHA256']
  with (D/(tag+'.original')).open('xb') as f:f.write(raw)
 receipt.update(row)
with (D/'retry-metadata-receipt.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,files=len(receipt.get('files',{})),tensors=receipt.get('tensors'),shards=len(receipt.get('shards',[])),weight_values_read=False,GPU_calls=0)))
