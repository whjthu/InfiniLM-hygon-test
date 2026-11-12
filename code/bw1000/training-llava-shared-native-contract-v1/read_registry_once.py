"""Read installed native registry/template bytes; no runtime imports."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
remote=b'''
from pathlib import Path
import base64,hashlib,importlib.metadata as md,json,sys
dist=md.distribution('ms-swift');out={}
for name in ('swift/model/models/llava.py','swift/template/templates/llava.py','swift/model/model_arch.py'):
 p=Path(dist.locate_file(name));assert p.is_file() and p.stat().st_size<512*1024
 raw=p.read_bytes();out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode())
assert 'torch' not in sys.modules
print(json.dumps(dict(version=dist.version,files=out,Torch_imported=False,GPU_launched=False)))
'''
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
with (D/'registry-intent.json').open('x') as f:json.dump(dict(read_only=True,payload_SHA256=hashlib.sha256(remote).hexdigest()),f)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90)
for n,raw in [('registry.stdout.private.json',r.stdout),('registry.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 v=json.loads(r.stdout)
 for name,pin in v['files'].items():
  raw=base64.b64decode(pin.pop('b64'),validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=D/'native-source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 out['value']=v
with (D/'registry-result.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,files=len(out.get('value',{}).get('files',{})),Torch_imported=False,GPU_launched=False)))
