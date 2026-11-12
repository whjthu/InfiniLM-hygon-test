"""Resolve original Swift InternVL registration from exact package source files."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import base64,hashlib,importlib.metadata as md,json,sys
rows={};contents={}
for package,name in [('ms-swift','swift/model/models/__init__.py'),('ms-swift','swift/model/models/internlm.py'),('ms-swift','swift/template/templates/internvl.py'),('transformers','transformers/models/qwen3/modeling_qwen3.py')]:
 p=Path(md.distribution(package).locate_file(name));assert p.is_file() and 0<p.stat().st_size<=2*1024**2;raw=p.read_bytes()
 rows[name]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());contents[name]=base64.b64encode(raw).decode()
assert 'torch' not in sys.modules
print(json.dumps(dict(files=rows,contents=contents,Torch_imported=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('registry.stdout.private.json',r.stdout),('registry.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
value=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 v=json.loads(r.stdout);contents=v.pop('contents')
 for name,text in contents.items():
  raw=base64.b64decode(text,validate=True);assert hashlib.sha256(raw).hexdigest()==v['files'][name]['SHA256'];p=D/'original-source'/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 value['value']=v
with (D/'registry-result.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,source_files=len(value.get('value',{}).get('files',{})),Torch_imported=False,GPU_calls=0)))
