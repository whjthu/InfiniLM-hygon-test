"""Read exact native Swift checkpointing-lifecycle sources without executing them."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import base64,datetime,hashlib,json,sysconfig
site=Path(sysconfig.get_paths()['purelib']);files={}
for name in ('swift/pipelines/train/sft.py','swift/trainers/mixin.py','swift/pipelines/train/tuner.py'):
 p=site/name;a=p.stat();raw=p.read_bytes();b=p.stat();assert 0<len(raw)<300000 and (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 files[name]=dict(path=str(p),bytes=len(raw),mtime_ns=a.st_mtime_ns,SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,Torch_imported=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
with (D/'checkpoint-source.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'checkpoint-source.stderr.private.txt').open('xb') as f:f.write(r.stderr)
value=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 row=json.loads(r.stdout);target=D/'checkpoint-source';target.mkdir(exist_ok=False)
 for name,pin in row['files'].items():
  raw=base64.b64decode(pin.pop('base64'),validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 value['value']=row
with (D/'checkpoint-source-result.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,source_files=len(value.get('value',{}).get('files',{})),GPU_calls=0)))
