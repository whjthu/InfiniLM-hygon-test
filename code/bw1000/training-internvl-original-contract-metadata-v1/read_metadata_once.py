"""Read bounded original InternVL config and implementation sources, no Torch."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import base64,datetime,hashlib,importlib.metadata as md,json,sys
root=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/InternVL3_5-8B-HF')
assert root.is_dir();names=sorted(p.name for p in root.iterdir() if p.is_file());assert 0<len(names)<=40
selected={'config.json':root/'config.json','index.json':root/'model.safetensors.index.json'}
for package,relative in [('transformers','transformers/models/internvl/modeling_internvl.py'),('transformers','transformers/models/internvl/configuration_internvl.py'),('ms-swift','swift/model/models/internvl.py'),('ms-swift','swift/model/model_arch.py'),('ms-swift','swift/trainers/mixin.py'),('ms-swift','swift/pipelines/train/tuner.py')]:
 selected[relative]=Path(md.distribution(package).locate_file(relative))
files={};contents={}
for name,p in selected.items():
 if not p.is_file():files[name]=dict(path=str(p),exists=False);continue
 assert 0<p.stat().st_size<=4*1024**2
 raw=p.read_bytes();files[name]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());contents[name]=base64.b64encode(raw).decode()
inventory={name:dict(bytes=(root/name).stat().st_size,mtime_ns=(root/name).stat().st_mtime_ns) for name in names}
assert 'torch' not in sys.modules
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,contents=contents,model_directory_inventory=inventory,versions={p:md.version(p) for p in ('torch','transformers','ms-swift','peft','accelerate')},Torch_imported=False,GPU_calls=0,weight_values_read=False)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('metadata.stdout.private.json',r.stdout),('metadata.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 v=json.loads(r.stdout);contents=v.pop('contents')
 for name,text in contents.items():
  raw=base64.b64decode(text,validate=True);assert len(raw)==v['files'][name]['bytes'] and hashlib.sha256(raw).hexdigest()==v['files'][name]['SHA256']
  p=D/'original-source'/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 receipt['value']=v
with (D/'metadata-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,files=len(v.get('files',{})),source_files_present=sum('SHA256' in p for p in v.get('files',{}).values()),model_files=len(v.get('model_directory_inventory',{})),versions=v.get('versions'),Torch_imported=False,GPU_calls=0,weight_values_read=False)))
