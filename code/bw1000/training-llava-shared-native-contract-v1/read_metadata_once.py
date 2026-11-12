"""Read exact original OneVision configuration and code, never weights or samples."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
remote=b'''
from pathlib import Path
import base64,datetime,hashlib,json,sys
root=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/LLaVA-OneVision-1.5-8B-Instruct')
inventory=[];files={}
for p in sorted(root.iterdir()):
 if p.is_file():inventory.append(dict(name=p.name,bytes=p.stat().st_size,symlink=p.is_symlink()))
names={'config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','model.safetensors.index.json'}
names.update(p.name for p in root.iterdir() if p.is_file() and p.suffix=='.py')
for name in sorted(names):
 p=root/name
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<=4*1024**2
 raw=p.read_bytes();files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode())
assert 'config.json' in files and 'modeling_llavaonevision1_5.py' in files
assert files['modeling_llavaonevision1_5.py']['SHA256']=='bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198'
assert sum(v['bytes'] for v in files.values())<8*1024**2
assert 'torch' not in sys.modules
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),inventory=inventory,files=files,weight_payload_bytes_read=0,Torch_imported=False,GPU_launched=False)))
'''
with (D/'read-intent.json').open('x') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True,weight_payload_bytes=0),f)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
start=time.monotonic();r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90)
for n,raw in [('read.stdout.private.json',r.stdout),('read.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 value=json.loads(r.stdout);folder=D/'model-metadata';folder.mkdir(exist_ok=False)
 for name,pin in value['files'].items():
  raw=base64.b64decode(pin.pop('b64'),validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];(folder/name).write_bytes(raw)
 out['value']=value
with (D/'metadata-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,metadata_files=len(v.get('files',{})),metadata_bytes=sum(x['bytes'] for x in v.get('files',{}).values()),weight_payload_bytes_read=0,Torch_imported=False,GPU_launched=False)))
