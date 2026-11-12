"""Pin exact original metadata bytes needed by the native HF processor."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import datetime,hashlib,json,sys
root=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/InternVL3_5-8B-HF')
names=['config.json','model.safetensors.index.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','added_tokens.json','merges.txt','vocab.json','chat_template.jinja','video_preprocessor_config.json']
files={}
for name in names:
 p=root/name;a=p.stat();assert not p.is_symlink() and 0<a.st_size<=16*1024**2;raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino,a.st_dev)==(b.st_size,b.st_mtime_ns,b.st_ino,b.st_dev)
 files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
assert 'torch' not in sys.modules
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_root=str(root),files=files,total_metadata_bytes=sum(p['bytes'] for p in files.values()),model_weight_values_read=False,Torch_imported=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('processor-metadata.stdout.private.json',r.stdout),('processor-metadata.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'processor-metadata-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,metadata_files=len(v.get('files',{})),metadata_bytes=v.get('total_metadata_bytes'),model_weight_values_read=False,Torch_imported=False,GPU_calls=0)))
