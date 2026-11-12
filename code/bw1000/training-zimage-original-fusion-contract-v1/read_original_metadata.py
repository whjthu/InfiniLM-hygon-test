"""Read original DiffSynth fusion source and safetensors headers; no tensor bodies."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

D = Path(__file__).resolve().parent
A = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
launch = A/'cross_platform_reference_r1/raw/node27/node/results/zimage-cuda-sft-baseline-r2-repeat1/launch.json'
raw = launch.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'f0bc54e478ad0e8b5d9e0b7fe632a172e5aa6d2e9d6fa166b6d09d280ecdc89d'
original = json.loads(raw)
remote = ('ORIGINAL='+repr(original)+'\n').encode()+br'''
from pathlib import Path
import base64,datetime,hashlib,json,struct,sys,time
start=time.monotonic();deadline=start+45
def check():
 if time.monotonic()>deadline:raise TimeoutError('bounded original metadata read expired')
def sha(b):return hashlib.sha256(b).hexdigest()
source=Path(ORIGINAL['source']);model=Path(ORIGINAL['weights']);adapter=Path(ORIGINAL['config']['preset_lora_path'])
sources={}
names=['examples/z_image/model_training/train.py','diffsynth/pipelines/z_image.py','diffsynth/models/z_image_dit.py','diffsynth/core/loader/lora.py','diffsynth/trainers/utils.py']
total=0
for name in names:
 check();p=source/name
 if not p.is_file():sources[name]=dict(exists=False);continue
 assert not p.is_symlink() and p.stat().st_size<=2*1024**2
 b=p.read_bytes();total+=len(b);assert total<=8*1024**2
 sources[name]=dict(exists=True,bytes=len(b),SHA256=sha(b),utf8=b.decode('utf8'))
headers={};body_bytes=0;header_bytes=0
files=[model/'transformer'/('diffusion_pytorch_model-%05d-of-00003.safetensors'%i) for i in (1,2,3)]+[adapter]
for p in files:
 check();s=p.stat();assert p.is_file() and not p.is_symlink()
 with p.open('rb') as f:
  prefix=f.read(8);assert len(prefix)==8;length=struct.unpack('<Q',prefix)[0];assert 0<length<=4*1024**2
  b=f.read(length);assert len(b)==length
 header_bytes+=len(b)+8;h=json.loads(b);metadata=h.pop('__metadata__',None)
 assert all(type(v)==dict and set(v)=={'dtype','shape','data_offsets'} for v in h.values())
 headers[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,
   header_bytes=length,header_SHA256=sha(b),metadata=metadata,tensors=h)
check();assert 'torch' not in sys.modules
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 original_launch_SHA256='f0bc54e478ad0e8b5d9e0b7fe632a172e5aa6d2e9d6fa166b6d09d280ecdc89d',
 source_root=str(source),model_root=str(model),preset_adapter=str(adapter),sources=sources,headers=headers,
 source_bytes_read=total,safetensors_header_bytes_read=header_bytes,tensor_body_bytes_read=0,
 Torch_imported=False,GPU_initialized=False,model_constructed=False,weights_loaded=False,
 preset_adapter_fused=False,inside_seconds=time.monotonic()-start)))
'''
intent = dict(source_launch_SHA256=hashlib.sha256(raw).hexdigest(), payload_SHA256=hashlib.sha256(remote).hexdigest(),
              read_only=True, tensor_body_bytes_allowed=0, new_Slurm_steps=0, GPU_initialized=False)
with (D/'read-intent.json').open('x') as f:
    json.dump(intent,f,indent=2)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
start=time.monotonic()
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,b in [('original-metadata.private.json',r.stdout),('original-stderr.private.txt',r.stderr)]:
    with (D/name).open('xb') as f:f.write(b)
receipt=dict(**intent,transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,
             stdout_bytes=len(r.stdout),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
    v=json.loads(r.stdout)
    receipt['summary']={k:v[k] for k in ['source_bytes_read','safetensors_header_bytes_read','tensor_body_bytes_read','Torch_imported','GPU_initialized','preset_adapter_fused','inside_seconds']}
    receipt['summary'].update(source_files_present=sum(s['exists'] for s in v['sources'].values()),header_files=len(v['headers']),tensor_metadata_entries=sum(len(s['tensors']) for s in v['headers'].values()))
with (D/'read-result.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
