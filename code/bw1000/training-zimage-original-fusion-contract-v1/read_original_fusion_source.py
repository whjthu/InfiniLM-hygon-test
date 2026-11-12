"""Follow exact imports observed in original training and pipeline source."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
old=json.loads((D/'original-metadata.private.json').read_bytes())
names=['diffsynth/diffusion/__init__.py','diffsynth/diffusion/base_pipeline.py','diffsynth/utils/lora.py']
remote=('ROOT='+repr(old['source_root'])+'\nNAMES='+repr(names)+'\n').encode()+br'''
from pathlib import Path
import hashlib,json,sys,time
start=time.monotonic();sources={};total=0
for n in NAMES:
 assert time.monotonic()-start<20
 p=Path(ROOT)/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*1024**2
 b=p.read_bytes();total+=len(b);assert total<4*1024**2
 sources[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),utf8=b.decode('utf8'))
assert 'torch' not in sys.modules
print(json.dumps(dict(sources=sources,source_bytes_read=total,tensor_body_bytes_read=0,Torch_imported=False,GPU_initialized=False)))
'''
with (D/'fusion-source-read-intent.json').open('x') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),source_files=names,read_only=True),f,indent=2)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
start=time.monotonic();r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
for n,b in [('original-fusion-source.private.json',r.stdout),('fusion-source-stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(b)
receipt=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_Slurm_steps=0)
if r.returncode==0:
 v=json.loads(r.stdout)
 for n,s in v['sources'].items():
  p=D/'original-source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(s['utf8'].encode())
 receipt.update(source_files=len(v['sources']),source_bytes_read=v['source_bytes_read'],Torch_imported=False,GPU_initialized=False)
with (D/'fusion-source-read-result.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
