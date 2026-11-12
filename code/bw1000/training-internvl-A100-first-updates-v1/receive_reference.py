"""Install one complete public framework bundle and invoke its owned supervisor."""
import base64,gzip,hashlib,io,json,os,socket,subprocess,time
from pathlib import Path
epoch=time.time();anchor=time.monotonic()
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1')
assert not ROOT.exists(),'Reconcile original namespace; never replay'
ROOT.mkdir(mode=0o700);src=ROOT/'src';src.mkdir(mode=0o700);pins={}
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
with (ROOT/'install-intent.json').open('x') as f:json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+900,owner=owner,replay_allowed=False,GPU_cards=8),f,indent=2)
print(json.dumps(dict(stage='REMOTE_INSTALL_INTENT',claim=CLAIM,request_id=REQUEST,owner=owner,epoch=epoch,deadline=epoch+900)),flush=True)
for name,item in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(item['b64'],validate=True);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['SHA256']
 p=src/part;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
 pins[name]=item['SHA256']
fm=json.loads((src/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==349
with gzip.GzipFile(fileobj=io.BytesIO((src/'FRAMEWORK_BUNDLE.json.gz').read_bytes())) as f:bundle_raw=f.read(16*1024**2+1)
assert len(bundle_raw)<=16*1024**2 and hashlib.sha256(bundle_raw).hexdigest()==fm['bundle_uncompressed_SHA256']
bundle=json.loads(bundle_raw);assert set(bundle)==set(fm['files'])
(ROOT/'framework').mkdir(mode=0o700)
for name,pin in fm['files'].items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(bundle[name],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
 p=ROOT/'framework'/part;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
 p.chmod(pin['mode'])
installed=dict(claim=CLAIM,request_id=REQUEST,files=pins,owner=owner,epoch=epoch,deadline=epoch+900)
with (ROOT/'installed.json').open('x') as f:json.dump(installed,f,indent=2)
print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',claim=CLAIM,request_id=REQUEST,owner=owner,epoch=epoch,deadline=epoch+900,framework_files=349)),flush=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GPU_SCOPE_EPOCH=str(epoch),GPU_SCOPE_MONOTONIC_ANCHOR=str(anchor))
try:
 with (ROOT/'supervisor.private.log').open('xb') as log:r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(src/'run_reference_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+895-time.time()))
 value=dict(supervisor_rc=r.returncode,claim=CLAIM,request_id=REQUEST)
except subprocess.TimeoutExpired:value=dict(supervisor_rc=None,claim=CLAIM,request_id=REQUEST,state='ORIGINAL_SUPERVISOR_OBSERVATION_TIMEOUT_REMOTE_OUTCOME_UNKNOWN')
for name in ('supervisor-intent.json','prelaunch-result.json','gpu-intent.json','outcome.json'):
 if (ROOT/name).is_file():
  raw=(ROOT/name).read_bytes();assert len(raw)<=2*1024**2;value[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
with (ROOT/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
