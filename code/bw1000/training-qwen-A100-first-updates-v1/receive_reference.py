"""Stage this original immutable Qwen reference and call its one supervisor."""
import base64,hashlib,json,os,socket,subprocess,time
from pathlib import Path
epoch=time.time();anchor=time.monotonic()
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-A100-first-updates-v1');assert not ROOT.exists(),'Reconcile exact original namespace,never replay';ROOT.mkdir(mode=0o700);src=ROOT/'src';src.mkdir(mode=0o700);pins={}
with (ROOT/'install-intent.json').open('x') as f:json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+900,replay_allowed=False,GPU_cards=8),f,indent=2)
for name,item in FILES.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts;raw=base64.b64decode(item['b64']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['SHA256'];(src/name).parent.mkdir(parents=True,exist_ok=True);(src/name).write_bytes(raw);pins[name]=item['SHA256']
fm=json.loads((src/'FRAMEWORK_FILES.json').read_bytes());(ROOT/'framework').mkdir(mode=0o700)
for name,pin in fm['files'].items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts;p=src/'deltas'/part if name in fm['deltas'] else Path(fm['parent_directory'])/part;raw=p.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];dest=ROOT/'framework'/part;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);dest.chmod(pin['mode'])
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]));installed=dict(claim=CLAIM,request_id=REQUEST,files=pins,owner=owner,epoch=epoch,deadline=epoch+900)
with (ROOT/'installed.json').open('x') as f:json.dump(installed,f,indent=2)
print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',claim=CLAIM,request_id=REQUEST,owner=owner)),flush=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GPU_SCOPE_EPOCH=str(epoch),GPU_SCOPE_MONOTONIC_ANCHOR=str(anchor))
try:
 with (ROOT/'supervisor.private.log').open('xb') as log:r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(src/'run_reference_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+895-time.time()))
 value=dict(supervisor_rc=r.returncode,claim=CLAIM,request_id=REQUEST)
except subprocess.TimeoutExpired:value=dict(supervisor_rc=None,claim=CLAIM,request_id=REQUEST,state='ORIGINAL_SUPERVISOR_OBSERVATION_TIMEOUT_REMOTE_OUTCOME_UNKNOWN')
for name in ['supervisor-intent.json','prelaunch-result.json','gpu-intent.json','outcome.json']:
 if (ROOT/name).is_file():
  raw=(ROOT/name).read_bytes();value[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
with (ROOT/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
