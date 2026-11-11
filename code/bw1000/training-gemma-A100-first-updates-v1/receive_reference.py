"""Stage this original immutable Gemma reference and call its one supervisor."""
import base64,hashlib,json,os,socket,subprocess
from pathlib import Path
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-A100-first-updates-v1');assert not ROOT.exists(),'Reconcile exact original namespace,never replay';ROOT.mkdir(mode=0o700);src=ROOT/'src';src.mkdir(mode=0o700);pins={}
for name,item in FILES.items():
 assert Path(name).name==name;raw=base64.b64decode(item['b64']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['SHA256'];(src/name).write_bytes(raw);pins[name]=item['SHA256']
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]));installed=dict(claim=CLAIM,request_id=REQUEST,files=pins,owner=owner)
with (ROOT/'installed.json').open('x') as f:json.dump(installed,f,indent=2)
print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',claim=CLAIM,request_id=REQUEST,owner=owner)),flush=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
try:
 with (ROOT/'supervisor.private.log').open('xb') as log:r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(src/'run_reference_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1020)
 value=dict(supervisor_rc=r.returncode,claim=CLAIM,request_id=REQUEST)
except subprocess.TimeoutExpired:value=dict(supervisor_rc=None,claim=CLAIM,request_id=REQUEST,state='ORIGINAL_SUPERVISOR_OBSERVATION_TIMEOUT_REMOTE_OUTCOME_UNKNOWN')
for name in ['supervisor-intent.json','prelaunch-result.json','gpu-intent.json','outcome.json']:
 if (ROOT/name).is_file():
  raw=(ROOT/name).read_bytes();value[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
with (ROOT/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
