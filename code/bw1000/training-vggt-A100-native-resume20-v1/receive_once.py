"""Install once and preserve one original epoch through supervisor and native caller."""
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-A100-native-resume20-v1');assert not W.exists(),'Reconcile original namespace, never replay';epoch=time.time();W.mkdir(mode=0o700);S=W/'src';S.mkdir(mode=0o700);pins={}
for name,pin in FILES.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts;raw=base64.b64decode(pin['b64']);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];(S/p).parent.mkdir(parents=True,exist_ok=True);(S/p).write_bytes(raw);pins[name]=pin['SHA256']
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
intent=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+900,owner=owner,files=pins,replay_allowed=False)
with (W/'intent.json').open('x') as f:json.dump(intent,f,indent=2)
print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',claim=CLAIM,request_id=REQUEST,owner=owner,epoch=epoch,deadline=epoch+900)),flush=True)
with (W/'supervisor.private.log').open('xb') as log:
 try:r=subprocess.run(['/home/whj/nvidia-baseline/env/bin/python','-B','-u',str(S/'supervise_once.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+900-time.time()));rc=r.returncode;state='ORIGINAL_RETURNED'
 except subprocess.TimeoutExpired:rc=None;state='ORIGINAL_REMOTE_OUTCOME_UNKNOWN'
value=dict(claim=CLAIM,request_id=REQUEST,supervisor_rc=rc,state=state)
for name in ['supervisor-intent.json','prelaunch-result.json','gpu-intent.json','outcome.json']:
 p=W/name
 if p.is_file():raw=p.read_bytes();value[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
with (W/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
