from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-forward-provider-counterfactual-v1');assert not W.exists();W.mkdir(mode=0o700);(W/'src').mkdir()
epoch=time.time();deadline=epoch+1200
pins={}
for n,row in FILES.items():
 part=Path(n);assert not part.is_absolute() and '..' not in part.parts;raw=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256'];p=W/'src'/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
 pins[n]=row['SHA256']
with (W/'installed.json').open('x') as f:json.dump(dict(claim=CLAIM,files=pins),f,indent=2)
with (W/'intent.json').open('x') as f:json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),GPU_cards=8,optimizer_updates=0,raw_tensor_transfer_bytes=0),f,indent=2)
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline)),flush=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');forced=False
with (W/'supervisor.log').open('xb') as log:
 p=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/run_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:p.wait(timeout=max(.1,deadline-time.time()+20))
 except subprocess.TimeoutExpired:forced=True
out=dict(claim=CLAIM,request_id=REQUEST,supervisor_rc=p.returncode,supervisor_observation_timeout=forced,remote_outcome='UNKNOWN' if forced else 'SUPERVISOR_RETURNED',raw_tensor_transfer_bytes=0,Model_GO=False)
for name in ('prelaunch-result.json','outcome.json'):
 if (W/name).exists():
  raw=(W/name).read_bytes();out[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
if p.returncode and not forced:out['supervisor_error_tail']=(W/'supervisor.log').read_text(errors='replace')[-1400:]
with (W/'outer-result.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
