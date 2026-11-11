from pathlib import Path
import hashlib,json,os,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-forward-provider-counterfactual-v1')
raw=(W/'prelaunch-result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==PRELAUNCH_PIN
pre=json.loads(raw);outer=json.loads((W/'outer-result.json').read_bytes());intent=json.loads((W/'intent.json').read_bytes())
assert pre['state']=='PRELAUNCH_OCCUPIED_AFTER_CPU' and not pre['GPU_launched'] and outer['supervisor_rc']==0 and not outer['supervisor_observation_timeout']
assert not (W/'gpu-intent.json').exists() and not (W/'native.log').exists() and not (W/'arms').exists() and not (W/'outcome.json').exists()
assert hashlib.sha256((W/'src/run_once.py').read_bytes()).hexdigest()==CONTROLLER_PIN
assert intent['deadline']==intent['epoch']+1200 and intent['epoch']+1080-time.time()>300
old=Path('/proc',str(intent['PID']),'stat')
if old.exists():assert int(old.read_text().split(') ',1)[1].split()[19])!=intent['startticks']
S=W/'resume-once';S.mkdir(mode=0o700)
with (S/'intent.json').open('x') as f:json.dump(dict(resume_claim=RESUME_CLAIM,original_claim=intent['claim'],original_request_id=intent['request_id'],request_id=REQUEST,original_epoch=intent['epoch'],original_deadline=intent['deadline'],epoch_reset=False,host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),prior_GPU_launched=False),f,indent=2)
print(json.dumps(dict(stage='RESUME_INTENT_CREATED',claim=RESUME_CLAIM,request_id=REQUEST,original_epoch=intent['epoch'],original_deadline=intent['deadline'],epoch_reset=False)),flush=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');unknown=False
with (S/'supervisor.log').open('xb') as log:
 p=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/run_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:p.wait(timeout=max(.1,intent['deadline']-time.time()+20))
 except subprocess.TimeoutExpired:unknown=True
out=dict(claim=RESUME_CLAIM,request_id=REQUEST,supervisor_rc=p.returncode,supervisor_observation_timeout=unknown,original_epoch=intent['epoch'],original_deadline=intent['deadline'],epoch_reset=False,remote_outcome='UNKNOWN' if unknown else 'SUPERVISOR_RETURNED',raw_tensor_transfer_bytes=0,Model_GO=False)
if (W/'outcome.json').exists():
 raw=(W/'outcome.json').read_bytes();out['outcome.json']=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
if p.returncode and not unknown:out['supervisor_error_tail']=(S/'supervisor.log').read_text(errors='replace')[-1400:]
with (S/'result.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
