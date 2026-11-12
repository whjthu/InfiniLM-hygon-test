"""One original-source HOST CPU child, finite and exclusive; no GPU/service."""
from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hymotion-transformer-meta-CPU-v1');epoch=time.time();anchor=time.monotonic();deadline=epoch+240
def left(reserve=0):
 value=min(deadline-time.time(),240-(time.monotonic()-anchor))-reserve
 if value<=0:raise TimeoutError('original CPU scope exhausted')
 return value
assert not W.exists();W.mkdir(mode=0o700);(W/'source').mkdir()
for n,pin in FILES.items():
 p=Path(n);assert not p.is_absolute() and '..' not in p.parts;raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];dest=W/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(raw)
with (W/'intent.json').open('x') as f:json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,work_deadline=epoch+180,host=socket.gethostname(),source_files={n:p['SHA256'] for n,p in FILES.items()},GPU_cards=0),f,indent=2)
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline)),flush=True)
env=os.environ.copy();env.update(META_CPU_EPOCH=str(epoch),META_CPU_DEADLINE=str(epoch+180),PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(W/'private-pycache'),PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');forced=False;error=None;child=None
try:
 timeout=epoch+190-time.time();assert timeout>10
 with (W/'CPU.log').open('xb') as log:
  child=subprocess.Popen(['/usr/bin/timeout','--kill-after=5s',str(timeout),'/home/whj/nvidia-baseline/env/bin/python','-B','-u',str(W/'source/run_checks.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();owner=dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname());(W/'owned-child.json').write_text(json.dumps(owner,indent=2));child.wait(timeout=left(40))
except BaseException as e:forced=True;error=type(e).__name__
finally:
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=min(5,left(20)))
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=min(5,left(10)))
out=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,returncode=child.returncode if child else None,forced_cleanup=forced,error_type=error,owned_child_reaped=bool(child and child.poll() is not None),wall_seconds=time.monotonic()-anchor,GPU_cards=0)
p=W/'CPU-result.json'
if p.is_file():
 raw=p.read_bytes();out['CPU-result.json']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out['CPU_log_SHA256']=hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest();out['passed']=bool(child and child.returncode==0 and not forced and out.get('CPU-result.json',{}).get('value',{}).get('passed'))
with (W/'outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
