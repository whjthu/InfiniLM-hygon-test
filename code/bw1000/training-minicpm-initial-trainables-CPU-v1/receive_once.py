import base64,hashlib,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-initial-trainables-CPU-v1');epoch=time.time();deadline=epoch+180
ROOT.mkdir(mode=0o700);pins={}
for name,item in FILES.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 raw=base64.b64decode(item['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['SHA256']
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
 pins[name]=item['SHA256']
with (ROOT/'intent.json').open('x') as f:json.dump({'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'host':socket.gethostname(),'files':pins,'GPU_allocation':0},f)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',SESSION_CPU_EPOCH=str(epoch),SESSION_CPU_DEADLINE=str(deadline))
with (ROOT/'CPU.log').open('xb') as log:
 p=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(ROOT/'run_CPU.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split();owner={'PID':p.pid,'PGID':int(stat[2]),'startticks':int(stat[19]),'host':socket.gethostname()}
 with (ROOT/'owner.json').open('x') as f:json.dump(owner,f)
 forced=False
 try:p.wait(timeout=max(1,deadline-time.time()-15))
 except subprocess.TimeoutExpired:
  forced=True;os.killpg(p.pid,15)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,9);p.wait(timeout=3)
raw=(ROOT/'CPU.log').read_bytes();out={'claim_commit':CLAIM,'returncode':p.returncode,'epoch':epoch,'deadline':deadline,'forced_cleanup':forced,'wall_seconds':time.time()-epoch,'CPU_log_SHA256':hashlib.sha256(raw).hexdigest(),'CPU_log_tail':raw.decode(errors='replace')[-3500:] if p.returncode else None}
if (ROOT/'CPU-result.json').exists():
 data=(ROOT/'CPU-result.json').read_bytes();out['CPU-result.json']={'SHA256':hashlib.sha256(data).hexdigest(),'value':json.loads(data)}
with (ROOT/'outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
