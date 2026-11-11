"""Run one independently owned CPU audit against the actual original terminal."""
from pathlib import Path
import base64,hashlib,json,os,signal,socket,stat,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-A100-reference-root-reader-v1');assert not W.exists();W.mkdir(mode=0o700);(W/'src').mkdir();epoch=time.time();anchor=time.monotonic();deadline=epoch+660
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,GPU_cards=0))
for name,pin in FILES.items():
 assert Path(name).name==name;raw=base64.b64decode(pin['b64']);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];(W/'src'/name).write_bytes(raw)
expected=json.loads((W/'src/ORIGINAL_REFERENCE.json').read_bytes());root=Path(expected['original_root']);raw=(root/'outcome.json').read_bytes();outcome=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==expected['original_outcome_SHA256'] and outcome['claim']==expected['original_claim'] and outcome['request_id']==expected['original_request'] and outcome['native_returncode']==0 and outcome['keeper_passed']
fm=json.loads((W/'src/original-GPU-FRAMEWORK_FILES.json').read_bytes());assert fm['framework_commit']==expected['framework_commit'] and len(fm['files'])==314
for name,pin in fm['files'].items():
 p=root/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',REFERENCE_READER_EPOCH=str(epoch),REFERENCE_READER_DEADLINE=str(epoch+600));registry=None;child=None;error=None;forced=False
try:
 with (W/'reader.private.log').open('xb') as log:
  child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/read_reference.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);parts=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();registry=dict(PID=child.pid,PGID=int(parts[2]),startticks=int(parts[19]));dump('reader-owner.json',registry);child.wait(timeout=max(.1,min(deadline-time.time(),660-(time.monotonic()-anchor))-20))
except BaseException as e:error=type(e).__name__
finally:
 if child and child.poll() is None:
  forced=True;p=Path('/proc',str(child.pid),'stat')
  if p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==registry['startticks']:os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
result=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,reader_owner=registry,reader_returncode=child.returncode if child else None,error_type=error,forced_cleanup=forced,framework_files_verified=314,GPU_cards=0,raw_tensor_transfer_bytes=0,wall_seconds=time.monotonic()-anchor,log_SHA256=hashlib.sha256((W/'reader.private.log').read_bytes()).hexdigest())
for name in ['independent-reference.json','independent-post.json','failure.json','audit-progress.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<2000000;result[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
result['passed']=bool(child and child.returncode==0 and not forced and error is None and 'independent-reference.json' in result and 'independent-post.json' in result);dump('outcome.json',result);print(json.dumps(result))
