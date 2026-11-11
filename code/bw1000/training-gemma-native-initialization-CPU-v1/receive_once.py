from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-initialization-CPU-v1');epoch=time.time();start=time.monotonic();deadline=epoch+300
assert not W.exists();W.mkdir(mode=0o700)
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,GPU_cards=0));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('single CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,290)
try:
 for name,pin in FILES.items():
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts;raw=base64.b64decode(pin['b64']);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=W/'src'/part;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 mf=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());base=Path(mf['base_directory']);closure=[]
 for row in mf['files']:
  part=Path(row['name']);assert not part.is_absolute() and '..' not in part.parts
  delta=W/'src/deltas'/part;source=delta if delta.is_file() else base/part;assert source.is_file() and not source.is_symlink();raw=source.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'];target=W/'framework'/part;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(row['mode']);assert target.stat().st_mode&0o777==row['mode'];closure.append(row)
 dump('framework-closure.json',dict(commit=mf['commit'],tree=mf['tree'],files=closure,verified_files=len(closure)))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
 affinity=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,affinity)
 with (W/'CPU.private.log').open('xb') as log:r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/run_checks.py')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+270-time.time()))
 p=W/'CPU-result.json';value=json.loads(p.read_bytes()) if p.is_file() else None
 result=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,returncode=r.returncode,CPU_result=value,CPU_result_SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,framework_files=len(closure),framework_commit=mf['commit'],CPU_affinity=affinity,log_SHA256=hashlib.sha256((W/'CPU.private.log').read_bytes()).hexdigest(),passed=bool(r.returncode==0 and value and value['passed']),GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
except BaseException as error:result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,error_type=type(error).__name__,GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
