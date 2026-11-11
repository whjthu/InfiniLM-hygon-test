"""Install full shared source and run one bounded original-host structural CPU check."""
from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-update-contract-CPU-v2');epoch=time.time();start=time.monotonic();deadline=epoch+300;assert not W.exists();W.mkdir(mode=0o700)
def dump(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]));dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,GPU_cards=0));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original structural CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,290)
result=None
try:
 for name,pin in FILES.items():
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts;raw=base64.b64decode(pin['b64']);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=W/part;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(pin.get('mode',0o600))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CONTRACT_CPU_EPOCH=str(epoch),CONTRACT_CPU_DEADLINE=str(epoch+260))
 affinity=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,affinity)
 with (W/'CPU.private.log').open('xb') as log:r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/run_checks.py')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+270-time.time()))
 p=W/'CPU-result.json';value=json.loads(p.read_bytes()) if p.is_file() else None;result=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,returncode=r.returncode,CPU_result=value,CPU_result_SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,CPU_affinity=affinity,log_SHA256=hashlib.sha256((W/'CPU.private.log').read_bytes()).hexdigest(),passed=bool(r.returncode==0 and value and value.get('passed')),GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
 binding=W/'GEMMA_NATIVE_BINDING.json'
 if binding.is_file():
  raw=binding.read_bytes();assert len(raw)<=2*1024**2;result['binding']=json.loads(raw);result['binding_SHA256']=hashlib.sha256(raw).hexdigest()
except BaseException as error:result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,error_type=type(error).__name__,GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
