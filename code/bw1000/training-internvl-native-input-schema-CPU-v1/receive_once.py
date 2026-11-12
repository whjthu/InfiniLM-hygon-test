"""One published CPU-only Qwen scope with bounded owned-process cleanup."""
from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-native-input-schema-CPU-v1')
epoch=time.time();anchor=time.monotonic();deadline=epoch+300
assert not W.exists();W.mkdir(mode=0o700)
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def process(pid):
 try:
  fields=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
  return dict(PID=pid,state=fields[0],PGID=int(fields[2]),startticks=int(fields[19]))
 except (FileNotFoundError,ProcessLookupError):return None
def members(pgid):
 result=[]
 for name in os.listdir('/proc'):
  if name.isdigit():
   row=process(int(name))
   if row and row['PGID']==pgid and row['state']!='Z':result.append(row)
 return result
def rss(rows):
 total=0
 for row in rows:
  try:
   fields=Path('/proc/%d/statm'%row['PID']).read_text().split();total+=int(fields[1])*os.sysconf('SC_PAGE_SIZE')
  except (FileNotFoundError,ProcessLookupError):pass
 return total
host_owner=process(os.getpid());host_owner['host']=socket.gethostname()
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,GPU_cards=0,replay_allowed=False))
child=None;owned=None;result=None;reason=None;peak=0
try:
 for name,pin in FILES.items():
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts
  raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  p=W/part;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
  p.chmod(0o600)
 print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',request_id=REQUEST,claim=CLAIM,owner=host_owner,epoch=epoch,deadline=deadline)),flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
  PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CONTRACT_CPU_EPOCH=str(epoch),CONTRACT_CPU_DEADLINE=str(epoch+260))
 affinity=sorted(os.sched_getaffinity(0))[:4];assert len(affinity)==4;os.sched_setaffinity(0,affinity)
 with (W/'CPU.private.log').open('xb') as log:
  assert time.time()<epoch+250
  child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/run_checks.py')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owned=process(child.pid);assert owned and owned['PGID']==child.pid;dump('CPU-child-owner.json',owned)
  while child.poll() is None:
   peak=max(peak,rss(members(owned['PGID'])))
   if time.time()>=epoch+270 or time.monotonic()-anchor>=270:reason='CPU_DEADLINE';break
   if peak>8*1024**3:reason='CPU_RSS_LIMIT';break
   time.sleep(.2)
except BaseException as error:reason='SUPERVISOR_'+type(error).__name__
finally:
 if child is not None and owned is not None:
  for sig,budget in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
   rows=members(owned['PGID'])
   if not rows:break
   # The process-group leader was created by this supervisor; members cannot
   # authorize a signal to a reused group leader with a different identity.
   leader=process(owned['PID'])
   assert leader is None or leader['startticks']==owned['startticks']
   if child.poll() is None or rows:
    try:os.killpg(owned['PGID'],sig)
    except ProcessLookupError:pass
   until=time.monotonic()+budget
   while members(owned['PGID']) and time.monotonic()<until:time.sleep(.1)
  child.wait(timeout=2)
 clean=bool(owned and not members(owned['PGID']))
 p=W/'CPU-result.json';value=json.loads(p.read_bytes()) if p.is_file() else None
 result=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,child=owned,
  returncode=child.returncode if child else None,stop_reason=reason,owned_group_reaped=clean,
  CPU_result=value,CPU_result_SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
  CPU_affinity=sorted(os.sched_getaffinity(0)),peak_group_RSS_bytes=peak,
  passed=bool(child and child.returncode==0 and value and value.get('passed') and clean and reason is None),
  GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-anchor)
 for name in ('checkpoint-headers.json','native-meta-schema.json','native-checkpoint-key-schema.json','encoded/input-identities.json'):
  p=W/name
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<=2*1024**2
   result[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 dump('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
