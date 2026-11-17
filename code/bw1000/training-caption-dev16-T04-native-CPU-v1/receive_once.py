"""New fixed16 inference CPU-only scope. Exact metadata-only model view, original owned-process supervisor."""
from pathlib import Path
import base64,hashlib,json,math,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/training-caption-dev16-T04-native-CPU-v1')
epoch=float(EPOCH);anchor=time.monotonic();received_at=time.time();deadline=epoch+300
assert math.isfinite(epoch) and 0<epoch<=received_at<epoch+250
initial=deadline-received_at
def remaining(offset=300):
 return min(epoch+offset-time.time(),initial-(time.monotonic()-anchor)-(300-offset))
def guard(offset=270):
 if remaining(offset)<=0:raise TimeoutError('Original dispatch epoch exhausted')
def alarm(*_):raise TimeoutError('Original dispatch epoch exhausted')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining(270))
affinity=[v for v in sorted(os.sched_getaffinity(0)) if v not in (0,1)][:2];assert len(affinity)==2;os.sched_setaffinity(0,affinity)
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
dump('intent.json',dict(claim_SHA256=CLAIM_SHA256,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,GPU_cards=0,replay_allowed=False))
child=None;owned=None;result=None;reason=None;peak=0
try:
 for name,pin in FILES.items():
  guard()
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts
  raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  p=W/part;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
  p.chmod(0o600)
 metadata=json.loads((W/'src/MODEL_METADATA.json').read_bytes());model_source=Path(metadata['model_root']);view=W/'model-metadata-only';view.mkdir(mode=0o700)
 assert len(metadata['files'])==14
 for name,pin in metadata['files'].items():
  guard()
  assert Path(name).name==name and not name.endswith(('.safetensors','.pt','.pth','.bin','.ckpt'))
  source=model_source/name;assert source.is_file() and not source.is_symlink()
  raw=source.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  with (view/name).open('xb') as stream:stream.write(raw)
 assert {p.name for p in view.iterdir()}==set(metadata['files'])
 manifest_raw=(W/'data/DATA_MANIFEST.json').read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()==DATA_MANIFEST_SHA256
 manifest=json.loads(manifest_raw);assert len(manifest['images'])==16
 expected_data={'DATA_MANIFEST.json','inputs.jsonl'}|{row['name'] for row in manifest['images']}
 assert {p.relative_to(W/'data').as_posix() for p in (W/'data').rglob('*') if p.is_file()}==expected_data
 assert not any(p.is_symlink() for p in (W/'data').rglob('*'))
 print(json.dumps(dict(stage='REMOTE_SOURCE_AND_FIXED16_INSTALLED',request_id=REQUEST,claim=CLAIM,owner=host_owner,epoch=epoch,deadline=deadline,metadata_only_files=14,no_weights_in_view=True)),flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
  PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_MODULES_CACHE=str(W/'modules-cache'),CONTRACT_CPU_EPOCH=str(epoch),CONTRACT_CPU_DEADLINE=str(epoch+260))
 with (W/'CPU.private.log').open('xb') as log:
  guard(250)
  argv=['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'src/check_native.py'),'--source',str(W/'src'),'--data',str(W/'data'),'--model-view',str(view),'--framework','/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/framework','--data-manifest-sha256',DATA_MANIFEST_SHA256,'--epoch',str(epoch),'--deadline',str(epoch+260),'--output',str(W/'cpu')]
  child=subprocess.Popen(argv,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owned=process(child.pid);assert owned and owned['PGID']==child.pid;dump('CPU-child-owner.json',owned)
  while child.poll() is None:
   peak=max(peak,rss(members(owned['PGID'])))
   if remaining(270)<=0:reason='CPU_DEADLINE';break
   if peak>8*1024**3:reason='CPU_RSS_LIMIT';break
   time.sleep(.2)
except BaseException as error:reason='SUPERVISOR_'+type(error).__name__
finally:
 # Reserve the original final30s for owned cleanup/accounting; never reset epoch.
 signal.setitimer(signal.ITIMER_REAL,0)
 guard(298);signal.setitimer(signal.ITIMER_REAL,remaining(298))
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
   until=time.monotonic()+max(0,min(budget,remaining(294)))
   while members(owned['PGID']) and time.monotonic()<until:time.sleep(.1)
  child.wait(timeout=max(.001,min(2,remaining(296))))
 clean=bool(owned and not members(owned['PGID']))
 p=W/'cpu/CPU-result.json';value=json.loads(p.read_bytes()) if p.is_file() else None
 result=dict(claim_SHA256=CLAIM_SHA256,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,child=owned,
  returncode=child.returncode if child else None,stop_reason=reason,owned_group_reaped=clean,
  CPU_result=value,CPU_result_SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
  CPU_affinity=sorted(os.sched_getaffinity(0)),affinity_excludes_0_1=not bool(set(os.sched_getaffinity(0))&{0,1}),peak_group_RSS_bytes=peak,group_RSS_threshold_bytes=8*1024**3,group_RSS_threshold_is_observed_not_hard_commit=True,
  passed=bool(child and child.returncode==0 and value and value.get('passed') and clean and reason is None),
  GPU_cards=0,GPU_GO=False,Model_GO=False,receiver_wall_seconds=time.monotonic()-anchor,wall_seconds=time.time()-epoch,
  received_at=received_at,budget_origin="client_dispatch_epoch",remaining_whole_seconds=remaining())
 for name in ('input-identities-private.json','native-providers.json','progress.json','failure-private.json'):
  p=W/'cpu'/name
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<=2*1024**2
   result[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 guard(298);dump('outcome.json',result);print(json.dumps(result),flush=True)
 signal.setitimer(signal.ITIMER_REAL,0)
if not result['passed']:raise SystemExit(1)
