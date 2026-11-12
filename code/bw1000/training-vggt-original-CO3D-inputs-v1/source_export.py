"""One bounded export of the exact original CO3D workload input files."""
from pathlib import Path
import base64,hashlib,json,math,os,signal,socket,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-original-CO3D-input-export-v1');assert not W.exists(),'reconcile original export; never replay';W.mkdir(mode=0o700);start=time.monotonic();assert math.isfinite(EPOCH) and DEADLINE-EPOCH==600 and EPOCH<=time.time()<DEADLINE
raw=base64.b64decode(MANIFEST_B64,validate=True);assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA256;mf=json.loads(raw);assert len(mf['files'])==45 and sum(p['bytes'] for p in mf['files'])==4914255
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def expired(*_):raise TimeoutError('original source export epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,min(120,DEADLINE-time.time()))
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,manifest_SHA256=MANIFEST_SHA256,planned_files=45,planned_bytes=4914255,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])),GPU_cards=0,replay_allowed=False))
baseline=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-original-inputs-CPU-v1/DATA_FILES.private.json');assert hashlib.sha256(baseline.read_bytes()).hexdigest()==mf['source_data_manifest_SHA256'];roots=[Path(mf['source_config'][k]) for k in ['dataset_root','annotation_root']];body={};completed=[]
try:
 for pin in mf['files']:
  if time.monotonic()-start>=120 or time.time()>=DEADLINE:raise TimeoutError('original source export deadline exhausted')
  p=Path(pin['source']);relative=[p.is_relative_to(root) for root in roots];assert any(relative) and p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes']<=32*1024**2 and pin['name'] not in body
  before=p.stat();value=p.read_bytes();after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and len(value)==pin['bytes'] and hashlib.sha256(value).hexdigest()==pin['SHA256'];body[pin['name']]=base64.b64encode(value).decode();completed.append(dict(name=pin['name'],bytes=len(value),SHA256=pin['SHA256']))
 payload=(json.dumps(dict(files=body,manifest_SHA256=MANIFEST_SHA256,request_id=REQUEST),separators=(',',':'))+'\n').encode();assert len(payload)<8*1024**2
 with (W/'payload.private.json').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 receipt=dict(passed=True,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,manifest_SHA256=MANIFEST_SHA256,completed_files=len(completed),completed_bytes=sum(v['bytes'] for v in completed),original_payload_SHA256=hashlib.sha256(payload).hexdigest(),original_payload_bytes=len(payload),wall_seconds=time.monotonic()-start,GPU_cards=0,model_weight_bytes=0);dump('outcome.json',receipt);print(json.dumps(dict(receipt=receipt,source_outcome_SHA256=hashlib.sha256((W/'outcome.json').read_bytes()).hexdigest(),files=body)))
except BaseException as error:dump('failure.json',dict(claim=CLAIM,request_id=REQUEST,error_type=type(error).__name__,completed_files=completed,planned_files=45,planned_bytes=4914255));raise
