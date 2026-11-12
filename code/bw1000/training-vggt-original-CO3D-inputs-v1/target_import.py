"""Write only exact claimed dataset bytes into one fresh private target namespace."""
from pathlib import Path
import base64,hashlib,json,math,os,shutil,signal,socket,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vggt-original-CO3D-inputs-v1');assert not W.exists(),'reconcile original target import; never replay';assert math.isfinite(EPOCH) and DEADLINE-EPOCH==600 and EPOCH<=time.time()<DEADLINE and shutil.disk_usage(W.parent).free>=64*1024**2;W.mkdir(mode=0o700);start=time.monotonic()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def expired(*_):raise TimeoutError('original target import epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,min(180,DEADLINE-time.time()));raw=base64.b64decode(MANIFEST_B64,validate=True);assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA256;mf=json.loads(raw);assert len(mf['files'])==45 and sum(v['bytes'] for v in mf['files'])==4914255 and set(BODIES)=={v['name'] for v in mf['files']};assert SOURCE_RECEIPT['claim']==CLAIM and SOURCE_RECEIPT['request_id']==REQUEST and SOURCE_RECEIPT['manifest_SHA256']==MANIFEST_SHA256
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,manifest_SHA256=MANIFEST_SHA256,source_receipt=SOURCE_RECEIPT,planned_files=45,planned_bytes=4914255,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])),GPU_cards=0,replay_allowed=False));(W/'TRANSFER_MANIFEST.private.json').write_bytes(raw);completed=[]
try:
 for pin in mf['files']:
  if time.time()>=DEADLINE:raise TimeoutError('original target import deadline exhausted')
  part=Path(pin['name']);assert not part.is_absolute() and '..' not in part.parts and part.parts[0] in ['co3d-single','co3d-vggt-annotations'];value=base64.b64decode(BODIES[pin['name']],validate=True);assert len(value)==pin['bytes'] and hashlib.sha256(value).hexdigest()==pin['SHA256'];p=W/'inputs'/part;p.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
  with p.with_suffix(p.suffix+'.partial').open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
  partial=p.with_suffix(p.suffix+'.partial');assert hashlib.sha256(partial.read_bytes()).hexdigest()==pin['SHA256'];os.link(partial,p);partial.unlink();completed.append(dict(name=pin['name'],bytes=len(value),SHA256=pin['SHA256']))
 receipt=dict(passed=True,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,manifest_SHA256=MANIFEST_SHA256,completed_files=45,completed_bytes=4914255,target_file_manifest_SHA256=hashlib.sha256(json.dumps(completed,sort_keys=True,separators=(',',':')).encode()).hexdigest(),files=completed,source_receipt=SOURCE_RECEIPT,wall_seconds=time.monotonic()-start,GPU_cards=0,model_weight_bytes=0,actual_preprocessing_executed=False,full_CO3D_or_heldout_ready=False);dump('outcome.json',receipt);print(json.dumps(dict(receipt=receipt,target_outcome_SHA256=hashlib.sha256((W/'outcome.json').read_bytes()).hexdigest())))
except BaseException as error:dump('failure.json',dict(claim=CLAIM,request_id=REQUEST,error_type=type(error).__name__,completed_files=completed,planned_files=45,planned_bytes=4914255));raise
