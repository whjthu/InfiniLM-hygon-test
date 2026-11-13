"""One published GPU-only HunyuanVideo scope with bounded owned-process cleanup."""
from pathlib import Path
import base64,hashlib,json,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1')
epoch=time.time();anchor=time.monotonic();deadline=epoch+3600
cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
selected=[[v.strip() for v in row.split(',')] for row in cards if GPU_UUID in row]
assert len(selected)==1 and selected[0][1]==GPU_UUID and selected[0][2]==GPU_PCI and int(selected[0][4])<=16 and int(selected[0][5])==0
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
assert not any(GPU_UUID in row for row in apps)
assert selected[0][0]==GPU_INDEX
import stat
def identity(path):
 s=Path(path).lstat()
 assert stat.S_ISREG(s.st_mode)
 return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
bound=json.loads(base64.b64decode(FILES['src/BINDING.json']['b64'],validate=True))
assert bound['gpu_index']==GPU_INDEX
assert all(identity(p)==v for group in ('files','sources') for p,v in bound[group].items())
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
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,GPU_cards=1,replay_allowed=False))
child=None;owned=None;result=None;reason=None;peak=0
try:
 for name,pin in FILES.items():
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts
  raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  p=W/part;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
  p.chmod(0o600)
 print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',request_id=REQUEST,claim=CLAIM,owner=host_owner,epoch=epoch,deadline=deadline)),flush=True)
 import gzip,io
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==238 and fm['commit']=='a3afab4faaa10a0bc908a25fb88c6abd20f45224'
 with gzip.GzipFile(fileobj=io.BytesIO((W/'src/FRAMEWORK_BUNDLE.json.gz').read_bytes())) as z:raw=z.read(16*1024**2+1)
 assert len(raw)<=16*1024**2 and hashlib.sha256(raw).hexdigest()==fm['bundle_uncompressed_SHA256']
 bundle=json.loads(raw);assert set(bundle)==set(fm['files'])
 for name,pin in fm['files'].items():
  part=Path(name);assert not part.is_absolute() and '..' not in part.parts
  raw=base64.b64decode(bundle[name],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  p=W/'framework'/part;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
  p.chmod(pin['mode'])
 env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES=GPU_INDEX,CUDA_DEVICE_ORDER='PCI_BUS_ID',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
  PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CONTRACT_GPU_EPOCH=str(epoch),CONTRACT_GPU_DEADLINE=str(epoch+3500))
 affinity=sorted(os.sched_getaffinity(0))[:4];assert len(affinity)==4;os.sched_setaffinity(0,affinity)
 with (W/'GPU.private.log').open('xb') as log:
  assert time.time()<epoch+3490
  child=subprocess.Popen(['/home/whj/nvidia-baseline/env-hunyuan/bin/python','-B','-u',str(W/'src/run_checks.py')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owned=process(child.pid);assert owned and owned['PGID']==child.pid;dump('GPU-child-owner.json',owned)
  while child.poll() is None:
   peak=max(peak,rss(members(owned['PGID'])))
   if time.time()>=epoch+3570 or time.monotonic()-anchor>=3570:reason='GPU_DEADLINE';break
   if peak>96*1024**3:reason='GPU_RSS_LIMIT';break
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
 p=W/'GPU-result.json';value=json.loads(p.read_bytes()) if p.is_file() else None
 result=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=host_owner,child=owned,
  returncode=child.returncode if child else None,stop_reason=reason,owned_group_reaped=clean,
  GPU_result=value,GPU_result_SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
  GPU_affinity=sorted(os.sched_getaffinity(0)),peak_group_RSS_bytes=peak,
  passed=bool(child and child.returncode==0 and value and value.get('passed') and clean and reason is None),
  GPU_cards=1,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-anchor)
 try:
  post_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
  post_clear=not any(line.split(',')[0].strip()==GPU_UUID for line in post_apps.splitlines())
 except (OSError,subprocess.SubprocessError):post_clear=None
 result['GPU_post_no_compute_process']=post_clear
 result['passed']=bool(result['passed'] and post_clear is True)
 for name in ('FUSION_IDENTITY.json',):
  p=W/name
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<=2*1024**2
   result[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 dump('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
