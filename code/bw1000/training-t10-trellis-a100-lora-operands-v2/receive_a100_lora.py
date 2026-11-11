"""One finite original A100 first-backward capture, with exact owned cleanup."""
from pathlib import Path
import base64,hashlib,json,math,os,signal,socket,subprocess,sys,time
base=Path('/home/whj/nvidia-baseline');root=base/'alignment-assets/hygon-trellis-lora-operands-v2'
epoch=time.time();mono=time.monotonic();deadline=epoch+320
def left(offset=320,reserve=0):
 value=min(epoch+offset-time.time(),offset-(time.monotonic()-mono))-reserve
 if not math.isfinite(value) or value<=0:raise TimeoutError('original finite A100 scope expired')
 return value
def query(columns,kind='gpu'):
 raw=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+columns,'--format=csv,noheader,nounits'],text=True,timeout=min(10,left(320,10)))
 return [[x.strip() for x in line.split(',')] for line in raw.splitlines() if line.strip()]
assert not root.exists() and not (base/'alignment-assets/hygon-trellis-lora-operands-v1').exists()
assert sys.executable=='/usr/bin/python3' and sys.version_info[:2]==(3,12)
cards=query('index,uuid,memory.used');first,=[row for row in cards if row[0]=='0'];assert len(cards)==8 and int(first[2])<100 and not query('gpu_uuid,pid,used_gpu_memory','compute-apps')
root.mkdir(mode=0o700);(root/'src').mkdir(mode=0o700)
for name,row in FILES.items():
 assert Path(name).name==name;raw=base64.b64decode(row['b64'],validate=True);assert len(raw)<100000 and hashlib.sha256(raw).hexdigest()==row['SHA256']
 with (root/'src'/name).open('xb') as f:f.write(raw)
sys.path.insert(0,str(root/'src'));from owned_capture import snapshot,refresh,cleanup
def save(name,value):
 with (root/name).open('x') as f:json.dump(value,f,indent=2)
save('intent.json',dict(claim=CLAIM,epoch=epoch,deadline=deadline,native_deadline=epoch+260,host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),physical_card=0,card_UUID_SHA256=hashlib.sha256(first[1].encode()).hexdigest(),GPU_optimizer_updates=0,source_pins={n:v['SHA256'] for n,v in FILES.items()}))
assert epoch+30>=time.time();left(260,200)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NVIDIA_BASELINE_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',T10_EPOCH=str(epoch),T10_DEADLINE=str(epoch+260))
registry={};child=None;error=None;timed_out=False
try:
 with (root/'capture.log').open('xb') as log:
  child=subprocess.Popen(['/usr/bin/python3','-B','-u',str(root/'src/bounded_capture.py'),str(root/'capture')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row=snapshot()[child.pid];registry[child.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')};save('owned-child.json',registry[child.pid])
  while child.poll() is None:
   refresh(registry,root)
   try:child.wait(timeout=min(.5,left(260)))
   except subprocess.TimeoutExpired:pass
except TimeoutError:timed_out=True
except BaseException as failure:error=dict(type=type(failure).__name__,message=str(failure)[:250])
finally:
 cleaned=cleanup(registry,root,10)
 if child is not None:
  try:child.wait(timeout=min(2,left(320,20)))
  except subprocess.TimeoutExpired:error=error or dict(type='OWNED_CHILD_REMAINS')
post=query('index,uuid,memory.used');actual,=[row for row in post if row[0]=='0'];assert actual[1]==first[1]
apps=query('gpu_uuid,pid,used_gpu_memory','compute-apps');owned_apps=[x for x in apps if int(x[1]) in registry]
raw=(root/'capture.log').read_bytes();assert len(raw)<256000
out=dict(claim=CLAIM,epoch=epoch,deadline=deadline,native_deadline=epoch+260,returncode=child.returncode if child else None,timed_out=timed_out,error=error,cleanup=cleaned,owned_process_registry=registry,owned_GPU_compute_PIDs_remaining=[int(x[1]) for x in owned_apps],post_global_GPU_compute_count=len(apps),post_card_memory_MiB=int(actual[2]),card_UUID_SHA256=hashlib.sha256(actual[1].encode()).hexdigest(),log_SHA256=hashlib.sha256(raw).hexdigest(),wall_seconds=time.monotonic()-mono,GPU_optimizer_updates=0)
p=root/'capture/receipt.json'
if p.is_file():
 data=p.read_bytes();assert len(data)<100000;out['capture_receipt']=dict(bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),value=json.loads(data))
out['passed']=bool(child and child.returncode==0 and not timed_out and error is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and not owned_apps and int(actual[2])<100 and out.get('capture_receipt',{}).get('value',{}).get('original_inputs_initial_state_prediction_loss_gradients_exact'))
if not out['passed']:out['error_lines']=[line[-250:] for line in raw.decode(errors='replace').splitlines() if line.startswith(('AssertionError','RuntimeError','TypeError','KeyError','AttributeError','ModuleNotFoundError','TimeoutError'))][-5:]
save('outcome.json',out);print(json.dumps(out))
