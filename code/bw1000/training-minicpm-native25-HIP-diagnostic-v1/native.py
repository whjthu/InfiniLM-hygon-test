"""One original eight-card native training process forest, inside its Slurm step."""
import ctypes,hashlib,importlib.metadata as metadata,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
from scope import ROOT,NODE,PARENT,PORT,NATIVE_SECONDS,TERM_SECONDS,remaining,EpochClock
from snapshot import snapshot as device_snapshot
from owned_processes import snapshot,refresh,cleanup

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()

def record(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)

def main():
 intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];clock=EpochClock(epoch);clock.left(NATIVE_SECONDS)
 assert socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']==PARENT
 assert int(os.environ['SLURM_CPUS_PER_TASK'])==64 and len(os.sched_getaffinity(0))>=64
 before=json.loads((ROOT/'before.json').read_bytes());leased=device_snapshot(True)
 assert leased['node']==NODE and leased['pci']==before['pci'] and leased['boot_id']==before['boot_id'] and leased['kfd_holder_count']==0
 assert 0<=leased['monotonic_ns']-before['monotonic_ns']<=45_000_000_000
 with socket.socket() as check:check.bind(('127.0.0.1',PORT))
 hip=ctypes.CDLL('libamdhip64.so');hip.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)];hip.hipGetDeviceCount.restype=ctypes.c_int
 hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];hip.hipDeviceGetPCIBusId.restype=ctypes.c_int
 count=ctypes.c_int();assert hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8
 cards=[]
 for i in range(8):
  buffer=ctypes.create_string_buffer(32);assert hip.hipDeviceGetPCIBusId(buffer,len(buffer),i)==0
  pci=buffer.value.decode().lower()
  if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):pci='0000:'+pci
  assert re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci)
  cards.append(pci)
 assert len(set(cards))==8 and sorted(cards)==leased['pci']
 deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());framework=Path(deps['framework']['source_directory'])
 proc=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
 assigned=dict(node=NODE,step=PARENT+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(proc[19]),PCI_by_index=cards,leased_snapshot=leased,CPU_affinity=sorted(os.sched_getaffinity(0)),SLURM_CPUS_PER_TASK=64,framework_commit=deps['framework']['framework_commit'])
 record('assigned.json',assigned)
 versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate','tokenizers','torchvision')}
 assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
 record('actual-versions.json',dict(versions=versions,driver_module_version=Path('/sys/module/amdgpu/version').read_text().strip() if Path('/sys/module/amdgpu/version').exists() else None,LD_LIBRARY_PATH_SHA256=hashlib.sha256(os.environ.get('LD_LIBRARY_PATH','').encode()).hexdigest(),PYTHONPATH=os.environ.get('PYTHONPATH')))
 argv=[sys.executable,'-B',str(framework/'tools/hygon_training_recipe.py'),'T05.sft',
  '--a100-launch',str(ROOT/'src/original-launch.json'),'--swift',deps['swift']['swift_script'],
  '--models',str(Path(deps['model']['model_directory']).parent),'--dataset',deps['inputs']['container_dataset'],
  '--plugins',str(Path(deps['inputs']['host_root'])/'plugins'),'--output',str(ROOT/'train'),
  '--master-port',str(PORT),'--diagnostic-plugin',str(framework/'vlm/plugins/native_dropout_alignment.py'),
  '--diagnostic-callback','native_lora_dropout_alignment','--diagnostic-plugin',str(framework/'vlm/plugins/first_updates.py'),
  '--diagnostic-callback','original_first_updates','--diagnostic-callback-position','before',
  '--run','--receipt',str(ROOT/'entry-receipt.jsonl')]
 assert not any(n in os.environ for n in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK','MASTER_ADDR'))
 env=os.environ.copy();env.update(FIRST_UPDATE_EPOCH=str(epoch),FIRST_UPDATE_DEADLINE=str(epoch+NATIVE_SECONDS),FIRST_UPDATE_MAX_SECONDS=str(NATIVE_SECONDS),FIRST_UPDATE_CAPTURE=str(ROOT/'capture'),FIRST_UPDATE_MODEL_SOURCE_SHA256=deps['qwen_provider_source_SHA256'],NATIVE_LORA_DROPOUT_CONFIG=str(ROOT/'src/MASK_CONFIG.json'),NATIVE_LORA_DROPOUT_CONFIG_SHA256=sha(ROOT/'src/MASK_CONFIG.json'))
 record('native-intent.json',dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,command=argv,framework_commit=deps['framework']['framework_commit'],GPU_cards=8,planned_updates=25,planned_capture_updates=2,planned_capture_micro_forwards_per_rank=8,mask_config_SHA256=env['NATIVE_LORA_DROPOUT_CONFIG_SHA256'],formal_performance=False))
 registry={};child=None;error=None;timed_out=False;last_saved=0
 try:
  with (ROOT/'native.log').open('xb') as log:
   child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row=snapshot().get(child.pid)
   if row:registry[child.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')}
   record('child-owner.json',dict(PID=child.pid,identity=registry.get(child.pid)))
   while child.poll() is None:
    refresh(registry,ROOT)
    if time.monotonic()-last_saved>=15:
     temp=ROOT/'owned-registry.tmp';temp.write_bytes(json.dumps(registry).encode());temp.replace(ROOT/'owned-registry.json');last_saved=time.monotonic()
    left=clock.left(NATIVE_SECONDS)
    try:child.wait(timeout=min(1,left))
    except subprocess.TimeoutExpired:pass
 except TimeoutError:timed_out=True
 except BaseException as failure:error=dict(type=type(failure).__name__,message=str(failure)[:300])
 finally:
  if child is not None and child.poll()==0:
   until=time.monotonic()+5
   while refresh(registry,ROOT) and time.monotonic()<until:time.sleep(.2)
  cleaned=cleanup(registry,ROOT,TERM_SECONDS)
  if child is not None:
   try:child.wait(timeout=1)
   except subprocess.TimeoutExpired:error=error or dict(type='OWNED_CHILD_REMAINS')
 counts={}
 for kind,pattern in [('baseline','train/nvidia-baseline/rank-%02d.json'),('capture','capture/rank-%02d/receipt.json'),('native25','capture/rank-%02d/native25-complete.json'),('masks','masks/rank-%02d/complete.json')]:counts[kind]=sum((ROOT/(pattern%i)).is_file() for i in range(8))
 out=dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,native_returncode=child.returncode if child else None,timed_out=timed_out,error=error,cleanup=cleaned,owners=registry,rank_record_counts=counts,native_log_SHA256=sha(ROOT/'native.log') if (ROOT/'native.log').exists() else None,normal_native_exit=bool(child and child.returncode==0 and not timed_out and error is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and set(counts.values())=={8}),qualification='Producer record counts and process exit only; independent full artifacts/numerics/recovery/performance/effects remain unqualified.',Model_GO=False)
 record('native-outcome.json',out);print(json.dumps({k:v for k,v in out.items() if k!='owners'}))
 return 0 if out['normal_native_exit'] else 1
if __name__=='__main__':raise SystemExit(main())
