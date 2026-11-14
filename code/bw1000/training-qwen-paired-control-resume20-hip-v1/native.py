"""One original eight-card native training process forest, inside its Slurm step."""
import ctypes,hashlib,importlib.metadata as metadata,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
from scope import ROOT,NODE,PARENT,PORT,NATIVE_SECONDS,TERM_SECONDS,remaining,EpochClock
from snapshot import snapshot as device_snapshot
from verify_package import FRAMEWORK, STAGE, ORIGINAL, MODEL, ASSETS
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
 assert before['full_fd_inventory_performed'] is True and leased['full_fd_inventory_performed'] is False and leased['holder_inventory_basis']=='kernel_KFD_proc_only'
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
 framework=FRAMEWORK
 sys.path.insert(0,str(STAGE));sys.path.insert(0,str(framework))
 from tools.hygon_vlm_recipe import build_launch
 proc=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
 assigned=dict(node=NODE,step=PARENT+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(proc[19]),PCI_by_index=cards,leased_snapshot=leased,CPU_affinity=sorted(os.sched_getaffinity(0)),SLURM_CPUS_PER_TASK=64,framework_commit='708916c9bc955c32087684570d6abf11b67e98b4')
 record('assigned.json',assigned)
 versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate','tokenizers','torchvision')}
 import torch
 runtime=dict(torch=str(torch.__version__),hip=torch.version.hip)
 record('torch-provider.json',dict(runtime=runtime,distribution=versions['torch'],
  provider_path=str(Path(torch.__file__).resolve())))
 assert runtime==dict(torch='2.10.0',hip='6.3.26113'), runtime
 assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
 record('actual-versions.json',dict(versions=versions,driver_module_version=Path('/sys/module/amdgpu/version').read_text().strip() if Path('/sys/module/amdgpu/version').exists() else None,LD_LIBRARY_PATH_SHA256=hashlib.sha256(os.environ.get('LD_LIBRARY_PATH','').encode()).hexdigest(),PYTHONPATH=os.environ.get('PYTHONPATH')))
 input_scope=json.loads((ROOT/'src/RESUME_INPUTS.json').read_bytes())
 checkpoint=ORIGINAL/'train/checkpoint-20'
 pins={k:v for k,v in input_scope['checkpoint20'].items() if k!='training_args.bin'}
 config=dict(checkpoint=str(checkpoint),checkpoint_files=pins)
 record('resume-config.json',config)
 launch=build_launch('T01.sft',(STAGE/'original-launch.json').read_bytes(),
  framework/'vlm/runtime/swift_entry.py',MODEL.parent,ASSETS/'vlm_report_coco64.hip.jsonl',ASSETS/'plugins',
  ROOT/'train',PORT,diagnostic_plugin=framework/'vlm/plugins/qwen_resume_training.py',
  diagnostic_callback='qwen_native_resume_witness',resume_from_checkpoint=checkpoint)
 launch['command'][launch['command'].index('--model')+1]=str(MODEL)
 assert launch['command'][launch['command'].index('--callbacks')+1]=='qwen_native_resume_witness'
 import qwen_vl_utils
 assert metadata.version("qwen-vl-utils")=="0.0.14"
 assert Path(qwen_vl_utils.__file__).resolve()==Path("/vision/qwen_vl_utils/__init__.py")
 record("vision-provider.json",dict(version="0.0.14",location=str(qwen_vl_utils.__file__)))
 argv=[sys.executable,'-B',*launch['command']]
 assert not any(n in os.environ for n in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK'))
 env=os.environ.copy()
 env.update(launch['environment_overrides'])
 env.update(MASTER_ADDR='127.0.0.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 env.update(NATIVE_RESUME_CONFIG=str(ROOT/'resume-config.json'),NATIVE_RESUME_CONFIG_SHA256=sha(ROOT/'resume-config.json'),
  QWEN_RESUME_BINDING=str(STAGE/'QWEN_TARGET_NATIVE_BINDING.json'),NATIVE_RESUME_EPOCH=str(epoch),
  NATIVE_RESUME_DEADLINE=str(epoch+NATIVE_SECONDS),NATIVE_RESUME_MAX_SECONDS=str(NATIVE_SECONDS))
 record('native-intent.json',dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,command=argv,
  framework_commit='708916c9bc955c32087684570d6abf11b67e98b4',GPU_cards=8,planned_updates=5,resume_from_step=20,
  planned_capture_updates=5,planned_capture_micro_forwards_per_rank=20,formal_performance=False))
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
 for kind,pattern in [('resume','train/resume-witness/rank-%02d/result.json')]:
  counts[kind]=sum((ROOT/(pattern%i)).is_file() for i in range(8))
 out=dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,native_process_spawned=child is not None,native_returncode=child.returncode if child else None,timed_out=timed_out,error=error,cleanup=cleaned,owners=registry,rank_record_counts=counts,native_log_SHA256=sha(ROOT/'native.log') if (ROOT/'native.log').exists() else None,normal_native_exit=bool(child and child.returncode==0 and not timed_out and error is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and set(counts.values())=={8}),qualification='Producer record counts and process exit only; independent full artifacts/numerics/recovery/performance/effects remain unqualified.',Model_GO=False)
 record('native-outcome.json',out);print(json.dumps({k:v for k,v in out.items() if k!='owners'}))
 return 0 if out['normal_native_exit'] else 1
if __name__=='__main__':raise SystemExit(main())
