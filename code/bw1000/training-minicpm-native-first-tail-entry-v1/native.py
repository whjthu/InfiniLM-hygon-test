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
 deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());framework=Path(deps['framework']['source_directory'])
 proc=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
 assigned=dict(node=NODE,step=PARENT+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(proc[19]),PCI_by_index=cards,leased_snapshot=leased,CPU_affinity=sorted(os.sched_getaffinity(0)),SLURM_CPUS_PER_TASK=64,framework_commit=deps['framework']['framework_commit'])
 record('assigned.json',assigned)
 versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate','tokenizers','torchvision')}
 assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
 record('actual-versions.json',dict(versions=versions,driver_module_version=Path('/sys/module/amdgpu/version').read_text().strip() if Path('/sys/module/amdgpu/version').exists() else None,LD_LIBRARY_PATH_SHA256=hashlib.sha256(os.environ.get('LD_LIBRARY_PATH','').encode()).hexdigest(),PYTHONPATH=os.environ.get('PYTHONPATH')))
 registry={};arms=[];error=None;last_saved=0
 def persist():
  temp=ROOT/'owned-registry.tmp';temp.write_bytes(json.dumps(registry).encode());temp.replace(ROOT/'owned-registry.json')
 def checkpoint_pins(folder):
  names=['trainer_state.json','args.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt']+['rng_state_%d.pth'%r for r in range(8)]
  pins={}
  for name in names:
   path=folder/name;assert path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=256*1024**2
   pins[name]=dict(bytes=path.stat().st_size,SHA256=sha(path))
  return pins
 def run_arm(name,config):
  nonlocal last_saved
  arm=ROOT/name;arm.mkdir(mode=0o700);cp=ROOT/(name+'-config.json');record(cp.name,config)
  argv=[sys.executable,'-B',str(framework/'tools/hygon_training_recipe.py'),'T05.sft','--a100-launch',str(ROOT/'src/original-launch.json'),'--swift',str(framework/'vlm/runtime/swift_entry.py'),'--models',str(Path(deps['model']['model_directory']).parent),'--dataset',deps['inputs']['container_dataset'],'--plugins',str(Path(deps['inputs']['host_root'])/'plugins'),'--output',str(arm/'train'),'--master-port',str(PORT),'--diagnostic-plugin',str(framework/'vlm/plugins/resume_training.py'),'--diagnostic-callback','native_resume_witness','--resume-from-checkpoint',config['checkpoint'],'--run','--receipt',str(arm/'entry-receipt.jsonl')]
  assert not any(n in os.environ for n in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK','MASTER_ADDR'))
  env=os.environ.copy();env.update(NATIVE_RESUME_CONFIG=str(cp),NATIVE_RESUME_CONFIG_SHA256=sha(cp),NATIVE_RESUME_EPOCH=str(epoch),NATIVE_RESUME_DEADLINE=str(epoch+NATIVE_SECONDS),NATIVE_RESUME_MAX_SECONDS=str(NATIVE_SECONDS))
  record(name+'-native-intent.json',dict(command=argv,epoch=epoch,deadline=epoch+NATIVE_SECONDS,planned_global_steps=list(range(config['checkpoint_step']+1,22)),config_SHA256=sha(cp),GPU_cards=8,formal_performance=False))
  p=None;failure=None;timed_out=False;started=time.monotonic()
  try:
   with (arm/'native.log').open('xb') as log:
    p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=snapshot().get(p.pid)
    if row:registry[p.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')}
    record(name+'-child-owner.json',dict(PID=p.pid,identity=registry.get(p.pid)))
    while p.poll() is None:
     refresh(registry,ROOT)
     if time.monotonic()-last_saved>=2:persist();last_saved=time.monotonic()
     try:p.wait(timeout=min(1,clock.left(NATIVE_SECONDS)))
     except subprocess.TimeoutExpired:pass
  except TimeoutError:timed_out=True
  except BaseException as failed:failure=dict(type=type(failed).__name__,message=str(failed)[:350])
  finally:
   if p is not None and p.poll()==0:
    until=time.monotonic()+5
    while refresh(registry,ROOT) and time.monotonic()<until:time.sleep(.2)
   cleaned=cleanup(registry,ROOT,TERM_SECONDS)
   if p is not None:
    try:p.wait(timeout=1)
    except subprocess.TimeoutExpired:failure=failure or dict(type='OWNED_CHILD_REMAINS')
   persist()
  records=[]
  for rank in range(8):
   root=arm/'train/resume-witness'/('rank-%02d'%rank);path=root/'result.json';owner=root/'owner.json';entry=root/'forward-entry-state.json'
   item=dict(rank=rank,result_present=path.is_file(),owner_present=owner.is_file(),entry_present=entry.is_file())
   if path.is_file():
    raw=path.read_bytes();assert len(raw)<2*1024**2;value=json.loads(raw);assert value['rank']==rank and value['world_size']==8 and value['native_resume_window_complete'] is False and value['declared_observation_window_complete'] is True and value['diagnostic_stop_requested'] is True and value['diagnostic_stop_step']==21 and value['completed_global_step']==21 and value['entry_state_step']==20 and value['entry_state_captured'] is True and value['plan']['checkpoint_global_step']==config['checkpoint_step'];item.update(result_bytes=len(raw),result_SHA256=sha(path),inputs=len(value['inputs']),outputs=len(value['outputs']),steps=[r['global_step'] for r in value['steps']],saved=value['saved_checkpoints'])
   if entry.is_file() and path.is_file():
    assert 0<entry.stat().st_size<=8*1024**2 and sha(entry)==value['entry_state_SHA256'];item.update(entry_bytes=entry.stat().st_size,entry_SHA256=sha(entry))
   if path.is_file():assert item.get('inputs')==(21-config['checkpoint_step'])*4 and item.get('outputs')==item.get('inputs') and item.get('steps')==list(range(config['checkpoint_step']+1,22)) and item.get('saved')==([20] if name=='control' else [])
   if owner.is_file():item['owner_SHA256']=sha(owner)
   records.append(item)
  normal=bool(p and p.returncode==0 and failure is None and not timed_out and cleaned==dict(owned_processes_remaining=0,termination_needed=False) and all(r['result_present'] and r['owner_present'] and r['entry_present'] for r in records))
  out=dict(name=name,native_returncode=p.returncode if p else None,error=failure,timed_out=timed_out,cleanup=cleaned,normal_native_exit=normal,rank_records=records,wall_seconds=time.monotonic()-started,native_log_SHA256=sha(arm/'native.log'))
  record(name+'-native-outcome.json',out);arms.append(out)
  if not normal:raise RuntimeError(name+' native window did not complete and clean normally')
  with socket.socket() as check:check.bind(('127.0.0.1',PORT))
 try:
  initial=json.loads((ROOT/'src/INITIAL_CHECKPOINT.json').read_bytes());assert initial['checkpoint_step']==10
  run_arm('control',dict(initial,entry_state_step=20,diagnostic_stop_step=21))
  folder=ROOT/'control/train/checkpoint-20';pins=checkpoint_pins(folder);state=json.loads((folder/'trainer_state.json').read_bytes());assert state['global_step']==20 and state['max_steps']==25 and state['epoch']==10.0
  record('generated-control-checkpoint20-pins.json',dict(checkpoint=str(folder),checkpoint_step=20,checkpoint_files=pins,derived_from='control normal native window',control_outcome_SHA256=sha(ROOT/'control-native-outcome.json')))
  run_arm('restored',dict(checkpoint=str(folder),checkpoint_step=20,checkpoint_files=pins,entry_state_step=20,diagnostic_stop_step=21))
 except BaseException as failed:error=dict(type=type(failed).__name__,message=str(failed)[:350])
 finally:
  cleaned=cleanup(registry,ROOT,TERM_SECONDS);persist()
 out=dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,arms=arms,error=error,cleanup=cleaned,owners=registry,normal_native_exit=bool(error is None and len(arms)==2 and all(a['normal_native_exit'] for a in arms) and cleaned==dict(owned_processes_remaining=0,termination_needed=False)),independent_pair_audit_pending=True,recovery_qualified=False,Model_GO=False,qualification='Two normally stopped native first-tail diagnostic windows; full25 recovery and performance are not claimed. Independent bound entry comparison and checkpoint20 audit is a separate0GPU step.')
 record('native-outcome.json',out);print(json.dumps({k:v for k,v in out.items() if k!='owners'}));return 0 if out['normal_native_exit'] else 1
if __name__=='__main__':raise SystemExit(main())