"""One original eight-card native training process forest, inside its Slurm step."""
import ctypes,hashlib,importlib.metadata as metadata,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
from scope import BASE,ROOT,NODE,PARENT,PORT,NATIVE_SECONDS,TERM_SECONDS,remaining,EpochClock
from snapshot import snapshot as device_snapshot
from verify_package import FRAMEWORK, STAGE
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
 from recipe import prepare
 proc=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
 assigned=dict(node=NODE,step=PARENT+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(proc[19]),PCI_by_index=cards,leased_snapshot=leased,CPU_affinity=sorted(os.sched_getaffinity(0)),SLURM_CPUS_PER_TASK=64,framework_commit='096157e7b63f7e9cd60cb53f31b10f2810ac2ddc')
 record('assigned.json',assigned)
 versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate','tokenizers','torchvision')}
 import torch
 runtime=dict(torch=str(torch.__version__),hip=torch.version.hip)
 record('torch-provider.json',dict(runtime=runtime,distribution=versions['torch'],
  provider_path=str(Path(torch.__file__).resolve())))
 assert runtime==dict(torch='2.10.0',hip='6.3.26113'), runtime
 assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
 record('actual-versions.json',dict(versions=versions,driver_module_version=Path('/sys/module/amdgpu/version').read_text().strip() if Path('/sys/module/amdgpu/version').exists() else None,LD_LIBRARY_PATH_SHA256=hashlib.sha256(os.environ.get('LD_LIBRARY_PATH','').encode()).hexdigest(),PYTHONPATH=os.environ.get('PYTHONPATH')))
 launch=prepare((STAGE/'original-launch.json').read_bytes(),STAGE/'QWEN_TARGET_NATIVE_BINDING.json',
  framework/'vlm/runtime/swift_entry.py',framework,ROOT,PORT,epoch=epoch,
  deadline=epoch+NATIVE_SECONDS,max_seconds=NATIVE_SECONDS)
 import qwen_vl_utils
 assert metadata.version("qwen-vl-utils")=="0.0.14"
 assert Path(qwen_vl_utils.__file__).resolve()==Path("/vision/qwen_vl_utils/__init__.py")
 record("vision-provider.json",dict(version="0.0.14",location=str(qwen_vl_utils.__file__)))
 command=launch['command']
 callbacks=command.index('--callbacks')+1
 assert command[callbacks:callbacks+3]==['native_initial_trainables','native_first_updates','nvidia_baseline']
 command.insert(callbacks+3,'qwen_native_rank1_vproj_lt_first_two')
 plugins=command.index('--external_plugins')+1
 command.insert(plugins,str(BASE/'src/candidate_plugin.py'))
 argv=[sys.executable,'-B',*command]
 assert not any(n in os.environ for n in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK'))
 env=os.environ.copy()
 env.update(launch['environment_overrides'])
 env.update(QWEN_REPEAT_OUTPUT=str(ROOT),QWEN_REPEAT_PINS=str(ROOT/'src/ARCHIVES.json'))
 env.update(MASTER_ADDR='127.0.0.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 record('native-intent.json',dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,
  framework_commit='096157e7b63f7e9cd60cb53f31b10f2810ac2ddc',GPU_cards=8,planned_updates=2,
  forwards_per_rank=8,original_first_input_parent='1087656.433',formal_performance=False))
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
 rank_results=[]
 for rank in range(8):
  cap=ROOT/'capture'/f'rank-{rank:02d}'
  files=(cap/'receipt.json',cap/'first-two-scope-complete.json',ROOT/f'lt-policy-rank-{rank:02d}.json')
  row={'rank':rank,'complete':False,'present':[f.is_file() for f in files]}
  if all(row['present']):
   try:
    result, stop, policy=(json.loads(f.read_bytes()) for f in files)
    row.update(actual_updates=result['optimizer_updates'], micro_forwards=result['micro_forwards'])
    row['complete']=(result['rank']==stop['rank']==policy['rank']==rank
       and result['state']=='TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED'
       and result['completed_global_step']==result['optimizer_updates']==stop['global_step']==2
       and result['micro_forwards']==8 and result['clip_calls']==2 and len(result['files'])==25
       and [r['global_step'] for r in policy['updates']]==[1,2]
       and policy['selected_site']=='decoder_10.q_proj' and policy['shortened_performance_end'] is True
       and policy['performance_report_emitted'] is False and len(policy['forwards'])==8
       and policy['forwards'][0]['value_lt_calls']==int(rank==1)
       and all(r['value_lt_calls'] in ((0,1) if rank==1 else (0,)) for r in policy['forwards'])
       and all(r['base_forward_calls']==1008 and r['selected_forward_calls']==4
               and r['value_lt_forwards']==sum(f['value_lt_calls'] for f in policy['forwards'][4*i:4*i+4])
               and r['value_lt_checkpoint_recomputes']>=0 for i,r in enumerate(policy['updates'])))
    row['receipts_SHA256']={f.name:sha(f) for f in files}
   except (ValueError,KeyError,TypeError,OSError): pass
  rank_results.append(row)
 counts={'completed':sum(row['complete'] for row in rank_results)}
 qualified=bool(counts['completed']==8)
 out=dict(epoch=epoch,native_deadline=epoch+NATIVE_SECONDS,native_process_spawned=child is not None,
  native_returncode=child.returncode if child else None,timed_out=timed_out,error=error,
  cleanup=cleaned,owners=registry,rank_record_counts=counts,rank_results=rank_results,
  native_log_SHA256=sha(ROOT/'native.log') if (ROOT/'native.log').exists() else None,
  normal_native_exit=bool(child and child.returncode==0 and not timed_out and error is None
     and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and qualified),formal_training=False)
 record('native-outcome.json',out);print(json.dumps({k:v for k,v in out.items() if k!='owners'}))
 return 0 if out['normal_native_exit'] else 1
if __name__=='__main__':raise SystemExit(main())
