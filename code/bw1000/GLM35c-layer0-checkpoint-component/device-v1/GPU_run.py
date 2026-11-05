"""Actual checkpoint Layer0 diagnostic only; not a Model or performance run."""
from pathlib import Path
import ctypes as C,json,hashlib,os,time,traceback,gc,threading
from shared import verify,sha,LABEL,NAME,ROOT_RESULT
from component_api import open_library,load_checkpoint,run_first_query,checked
from capture import Trace,compare
D=Path(__file__).resolve().parent;R=Path('/refactor')
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def pci(rank):
 lib=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');f=lib.cudaDeviceGetPCIBusId;f.argtypes=[C.c_char_p,C.c_int,C.c_int];f.restype=C.c_int;b=C.create_string_buffer(64);assert f(b,64,rank)==0;return b.value.decode().lower()
def owners(lib,expected):
 class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('addr',C.c_void_p)]
 process=C.CDLL(None);process.dladdr.argtypes=[C.c_void_p,C.POINTER(Info)];process.dladdr.restype=C.c_int;result={}
 for n in ['error','id_size','id','create','keys','load','finish','step','snapshot','read','reset','destroy']:
  info=Info();name='glm_l0_'+n;assert process.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info));p=Path(info.fname.decode()).resolve();assert p==Path(expected).resolve();result[name]=str(p)
 return result

def worker(rank,uid,start,physical,common):
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 assert len(common)>=32 and sorted(os.sched_getaffinity(0))==common
 assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4';owner_TID=threading.get_native_id();affinity=None
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for n in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/ROOT_RESULT/step;path=out/('rank%d.json'%rank)
 q=dict(completed=False,normal=False,rank=rank,actual_step=step,node=os.uname().nodename,PID=os.getpid(),parent_PID=os.getppid(),physical_PCI=physical[rank],pre_first_submit_common_CPU_affinity=common,owner_TID=owner_TID,affinity_policy_R2392_width=4,component_only=True,whole_Model_loaded=False,random_hidden=False,GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),Model_quality=False,performance=False,source_shapes='real Layer0/TP8PP1/C1/full32K; querywidth1vs8 atpast2949',snapshots=[],Native_trace=[])
 ctx=None;lib=None;trace=None;destroy_attempted=False
 def save():atomic(path,q)
 def limit(phase='owner'):
  if time.time()>=start+540:raise TimeoutError('original owner epoch540s '+phase)
 save()
 try:
  limit('before metadata');m,a,above=verify(R,os.environ['GLM_L0_DEVICE_SOURCE_SHA']);assert pci(rank)==physical[rank]
  props=torch.cuda.get_device_properties(rank);q['gcnArchName']=props.gcnArchName;assert props.gcnArchName.split(':')[0]=='gfx936'
  lib=open_library(a['library'],m['library_SHA256']);q['actual_symbol_owners']=owners(lib,a['library']);q['library_SHA256']=m['library_SHA256']
  lib.modern_glm_vector_policy_r8540.argtypes=[C.c_int,C.c_int];lib.modern_glm_vector_policy_r8540.restype=C.c_int;assert lib.modern_glm_vector_policy_r8540(rank,0)==0
  maps={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if line.split()and line.split()[-1].startswith('/')}
  q['actual_qualified_dependencies']={}
  for p,h in m['qualified_dependency_SHA256'].items():
   expected=Path(p).resolve();matching={Path(x).resolve()for x in maps if Path(x).name==Path(p).name}
   assert matching and matching=={expected}and sha(expected)==h,(p,list(map(str,matching)));q['actual_qualified_dependencies'][str(expected)]=h
  torch.cuda.set_device(rank);limit('create');assert sorted(os.sched_getaffinity(0))==common and threading.get_native_id()==owner_TID;u=C.create_string_buffer(uid,len(uid));cfg=(Path(m['model'])/'config.json').read_bytes();ctx=lib.glm_l0_create(1,rank,cfg,u,1)
  if not ctx:raise RuntimeError(lib.glm_l0_error().decode())
  affinity=sorted(os.sched_getaffinity(0));assert affinity==common[rank*4:rank*4+4] and threading.get_native_id()==owner_TID
  q['owner_CPU_affinity']=affinity;q['SDK_first_submit_affinity_observed']=True
  q['Context_pointer']=int(ctx);q['free_after_create']=list(torch.cuda.mem_get_info(rank));save()
  meta=json.loads((D/'CHECKPOINT_METADATA.json').read_bytes());q['load']=load_checkpoint(lib,ctx,model=m['model'],torch=torch,safe_open=safe_open,expected_metadata=meta,deadline_check=limit)
  q['free_after_load']=list(torch.cuda.mem_get_info(rank));assert q['free_after_load'][0]>4*1024**3
  q['actual_prompt_tokens']=len(above['prompt']);q['actual_prefix_chunk_widths']=above['chunk_widths'];q['actual_original_plan_SHA256']=m['original_plan_SHA256'];q['actual_frozen_plan_SHA256']=m['frozen_plan_SHA256'];save()
  trace=Trace(lib,above,out,rank,start+540);actual=run_first_query(trace,ctx,above,deadline_check=limit);q['snapshots']=trace.snapshots;q['Native_trace']=trace.calls;q['raw_bytes']=trace.used
  assert len(q['snapshots'])==2 and [s['proof']['width']for s in q['snapshots']]==[1,8]
  assert len([s for s in trace.calls if s['kind']=='step'])==2*len(above['chunk_widths'])+2 and len([s for s in trace.calls if s['kind']=='reset'])==3 and all(s['RC']==0 for s in trace.calls)
  q['comparison']=compare(actual);del actual;save();limit('destroy');expired=C.c_int();destroy_attempted=True;rc=lib.glm_l0_destroy(ctx,C.byref(expired));ctx=None
  checked(lib,rc);assert expired.value==1;q['normal_destroy']=True;q['Runtime_weak_expired']=True
  assert sorted(os.sched_getaffinity(0))==affinity and threading.get_native_id()==owner_TID;torch.cuda.set_device(rank);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(rank)
  t=torch.ones(32,device='cuda:'+str(rank));assert(t+1).sum().item()==64;torch.cuda.synchronize(rank);del t
  q.update(Torch_after_destroy=True,owner_device_contract=True,owner_thread_exit_imminent=True,all_fields_finite_with_only_causal_future_mask_INF=True,raw_bytes=trace.used,diagnostic_completed=True,completed=True,normal=True,free_after_close=list(torch.cuda.mem_get_info(rank)),owner_elapsed_since_GPUepoch=time.time()-start);save();print('GLM_L0_NORMAL '+str(rank),flush=True)
 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc())
  if trace is not None:q['Native_trace']=trace.calls;q['snapshots']=trace.snapshots;q['raw_bytes']=trace.used
  save();raise
 finally:
  if ctx and lib and not destroy_attempted:
   destroy_attempted=True;expired=C.c_int()
   try:q['failure_destroy_RC']=int(lib.glm_l0_destroy(ctx,C.byref(expired)));q['failure_Runtime_expired']=bool(expired.value)
   except BaseException as error:q['failure_destroy_error']=repr(error)
   save()

if __name__=='__main__':
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/ROOT_RESULT/step;out.mkdir(parents=True,exist_ok=False)
 entry=dict(completed=False,normal=False,PID=os.getpid(),node=os.uname().nodename,actual_step=step,GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),component_only=True,Model=False);path=out/'entry.json';atomic(path,entry)
 try:
  start=float(os.environ['GLM_L0_DEVICE_START']);assert time.time()<start+540
  common=sorted(os.sched_getaffinity(0));assert len(common)>=32 and os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'
  entry['pre_bootstrap_common_CPU_affinity']=common
  m,a,above=verify(R,os.environ['GLM_L0_DEVICE_SOURCE_SHA'])
  import torch,torch.multiprocessing as mp
  torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0);physical=[pci(i)for i in range(8)];assert len(set(physical))==8
  lib=open_library(a['library'],m['library_SHA256']);u=C.create_string_buffer(lib.glm_l0_id_size());checked(lib,lib.glm_l0_id(u));entry.update(physical_PCI_by_actual_CUDAindex=physical,actual_query_startup_before_owners=True,first_owner_start_elapsed=time.time()-start);atomic(path,entry)
  assert sorted(os.sched_getaffinity(0))==common,'Bootstrap must not narrow inherited owner CPU mask'
  mp.spawn(worker,args=(bytes(u),start,physical,common),nprocs=8,join=True)
  owners_=[json.loads((out/('rank%d.json'%i)).read_bytes())for i in range(8)];assert all(q['normal']and q['completed']for q in owners_)and sum(q['raw_bytes']for q in owners_)<=512*1024**2
  masks=[q['owner_CPU_affinity']for q in owners_];assert all(masks[i]==common[i*4:i*4+4]for i in range(8)) and len(set(sum(masks,[])))==32
  assert all(q['SDK_first_submit_affinity_observed'] and q['owner_TID']==q['PID']for q in owners_)
  entry.update(completed=True,normal=True,normal8=True,all_owner_threads_processes_joined=True,total_raw_bytes=sum(q['raw_bytes']for q in owners_),forwards_per_owner=2*len(above['chunk_widths'])+2,actual_layer=0,scope='GenuineLayer0 component diagnostic only; no generation/fullModel/performance');atomic(path,entry);print('GLM_L0_ALL8_NORMAL',flush=True)
 except BaseException as error:entry.update(completed=True,error=repr(error),traceback=traceback.format_exc());atomic(path,entry);raise
