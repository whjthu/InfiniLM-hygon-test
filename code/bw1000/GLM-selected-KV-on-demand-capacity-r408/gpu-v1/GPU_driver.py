"""TP8 actual-checkpoint attention geometry cost; NOT fullModel or routing."""
from pathlib import Path
import ctypes,json,hashlib,os,time,traceback,gc
import sys
sys.path.insert(0,'/refactor/incoming/GLM-selected-KV-on-demand-capacity-r408/cpu-v2')
from journal import Journal,aggregate
from consumer import consume,NativeLeaf,CASES,MODES,NUMERIC_ALERT_NRMSE,NUMERIC_DIAGNOSTIC_STOP_NRMSE
from checkpoint_loader import Checkpoint,load,KEYS
R=Path('/refactor');D=Path(__file__).resolve().parent;M=Path('/old/models/ZhipuAI-GLM-5.3')
V=ctypes.c_void_p;I=ctypes.c_int;Z=ctypes.c_size_t
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
 p=D/'GPU_MANIFEST.json';assert sha(p)==os.environ['GLM408_GPU_SOURCE_SHA256'];m=json.loads(p.read_bytes());assert m['GPU_GO']and not m['Model_GO']
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 source=R/'incoming/GLM-selected-KV-on-demand-capacity-r408/cpu-v2';assert sha(source/'CPU_SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256'];sm=json.loads((source/'CPU_SOURCE_MANIFEST.json').read_bytes());assert all(sha(source/n)==h for n,h in sm['files_SHA256'].items())
 p=Path(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256'];q=json.loads(p.read_bytes());assert q['completed']and q['passed']and q['HOST_TUs_compiled']==2 and not q['Native_dlopen']and not q['checkpoint_payload_IO'];assert sha(q['library'])==q['library_SHA256']==m['bridge_SHA256']
 for n in ['config.json','model.safetensors.index.json']:assert sha(M/n)==m['checkpoint_metadata_SHA256'][n]
 return q,m

def api(path):
 lib=ctypes.CDLL(str(path));lib.glm_layer_id_size_r5670.restype=Z;lib.glm_layer_id_r5670.argtypes=[V]
 lib.glm_attention_create_r398.argtypes=[I,ctypes.c_char_p,V];lib.glm_attention_create_r398.restype=V
 for n in ['keys','error','report']:getattr(lib,'glm_attention_'+n+'_r398').restype=ctypes.c_char_p
 lib.glm_attention_keys_r398.argtypes=[V];lib.glm_attention_load_r398.argtypes=[V,ctypes.c_char_p];lib.glm_attention_finish_r398.argtypes=[V]
 return lib

def pci(rank):
 lib=ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');lib.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,I,I];p=ctypes.create_string_buffer(64);assert lib.cudaDeviceGetPCIBusId(p,64,rank)==0;return p.value.decode()

def symbol_owners(lib,expected):
 class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('addr',V)]
 process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(Info)];process.dladdr.restype=I;out={}
 for n in ['glm_attention_create_r398','glm_attention_query_r398','glm_attention_compare_k_r398','glm_attention_control_r398','glm_attention_destroy_r398','glm_layer_id_r5670','glm_fp8_create_r776']:
  info=Info();assert process.dladdr(ctypes.cast(getattr(lib,n),V),ctypes.byref(info));actual=Path(info.fname.decode()).resolve();target=Path('/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so')if n=='glm_fp8_create_r776'else Path(expected);assert actual==target.resolve();out[n]=str(actual)
  if n=='glm_fp8_create_r776':assert sha(actual)=='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
 return out

def worker(rank,uid,start,physical):
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,cpus[4*rank:4*rank+4])
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for n in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=R/'results'/('GLM408-'+step+'-rank%d.json'%rank)
 assert not path.exists();q=dict(completed=False,passed=False,normal=False,actual_step=step,rank=rank,node=os.uname().nodename,PID=os.getpid(),parent_PID=os.getppid(),physical_PCI=physical[rank],CPU_affinity=sorted(os.sched_getaffinity(0)),GPU_source_manifest_SHA256=sha(D/'GPU_MANIFEST.json'),records=[],scope=__doc__,profile=0,Model=False,indexer_included=False,ordinary_host_selection_mapping_included=True,allocator_actual_peak='UNKNOWN; recorded snapshots are not peaks',numeric_policy=dict(alert_NRMSE=NUMERIC_ALERT_NRMSE,diagnostic_stop_NRMSE=NUMERIC_DIAGNOSTIC_STOP_NRMSE,scope='Same synthetic geometry numerical diagnostic; not task quality or acceptance; no bitwise criterion'))
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
 journal=Journal(path.with_suffix('.jsonl'))
 def update(records,error):journal.update(records,error)
 def limit():
  if time.time()>=start+540:raise TimeoutError('r398 original epoch540 inner gate; GPU600 total; no retry')
 save();ctx=None;leaf=None;lib=None
 try:
  limit();a,m=verify();assert pci(rank)==physical[rank]
  props=torch.cuda.get_device_properties(rank);q['gcnArchName']=props.gcnArchName;assert props.gcnArchName.split(':')[0]=='gfx936'
  q['library_SHA256']=a['library_SHA256'];lib=api(a['library']);q['actual_symbol_owners']=symbol_owners(lib,a['library'])
  maps=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if line.split()[-1].startswith('/')and'.so'in line.split()[-1]})
  q['actual_mapped_SDK']={p:sha(p)for p in maps if any(x in Path(p).name for x in ['infini','cudart','blas_policy','fp8_batched'])}
  for p,h in a['actual_dynamic_dependency_SHA256'].items():
   expected=Path(p).resolve()
   if any(x in expected.name for x in ['infini','cudart']):
    matches=[Path(x).resolve()for x in maps if Path(x).resolve().name==expected.name]
    assert matches and all(x==expected and sha(x)==h for x in matches),(str(expected),list(map(str,matches)))
  cfg=json.loads((M/'config.json').read_bytes());uidbuf=ctypes.create_string_buffer(uid,len(uid));beg=time.perf_counter();ctx=lib.glm_attention_create_r398(rank,json.dumps(cfg).encode(),uidbuf);assert ctx,lib.glm_attention_error_r398();q['create_wall_s']=time.perf_counter()-beg
  keys=json.loads(lib.glm_attention_keys_r398(ctx));assert sorted(keys)==KEYS
  def check(rc):
   if rc:raise RuntimeError(lib.glm_attention_error_r398().decode())
  beg=time.perf_counter();q['load']=load(lib,ctx,keys,Checkpoint(M,torch),check,limit);q['load_wall_s']=time.perf_counter()-beg;q['free_after_weightload']=torch.cuda.mem_get_info(rank)[0];assert q['free_after_weightload']>=4*1024**3
  limit();beg=time.perf_counter();generator=torch.Generator(device='cpu').manual_seed(398038)
  cpu=(torch.randn((32768,6144),generator=generator,dtype=torch.bfloat16)*.1).contiguous();input_hash=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest();hidden=cpu.cuda();del cpu;torch.cuda.synchronize(rank)
  q.update(synthetic_hidden_SHA256=input_hash,fixture_prepare_wall_s=time.perf_counter()-beg,input_shape=[32768,6144],actual_checkpoint_payload_loaded=True,wholeModel_loaded=False,hidden_is_Model_activation=False,selected_indices_are_Model_routing=False)
  before=time.perf_counter();leaf=NativeLeaf(lib,ctx,torch,hidden,start+540);records=consume(leaf,update);ctx=None;q['whole_component_sequence_wall_s']=time.perf_counter()-before
  assert len(records)==12 and sum(r['warm']for r in records)==4 and sum(not r['warm']for r in records)==8
  q['records']=records
  assert hashlib.sha256(hidden.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()==input_hash
  del hidden;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(rank);t=torch.ones(32,device='cuda');assert(t+1).sum().item()==64;torch.cuda.synchronize(rank)
  q.update(completed=True,passed=True,normal=True,normal_destroy=True,Runtime_weak_expired=True,Torch_after_destroy=True,policy_closed=True,measurement_calls=8,warm_calls=4,directK_diagnostics=2,weights_loaded_once=True,input_immutable=True,free_postclose=torch.cuda.mem_get_info(rank)[0],owner_service_residency_s=time.time()-start);print('GLM408_NORMAL '+str(rank),flush=True)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:
  journal.close();q['records']=aggregate(journal.path)['records']
  if leaf is not None and leaf.destroy_attempted:q['destroy_attempted_once']=True;q['destroy_RC']=getattr(leaf,'destroy_rc',None);q['Runtime_expired']=getattr(leaf,'runtime_expired',False)
  if ctx and lib and (leaf is None or not leaf.destroy_attempted):
   flag=I();lib.glm_attention_destroy_r398.argtypes=[V,ctypes.POINTER(I)]
   try:q['failure_destroy_RC']=int(lib.glm_attention_destroy_r398(ctx,ctypes.byref(flag)));q['failure_Runtime_expired']=bool(flag.value)
   except BaseException as e:q['failure_destroy_error']=repr(e)
  q['completed']=True;save()

if __name__=='__main__':
 import torch,torch.multiprocessing as mp
 a,m=verify();assert torch.cuda.device_count()==8;torch.cuda.set_device(0);physical=[pci(i)for i in range(8)];assert len(set(physical))==8
 lib=api(a['library']);uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0
 start=float(os.environ['GLM408_GPU_START']);assert time.time()<start+540
 mp.spawn(worker,args=(bytes(uid),start,physical),nprocs=8,join=True);print('GLM408_ALL8_NORMAL',flush=True)
