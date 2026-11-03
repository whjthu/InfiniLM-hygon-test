"""TP8 actual-checkpoint attention geometry cost; NOT fullModel or routing."""
from pathlib import Path
import ctypes,json,hashlib,os,time,traceback,gc,math,threading
from consumer import consume,NativeLeaf,CASES,MODES,NUMERIC_ALERT_NRMSE,NUMERIC_DIAGNOSTIC_STOP_NRMSE
from checkpoint_loader import Checkpoint,load,KEYS
R=Path('/refactor');D=Path(__file__).resolve().parent;M=Path('/old/models/ZhipuAI-GLM-5.3')
V=ctypes.c_void_p;I=ctypes.c_int;Z=ctypes.c_size_t
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
 p=D/'GPU_SOURCE_MANIFEST.json';assert sha(p)==os.environ['GLM433_GPU_SOURCE_SHA256'];m=json.loads(p.read_bytes());assert m['GPU_GO']and not m['Model_GO']
 for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
 p=R/'candidates/GLM-row-FP8-roundtrip-value-r433-diagnostic-v2-CPU/CPU-build.json';assert sha(p)=='b3d19be0f3dd6ea982d48eaea5381c088037556cc1bd7159c0d3a48ec1b595db';q=json.loads(p.read_bytes())
 assert q['completed']and q['passed']and q['producer']=='910407.879'and q['production_HOST_TUs_compiled']==1 and q['GPU_TUs_compiled']==0
 assert not q['Native_dlopen']and not q['Runtime']and not q['checkpoint_payload_IO']and len(q['final_objects_SHA256'])==30 and len(q['reused_objects_SHA256'])==29
 assert q['all_parent_strong_and_CABI_retained']and q['shared_dependencies_exact']and q['ldd_r_parent_equal']and q['actual_new_diagnostic_FFI_rawfile_contract_passed']
 assert sha(q['library'])==q['library_SHA256']==m['library_SHA256']=='80a14a45954f4eaeed2dfca64f9be9f7ceb47d404e6e7a746a9c2062b7a285db'
 for field in ['final_objects_SHA256','actual_dynamic_dependency_SHA256']:
  for path,h in q[field].items():assert sha(path)==h,path
 for path,h in q['compiled']['dependencies_SHA256'].items():assert sha(path)==h,path
 assert sha(q['compiled']['source'])==q['compiled']['source_SHA256']and sha(q['compiled']['MD_path'])==q['compiled']['MD_SHA256']
 for name in ['config.json','model.safetensors.index.json']:assert sha(M/name)==q['checkpoint_metadata_SHA256'][name]
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
 for n in ['glm_attention_create_r398','glm_attention_query_r398','glm_attention_compare_v_r398','glm_attention_control_r398','glm_attention_destroy_r398','glm_layer_id_r5670','glm_attention_value_roundtrip_r433','modern_glm_fp8_runtime_stats_r809','glm_attention_diag_arm_r433','glm_attention_diag_desc_r433','glm_attention_diag_read_r433','glm_attention_diag_clear_r433','glm_fp8_create_r776']:
  info=Info();assert process.dladdr(ctypes.cast(getattr(lib,n),V),ctypes.byref(info));actual=Path(info.fname.decode()).resolve();target=Path('/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so')if n=='glm_fp8_create_r776'else Path(expected);assert actual==target.resolve();out[n]=str(actual)
  if n=='glm_fp8_create_r776':assert sha(actual)=='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
 return out

def worker(rank,uid,start,physical):
 worker_enter=time.time()
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32 # Runtime apply_once must see all8x4 CPUs; no prepin
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for n in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=R/'results'/('GLM433-'+step+'-rank%d.json'%rank)
 assert not path.exists();q=dict(completed=False,normal=False,actual_step=step,rank=rank,node=os.uname().nodename,PID=os.getpid(),parent_PID=os.getppid(),physical_PCI=physical[rank],CPU_affinity_entry=cpus,GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),records=[],scope='Layer38realcheckpoint completeNoV attention; twofixedU2048selections/same32Ksynthetichidden, 12queries; originalrowFP8roundtrip/F32valueassociation, noModel/indexer/online; allcompletedoutputsrawsavedbeforeSTOP; formaldiagOFF',profile=0,Model=False,indexer_included=False,ordinary_host_selection_mapping_included=True,allocator_actual_peak='UNKNOWN; recorded snapshots are not peaks',numeric_policy=dict(alert_NRMSE=NUMERIC_ALERT_NRMSE,diagnostic_stop_NRMSE=NUMERIC_DIAGNOSTIC_STOP_NRMSE,scope='Same synthetic geometry numerical diagnostic; not task quality or acceptance; no bitwise criterion'))
 def save():
  def clean(v):
   if isinstance(v,float)and not math.isfinite(v):return None
   if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
   if isinstance(v,(list,tuple)):return [clean(x)for x in v]
   return v
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(q),indent=2,allow_nan=False));tmp.replace(path)
 def update(records,error):q['records']=records;q['partial_error']=error;save()
 def limit():
  if time.time()>=start+540:raise TimeoutError('r433 original epoch540 inner gate; GPU600 total; no retry')
 save();ctx=None;leaf=None;lib=None
 try:
  limit();a,m=verify();assert pci(rank)==physical[rank]
  props=torch.cuda.get_device_properties(rank);q['gcnArchName']=props.gcnArchName;q['actual_compute_units']=props.multi_processor_count;q['actual_total_memory']=props.total_memory;assert props.gcnArchName.split(':')[0]=='gfx936'
  q['library_SHA256']=a['library_SHA256'];lib=api(a['library']);q['actual_symbol_owners']=symbol_owners(lib,a['library'])
  maps=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if line.split()[-1].startswith('/')and'.so'in line.split()[-1]})
  q['actual_mapped_SDK']={p:sha(p)for p in maps if any(x in Path(p).name for x in ['infini','cudart','blas_policy','fp8_batched'])}
  for p,h in a['actual_dynamic_dependency_SHA256'].items():
   expected=Path(p).resolve()
   if any(x in expected.name for x in ['infini','cudart']):
    matches=[Path(x).resolve()for x in maps if Path(x).resolve().name==expected.name]
    assert matches and all(x==expected and sha(x)==h for x in matches),(str(expected),list(map(str,matches)))
  cfg=json.loads((M/'config.json').read_bytes());uidbuf=ctypes.create_string_buffer(uid,len(uid));beg=time.perf_counter();ctx=lib.glm_attention_create_r398(rank,json.dumps(cfg).encode(),uidbuf);assert ctx,lib.glm_attention_error_r398();q['create_wall_s']=time.perf_counter()-beg;q['worker_python_to_after_create_s']=time.time()-worker_enter;q['actual_owner_python_thread']=threading.get_ident()
  q['CPU_affinity_after_create']=sorted(os.sched_getaffinity(0));save()
  keys=json.loads(lib.glm_attention_keys_r398(ctx));assert sorted(keys)==KEYS
  def check(rc):
   if rc:raise RuntimeError(lib.glm_attention_error_r398().decode())
  beg=time.perf_counter();q['load']=load(lib,ctx,keys,Checkpoint(M,torch),check,limit);q['load_wall_s']=time.perf_counter()-beg;q['free_after_weightload']=torch.cuda.mem_get_info(rank)[0];assert q['free_after_weightload']>=4*1024**3
  limit();beg=time.perf_counter();generator=torch.Generator(device='cpu').manual_seed(398038)
  cpu=(torch.randn((32768,6144),generator=generator,dtype=torch.bfloat16)*.1).contiguous();input_hash=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest();hidden=cpu.cuda();del cpu;torch.cuda.synchronize(rank)
  q.update(synthetic_hidden_SHA256=input_hash,fixture_prepare_wall_s=time.perf_counter()-beg,input_shape=[32768,6144],actual_checkpoint_payload_loaded=True,wholeModel_loaded=False,hidden_is_Model_activation=False,selected_indices_are_Model_routing=False);save()
  lib.modern_owner_affinity_report_r2392.restype=ctypes.c_char_p
  q['CPU_affinity_after_load']=sorted(os.sched_getaffinity(0));q['owner_affinity_before_queries']=json.loads(lib.modern_owner_affinity_report_r2392());save()
  def first_submit_snapshot():
   report=json.loads(lib.modern_owner_affinity_report_r2392());mask=sorted(os.sched_getaffinity(0));mine=[v for v in report['owners']if v['device']==rank]
   q['first_successful_seed_affinity']=dict(actual_owner_report=report,actual_mask=mask,expected_mask=cpus[4*rank:4*rank+4]);save()
   assert len(mine)==1 and mine[0]['changes']==1 and list(map(int,mine[0]['verified_mask'].split(',')))==mask==cpus[4*rank:4*rank+4]
   return report
  raw_directory=R/'results'/('GLM433-'+step+'-rank%d-raw'%rank);raw_directory.mkdir(exist_ok=False);q['raw_directory']=str(raw_directory);save()
  before=time.perf_counter();leaf=NativeLeaf(lib,ctx,torch,hidden,start+540,raw_directory,rank);leaf.on_first_seed=first_submit_snapshot;records=consume(leaf,update);ctx=None;q['whole_component_sequence_wall_s']=time.perf_counter()-before
  assert len(records)==12 and sum(r['warm']for r in records)==4 and sum(not r['warm']for r in records)==8 and all(r['Native_CAPI_completed']and r['qualification_passed']and r['raw_outputs_complete']for r in records)
  q['owner_affinity_after_queries']=json.loads(lib.modern_owner_affinity_report_r2392())
  def masks(v):return sorted((x['device'],x['changes'],x['verified_mask'])for x in v['owners'])
  assert masks(q['first_successful_seed_affinity']['actual_owner_report'])==masks(q['owner_affinity_after_queries'])
  mine=[v for v in q['owner_affinity_after_queries']['owners']if v['device']==rank];assert len(mine)==1 and mine[0]['changes']==1 and len(mine[0]['verified_mask'].split(','))==4
  q['CPU_affinity']=sorted(os.sched_getaffinity(0));assert q['CPU_affinity']==cpus[4*rank:4*rank+4]
  readonly_check_start=time.perf_counter()
  assert hashlib.sha256(hidden.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()==input_hash
  q['readonly_full_hidden_check_s']=time.perf_counter()-readonly_check_start;q['readonly_full_hidden_check_bytes']=32768*6144*2
  del hidden;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(rank);t=torch.ones(32,device='cuda');assert(t+1).sum().item()==64;torch.cuda.synchronize(rank)
  q.update(FP8_runtime_final_stats=leaf.final_released_stats,completed=True,normal=True,normal_destroy=True,Runtime_weak_expired=True,Torch_after_destroy=True,policy_closed=True,measurement_calls=8,warm_calls=4,directV_diagnostics=0,query_calls=12,seed_calls=780,NoV_fixed=True,row_roundtrip_value_association_only_variable=True,diagnostic_snapshots=2,all_completed_query_raw_outputs=True,weights_loaded_once=True,input_immutable=True,free_postclose=torch.cuda.mem_get_info(rank)[0],owner_service_residency_s=time.time()-start);save();print('GLM433_NORMAL '+str(rank),flush=True)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc());save();raise
 finally:
  if leaf is not None and leaf.destroy_attempted:q['destroy_attempted_once']=True;q['destroy_RC']=getattr(leaf,'destroy_rc',None);q['Runtime_expired']=getattr(leaf,'runtime_expired',False);save()
  if ctx and lib and (leaf is None or not leaf.destroy_attempted):
   flag=I();lib.glm_attention_destroy_r398.argtypes=[V,ctypes.POINTER(I)]
   try:q['failure_destroy_RC']=int(lib.glm_attention_destroy_r398(ctx,ctypes.byref(flag)));q['failure_Runtime_expired']=bool(flag.value)
   except BaseException as e:q['failure_destroy_error']=repr(e)
   save()

if __name__=='__main__':
 parent_enter=time.time()
 import torch,torch.multiprocessing as mp
 a,m=verify();assert torch.cuda.device_count()==8;torch.cuda.set_device(0);physical=[pci(i)for i in range(8)];assert len(set(physical))==8
 lib=api(a['library']);uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0
 start=float(os.environ['GLM433_GPU_START']);assert time.time()<start+540
 main_record=R/'results'/('GLM433-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-main.json')
 main_record.write_text(json.dumps(dict(PID=os.getpid(),original_epoch=start,parent_python_to_pre_spawn_s=time.time()-parent_enter,seconds_since_GPU_epoch=time.time()-start,GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),physical_PCI=physical)))
 mp.spawn(worker,args=(bytes(uid),start,physical),nprocs=8,join=True);print('GLM433_ALL8_NORMAL',flush=True)
