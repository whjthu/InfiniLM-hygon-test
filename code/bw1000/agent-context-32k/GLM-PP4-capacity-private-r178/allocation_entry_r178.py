"""32-owner original Native constructor capacity probe, never weights/forward."""
import pathlib,json,hashlib,os,re,time,base64,ctypes,traceback
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Explicit publication/CPU/assigned-job GO before importing Torch or a Native DSO.
go_path=pathlib.Path(os.environ['GLM_ALLOCATION_GO_R178']);assert go_path.parent==R/'control'
go=json.loads(go_path.read_bytes());assert go['GPU_authorized']is True and go['operation']=='allocation_only_no_weights_no_forward'
assert go['branch']=='bw1000-experiments' and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])
assert go['source_manifest_SHA256']==sha(D/'source-manifest-r178.json')
assert go['CPU_receipt_SHA256']==sha(D/'CPU-result-r178.json')
cpu=json.loads((D/'CPU-result-r178.json').read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']
assert (go['TP'],go['PP'],go['rows'],go['capacity'])==(8,4,1,32768)
assert go['job_id']==os.environ['SLURM_JOB_ID'] and go['step_id']==os.environ['SLURM_STEP_ID']
assert os.environ['GLM_PROJECTED_HOST_R4504']=='0'and os.environ['GLM_NETWORK_TRIAL']=='shca'
assert (os.environ['GLM_PP_SIZE_R3603'],os.environ['GLM_CACHE_ROWS_R3627'],os.environ['GLM_CACHE_CAP_R3627'])==('4','1','32768')
assert len(os.sched_getaffinity(0))>=32
os.environ['GLM_DECODE_ROW_DEDUP_R5386']='0';os.environ['GLM_GROUPED_ROUTED_R5123']='0';os.environ['GLM_ASYNC_PROJECTED_HOST_R5011']='0';os.environ['GLM_GROUP_PROJECTED_HOST_R5030']='0';os.environ['GLM_BATCH_ALLSELECTED_R5096']='1'
from preflight_r133 import validate as pv
parent=pv()
from preengine_r133 import validate as ev
preengine=ev()
import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
stage=int(os.environ['SLURM_PROCID']);assert stage in range(4)
tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];bounds=[0,18,38,58,78]
baseline=(R/'incoming/modern_glm_full_model_probe_r2.py').read_text()
ns={};prefix=baseline[:baseline.index('def worker(rank,payload):')]
prefix=prefix.replace("r/'build/libmodern_glm_full_model_probe.so'",repr(parent['model_library_metadata']['library_actual_path']))
exec(prefix,ns)
lib,ptr,check,unique,model=[ns[k]for k in ['lib','ptr','check','unique','model']]
lib.modern_glm_model_shutdown.argtypes=[]
lib.modern_glm_partition_report_r3854.argtypes=[ctypes.c_void_p];lib.modern_glm_partition_report_r3854.restype=ctypes.c_char_p
lib.modern_glm_constructor_census_r4158.argtypes=[ctypes.c_void_p];lib.modern_glm_constructor_census_r4158.restype=ctypes.c_char_p
lib.modern_glm_unloaded_trim_r4052.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
lib.modern_glm_projected_host_counts_r4504.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
def write(p,q):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
def worker(rank,payload):
 torch.cuda.set_device(rank);torch.cuda.synchronize(rank)
 output=R/'results'/f'GLM-allocation-r178-{tag}-stage{stage}-rank{rank}.json';assert not output.exists()
 q={'completed':False,'stage':stage,'rank':rank,'TP':8,'PP':4,'rows':1,'capacity':32768,'actual_step':tag,'Native_SHA256':parent['model_library_metadata']['library_SHA256'],'source_manifest_SHA256':sha(D/'source-manifest-r178.json'),'CPU_receipt_SHA256':sha(D/'CPU-result-r178.json'),'weights_loaded':False,'forward_called':False,'finish_called':False,'constructor_allocates_parameter_storage':True,'policy_setters_not_called':True,'host_projected_bank_requested':False,'loaded_model_fit':None,'long_prefill_workspace_peak':None,'phase':'before_create','normal_destroy':False,'normal_shutdown':False,'memory_before_create':list(torch.cuda.mem_get_info(rank))}
 ctx=None;error=None
 try:
  assert q['memory_before_create'][0]>58*1024**3
  cfg=json.loads((model/'config.json').read_bytes())
  buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k]if k!='tp'else payload[k][stage]))for k in ['world','tp','control']]
  q['phase']='create';write(output,q)
  ctx=lib.modern_glm_model_create(rank,stage,1,32768,json.dumps(cfg).encode(),*buffers)
  if not ctx:raise RuntimeError('Native_constructor_failed: '+lib.modern_glm_model_error().decode())
  q['phase']='constructed';q['memory_after_create']=list(torch.cuda.mem_get_info(rank));write(output,q)
  raw=lib.modern_glm_partition_report_r3854(ctx);assert raw
  part=json.loads(raw);assert (part['PP'],part['stage'],part['device'],part['begin'],part['end'],part['weights_ready'])==(4,stage,rank,bounds[stage],bounds[stage+1],False)
  keys=part.pop('keys');q['actual_partition']=part;q['parameter_key_count']=len(keys);q['parameter_keys_SHA256']=hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()
  raw=lib.modern_glm_constructor_census_r4158(ctx);assert raw;census=json.loads(raw)
  # Only logical category totals/shape counts, never raw tensor bytes or addresses.
  q['constructor_logical_census']={k:census[k]for k in ['layer_begin','layer_end','rows','capacity','loaded','cache_category_bytes','cache_unique_start_bytes','weight_unique_start_bytes','weight_keys','scope']}
  h=(ctypes.c_uint64*7)();check(lib.modern_glm_projected_host_counts_r4504(rank,h,7));assert list(h)==[0]*7;q['host_projected_bank_stats']=list(h)
  stream=lambda:ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
  vote=torch.ones(1,device='cuda',dtype=torch.float32);check(lib.modern_glm_model_control(ctx,1,ptr(vote),stream()));assert vote.item()==32;q['world32_integer_SUM_exact']=True
  t=(ctypes.c_uint64*6)();check(lib.modern_glm_unloaded_trim_r4052(ctx,t,6));assert list(t)[3:]==[rank,stage,4]
  q['unloaded_trim']=list(t);assert lib.modern_glm_model_keys(ctx)==json.dumps(keys).encode() or set(json.loads(lib.modern_glm_model_keys(ctx)))==set(keys)
  torch.cuda.synchronize(rank);torch.cuda.empty_cache();q['memory_after_trim']=list(torch.cuda.mem_get_info(rank));q['constructor_memory_floor_512MiB_passed']=q['memory_after_trim'][0]>=512*1024**2
  vote.fill_(float(q['constructor_memory_floor_512MiB_passed']));check(lib.modern_glm_model_control(ctx,1,ptr(vote),stream()));q['all32_constructor_memory_floor_passed']=vote.item()==32
  q['phase']='destroy';write(output,q)
 except BaseException as exc:
  error=exc;q['error_class']=type(exc).__name__;q['error_phase']=q['phase'];q['diagnostic']=str(exc)[:384]
 finally:
  if ctx:
   try:check(lib.modern_glm_model_destroy(ctx));q['normal_destroy']=True
   except BaseException as exc:q['cleanup_error_class']=type(exc).__name__;error=error or exc
  try:check(lib.modern_glm_model_shutdown());q['normal_shutdown']=True
  except BaseException as exc:q['shutdown_error_class']=type(exc).__name__;error=error or exc
  try:
   torch.cuda.synchronize(rank);torch.cuda.empty_cache();q['memory_after_shutdown']=list(torch.cuda.mem_get_info(rank));q['Torch_postclose_liveness']=bool(torch.isfinite(torch.ones(8,device='cuda')).all());torch.cuda.synchronize(rank)
  except BaseException as exc:q['liveness_error_class']=type(exc).__name__;error=error or exc
  q['completed']=error is None and q.get('all32_constructor_memory_floor_passed',False)and q['normal_destroy']and q['normal_shutdown']and q.get('Torch_postclose_liveness',False);q['phase']='complete'if q['completed']else'failed';write(output,q)
 if error:raise error
 assert q['completed']
def guarded(rank,payload):
 try:worker(rank,payload)
 except BaseException:traceback.print_exc();os._exit(2)
if __name__=='__main__':
 p=R/'control'/f'GLM-allocation-r178-{tag}-ids.json'
 if stage==0:
  assert not p.exists();write(p,{'world':unique(),'tp':[unique()for _ in range(4)],'control':unique()})
 begin=time.monotonic()
 while not p.exists():
  if time.monotonic()-begin>120:raise TimeoutError('allocation control IDs missing')
  time.sleep(.1)
 mp.spawn(guarded,args=(json.loads(p.read_bytes()),),nprocs=8,join=True)
 print('GLM_ALLOCATION_R178_NORMAL',stage,flush=True)
