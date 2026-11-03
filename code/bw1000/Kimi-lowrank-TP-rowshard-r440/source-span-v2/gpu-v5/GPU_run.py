"""One actual layer1 complete MoE, TP16 across two eight-owner nodes, no Model."""
from pathlib import Path
import ctypes as C,gc,hashlib,importlib.util,json,os,socket,sys,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_policy(environ,hostname):
 node=int(environ['SLURM_PROCID']);assert node in[0,1]and hostname==['a02r3n03','a02r3n07'][node]
 assert environ['SLURM_JOB_ID']=='910371'
 fixed=dict(INFINILM_OWNER_CPU_WIDTH_R2392='4',INFINILM_KIMI_SPEC_MATH_R3676='0',INFINILM_KIMI_MOE_WORKSPACE_R4665='1',INFINILM_KIMI_MOE_EXECUTION_R4665='0',INFINILM_KIMI_I192_GROUPED_R8669Y='0',INFINILM_KIMI_I192_COMPACT_R8669ZC='1',INFINILM_KIMI_BATCH_I192_R9125='1',INFINILM_KIMI_PACKED_W2_R6122='0',INFINILM_KIMI_PACKED_AUTO_R6136='0')
 for k,v in fixed.items():assert environ.get(k)==v,(k,environ.get(k),v)
 assert all(environ.get(k)is None for k in ['INFINILM_KIMI_PROFILE_R9125','INFINILM_KIMI_MOE_AUX_M1_R4430','INFINILM_KIMI_ROUTER_M1_R4635'])
 return dict(node=node,host=hostname,TP=16,PP=1,C=1,grouped=True,tiled_W13=False,Runtime_device_tested=False)
def publish(path,q):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
 node=int(os.environ['SLURM_PROCID']);step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];start=time.monotonic();epoch=float(os.environ['KIMI_R440_GPU_START']);deadline=epoch+540
 out=R/'results'/('Kimi440-'+step+'-node%d.json'%node);assert not out.exists()
 q=dict(completed=False,passed=False,actual_step=step,PID=os.getpid(),node=socket.gethostname().split('.')[0],node_rank=node,source_manifest_SHA256=os.environ['KIMI_R440_GPU_SOURCE_SHA'],original_epoch=epoch,Model=False,HTTP=0,phase='source_verify',normal_close=False)
 publish(out,q);held=[]
 try:
  assert os.environ['SLURM_JOB_ID']=='910371'and q['node']==['a02r3n03','a02r3n07'][node]
  q['actual_policy']=validate_policy(os.environ,q['node'])
  mf=D/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==q['source_manifest_SHA256'];m=json.loads(mf.read_bytes())
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  parent=Path(m['parent_source_root']);assert sha(parent/'SOURCE_MANIFEST.json')==m['parent_source_SHA256']
  for p,h in json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(parent/p)==h,p
  receipt=Path(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];cpu=json.loads(receipt.read_bytes());assert cpu['passed']and cpu['actual_CPU']=='910407.912'and sha(cpu['library'])==m['library_SHA256']
  for v in cpu['compiled']:
   assert sha(v['object'])==v['object_SHA256']and sha(v['MD'])==v['MD_SHA256']
   for p,h in v['dependency_SHA256'].items():assert sha(p)==h,p
  assert cpu['reused_CPU910_pure_contract']['result']['passed']and not cpu['old432GPU_linked']
  q['entry_affinity']=sorted(os.sched_getaffinity(0));assert len(q['entry_affinity'])>=32
  q['phase']='actual_checkpoint_and_device';publish(out,q)
  # This is an actual allocated GPU job, never a hidden-GPU import probe.
  import numpy as np
  import torch
  from safetensors import safe_open
  torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
  q['device_properties']=[str(torch.cuda.get_device_properties(i))for i in range(8)]
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12',mode=C.RTLD_GLOBAL);cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int]
  PCI=[]
  for device in range(8):
   buf=C.create_string_buffer(32);assert cuda.cudaDeviceGetPCIBusId(buf,32,device)==0;PCI.append(buf.value.decode().lower())
  assert len(set(PCI))==8;q['physical_PCI_by_visible_device']=PCI;q['visible_count']=8
  q['memory_before_component']=[list(torch.cuda.mem_get_info(i))for i in range(8)]
  # No extra owner probe: original firstsubmit alone narrows the complete entry pool.
  library=Path(cpu['library']);assert sha(library)==m['library_SHA256'];lib=C.CDLL(str(library),mode=C.RTLD_LOCAL)
  q['owner_policy']='entry32+ through Runtimecreate; original firstsubmit narrows; exact4 verified after each completed forward; restore only after close'
  spec=importlib.util.spec_from_file_location('kimi_r440_actual_consumer',parent/'consumer.py');consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
  raw_input,info=consumer.load_input(R,json.loads((parent/'FIXTURE_CARD.json').read_bytes()));assert info['SHA256']=='3f84e8d80178c05212edcc5b703f907290df20925684fb610838c6d1f259ecca'
  x=np.frombuffer(raw_input,dtype='<u2').copy();assert x.size==7168;q['input']=info
  model=Path('/shared/Kimi-K3');index=model/'model.safetensors.index.json';assert sha(index)==m['checkpoint_index_SHA256'];mapping=json.loads(index.read_bytes())['weight_map'];cfg=json.loads((model/'config.json').read_bytes());assert sha(model/'config.json')==m['checkpoint_config_SHA256']
  prefix='language_model.model.layers.1.block_sparse_moe.';groups={}
  for key,shard in mapping.items():
   if key.startswith(prefix):groups.setdefault(shard,[]).append(key)
  assert groups
  descriptors=[];weight_proof={};loaded=time.monotonic()
  for shard,keys in sorted(groups.items()):
   assert time.time()<deadline,'Original540s worker bound includes CPU checkpoint'
   with safe_open(model/shard,framework='pt',device='cpu')as weights:
    for key in sorted(keys):
     tensor=weights.get_tensor(key).clone().contiguous();assert tensor.dtype in[torch.float32,torch.bfloat16,torch.uint8]
     held.append(tensor);descriptors.append(dict(name=key[len(prefix):],pointer=tensor.data_ptr(),shape=list(tensor.shape),dtype={torch.float32:0,torch.bfloat16:1,torch.uint8:2}[tensor.dtype]))
     data=tensor.view(torch.uint8).numpy();weight_proof[key]=dict(SHA256=hashlib.sha256(memoryview(data).cast('B')).hexdigest(),shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=tensor.numel()*tensor.element_size(),shard=shard)
  q['projection_checkpoint_identity']={}
  for role,shape in [('down',[3584,7168]),('up',[7168,3584])]:
   key=prefix+'routed_expert_'+role+'_proj.weight';v=weight_proof[key];assert v['shape']==shape and all(v[k]==expected for k,expected in m['projection_checkpoint_expected'][role].items());q['projection_checkpoint_identity'][role]=dict(checkpoint_key=key,**v)
  q['replication_evidence']='Same full BF16 checkpoint key/source bytes in all eight actual owner payloads per node; original ReplicatedLinear no-TP-split load. No GPU weight readback checksum is claimed.'
  q.update(CPU_checkpoint_load_s=time.monotonic()-loaded,CPU_checkpoint_bytes=sum(v.numel()*v.element_size()for v in held),checkpoint_weights=weight_proof,checkpoint_key_count=len(held))
  run=lib.modern_kimi_lowrank_full_moe_r440;run.argtypes=[C.c_char_p];run.restype=C.c_int;result=lib.modern_kimi_lowrank_full_moe_result_r440;result.argtypes=[];result.restype=C.c_char_p
  class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
  dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];bindings={}
  for name in ['modern_kimi_lowrank_full_moe_r440','modern_kimi_lowrank_full_moe_result_r440','modern_kimi_owner_affinity_preflight_r440']:
   v=DlInfo();assert dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(v));actual=v.name.decode();assert sha(actual)==m['library_SHA256'];bindings[name]=dict(path=actual,SHA256=sha(actual))
  q['actual_CABI_bindings']=bindings;q['mapped_libraries']={}
  for line in Path('/proc/self/maps').read_text().splitlines():
   path=line.split()[-1]
   if path.startswith('/')and any(tag in path for tag in ['libKimi_lowrank_TP_rowshard','libinfiniops.so','libinfinilm_runtime.so','libkimi_grouped_prefill.so','_infinilm.cpython','libinfinirt.so','libinfiniccl.so']):q['mapped_libraries'][path]=sha(path)
  for expected,h in m['mapped_dependencies_SHA256'].items():assert any(Path(path).resolve()==Path(expected).resolve()and value==h for path,value in q['mapped_libraries'].items()),expected
  rawdir=R/'results'/('Kimi440-'+step+'-node%d-raw'%node);rawdir.mkdir(exist_ok=False)
  payload=consumer.payload(cfg,descriptors,int(x.ctypes.data),str(rawdir),node,os.environ['KIMI_R440_MASTER_ADDR'],m['master_port'])
  q.update(phase='complete_MoE_OFF_ON_warm_then_ABBA',raw_directory=str(rawdir),actual_C1=True,TP=16,PP=1,long_factory=False,padded_scale_storage=True);publish(out,q)
  assert time.time()<deadline;begin=time.monotonic();rc=run(payload);q['complete_Native_session_s']=time.monotonic()-begin
  q['CAPI_result']=json.loads(result());q['returncode']=rc;q['phase']='CAPI_returned';publish(out,q)
  assert q['complete_Native_session_s']<=300,'Measurement qualification budget exceeded; original worker/shell bound unchanged'
  assert rc==0 and q['CAPI_result']['completed']and q['CAPI_result']['normal_CommunicationGroup_close']
  rows=q['CAPI_result']['reports'];assert len(rows)==8 and all(row['completed']and row['normal_Runtime_close']and row['both_modules_closed']for row in rows)
  q['projection_scope_summaries']=[dict(local_rank=row['local_rank'],global_tp_rank=row['global_tp_rank'],firstON=row['arms'][1])for row in rows]
  q['raw_outputs']=[]
  for rank,row in enumerate(rows):
   assert row['local_rank']==rank and row['global_tp_rank']==node*8+rank and row['TP']==16 and row['PP']==1
   assert row['entry_mask']==q['entry_affinity']and row['pinned_mask']==row['owner_mask']==q['entry_affinity'][rank*4:rank*4+4]and row['affinity_restored']and row['restored_mask']==q['entry_affinity']
   assert[v['mode']for v in row['arms']]==[0,1,0,1,1,0]
   for arm,record in enumerate(row['arms']):
    p=rawdir/('rank%d-arm%d.bf16'%(rank,arm));raw=p.read_bytes();assert len(raw)==7168*2
    values=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32);finite=bool(np.isfinite(values).all());rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)))
    q['raw_outputs'].append(dict(rank=rank,arm=arm,path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=[1,1,7168],dtype='BF16',endian='little',finite=finite,RMS=rms));publish(out,q)
    assert finite and rms>0 and record['nonfinite']==0 and record['input_and_tail_canary_unchanged']
    assert record['down_up_selected_delta']==[record['mode'],record['mode']]and record['pending_Tensor_handles_before_completion']==(14 if record['mode']else 0)and record['existing_completion_and_lease_release_timed']
  assert len(q['raw_outputs'])==48
  q['memory_after_component']=[list(torch.cuda.mem_get_info(i))for i in range(8)];q['memory_scope']='memGetInfo snapshots only, not peak or loaded Model fit'
  policy=C.CDLL(None).modern_blas_policy_r243_json;policy.restype=C.c_char_p;q['BLAS_policy']=json.loads(policy());assert q['BLAS_policy']['failures']==0 and len(q['BLAS_policy']['owners'])==8 and all(v['created']==v['verified']for v in q['BLAS_policy']['owners'])
  after={key:hashlib.sha256(memoryview(v.view(torch.uint8).numpy()).cast('B')).hexdigest()for key,v in zip(weight_proof,held)};assert after=={key:v['SHA256']for key,v in weight_proof.items()}
  assert hashlib.sha256(x.tobytes()).hexdigest()==info['SHA256'];q['CPU_weights_and_input_unchanged']=True
  held.clear();descriptors.clear();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();q.update(passed=True,normal_close=True,phase='normal8_and_group_closed',numeric_scope='Full MoE relative/finite/canary on one actual historical input and actual layer1 checkpoint; not Model or task quality')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc(),passed=False)
 finally:q.update(completed=True,complete_process_wall_s=time.monotonic()-start,driver_observer_cost='Weight hashing/seed/metadata/load/full output readback/save and validation separately included in processwall; primary native per-arm interval is original forward+completion');publish(out,q)
 print(json.dumps({k:q.get(k)for k in ['actual_step','node_rank','passed','normal_close','error']}),flush=True);return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
