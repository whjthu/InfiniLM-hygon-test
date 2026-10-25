"""CPU-prepared finite layer38 component; no execution without root's concrete GPU GO.
One native context/weight load per owner. Synthetic replicated BF16 hidden.
Ordinary wholelayer wall includes producer/proof/fallback CPU work; no Model TPS.
"""
import pathlib,json,ctypes,os,time,hashlib,contextlib,statistics,traceback
import re
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');base=r/'candidates/glm-prefill-layer-r8998';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
LIBRARY=pathlib.Path(json.loads((base/'CPU-build.json').read_text())['library'])
DSO_SHA='a0394bdda8c05b6a2b9d97df384803ebd3acabba9b98ac047ebcce3aaf5fa13c'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def api():
 lib=ctypes.CDLL(str(LIBRARY))
 lib.glm_layer_id_size_r5670.restype=Z;lib.glm_layer_id_r5670.argtypes=[V]
 lib.glm_layer_create_r5670.argtypes=[I,ctypes.c_char_p,V];lib.glm_layer_create_r5670.restype=V
 for name in ['error','report']:getattr(lib,'glm_layer_'+name+'_r5670').restype=ctypes.c_char_p
 lib.glm_layer_keys_r5670.argtypes=[V];lib.glm_layer_keys_r5670.restype=ctypes.c_char_p
 lib.glm_layer_load_r5670.argtypes=[V,ctypes.c_char_p]
 for name in ['finish','reset']:getattr(lib,'glm_layer_'+name+'_r5670').argtypes=[V]
 lib.glm_layer_forward_r5670.argtypes=[V,Z,Z,ctypes.POINTER(Z),V,V,V,I]
 lib.glm_layer_destroy_r5670.argtypes=[V,ctypes.POINTER(I)];lib.glm_layer_control_r5670.argtypes=[V,V]
 lib.modern_glm_dense_mode_r48_set.argtypes=[I]
 lib.modern_glm_prefill_prefix_plan_counts_r8703.argtypes=[I,I];lib.modern_glm_prefill_prefix_plan_counts_r8703.restype=ctypes.c_uint64
 lib.modern_glm_vector_policy_r8540.argtypes=[I,I]
 lib.glm_grouped_counts_r5123.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]
 return lib

def worker(rank,unique):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8])
 p=r/'results'/f'GLM-prefill-layer-r8998-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists()
 q={'completed':False,'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'library_SHA256':DSO_SHA,'layer':38,'API':'existing glm_layer_*_r5670','ordinary_metric':'wholelayer wall, not pureattention; includes proof/fallback CPU validation','source_input_scope':'Synthetic CPU-generated BF16 hidden replicated across TP8; real layer38 weights loaded once. No Model token/TPS/adoption claim.','CPU_affinity':sorted(os.sched_getaffinity(0)),'candidate':'A currentgroupedOFF/routed19; B guardedgroupedON/grid256; C samegrouped/grid1024. ObserverOFF/nativewholeprefixlayer/TP8CCL; originalSiTU/drains/prefix/vector fixed','runtime_floor_bytes':512*1024**2,'floor_provenance':'561 runtime512MiB floor, distinct from6208MiB preModeladmission; original5736GiB singlelayer postloadadmission preserved','fixture_admission_bytes':6*1024**3,'trials':[],'decode_fallback':[],'irregular_partial_selection_GPU':'NOT_SUPPORTED_BY_EXISTING_API_CPU_ONLY_4006_DIFFERENTIAL'}
 ctx=None;lib=None
 def save():temp=p.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(p)
 def ck(code):
  if code:raise RuntimeError((code,lib.glm_layer_error_r5670().decode()))
 def stats():
  a=(ctypes.c_uint64*5)();b=(ctypes.c_uint64*5)();c=(ctypes.c_uint64*4)();ck(lib.glm_grouped_counts_r5123(rank,a,5));assert helper.glm_prefill_grid_counts_r8998(rank,b,5)==0;assert helper.glm_certified_stats_r5248(ctypes.cast(c,V),4)==0;return list(a),list(b),list(c)
 def agree(value):
  vote=torch.tensor([float(bool(value))],dtype=torch.float32,device='cuda');torch.cuda.synchronize();ck(lib.glm_layer_control_r5670(ctx,V(vote.data_ptr())));assert vote.item()==8,('unanimous_quality',rank,vote.item())
 try:
  proof=json.loads((base/'CPU-build.json').read_text());assert proof['completed']and proof['library_SHA256']==DSO_SHA==sha(proof['library']);assert proof['compiled_TUs']==1 and len(proof['reused_native_objects_SHA256'])==28 and len(proof['reused_GPU_objects_SHA256'])==6
  manifest=json.loads((r/'incoming/GLM-prefill-layer-manifest-r8998.json').read_text())
  for rel,h in manifest['DSO_SHA256'].items():assert sha(r/rel)==h,('runtime_DSO_SHA',rel)
  lib=api();assert lib.modern_glm_dense_mode_r48_set(2)==0;ck(lib.modern_glm_vector_policy_r8540(rank,1));os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='1'
  policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p;assert json.loads(policy.modern_blas_policy_r243_json())['failures']==0
  class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
  process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];process.dladdr.restype=I
  owners={}
  helper=ctypes.CDLL(proof['helper']);assert sha(proof['helper'])==proof['helper_SHA256']=='672d785fb8b1e8803a16853a1e6a79f1a20fda5482bbc76b1be692e436fc8d52';helper.glm_prefill_grid_policy_r8998.argtypes=[I,I];helper.glm_prefill_grid_counts_r8998.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z];helper.glm_certified_stats_r5248.argtypes=[V,Z];assert helper.glm_prefill_grid_resolve_r8998()==0;assert helper.glm_prefill_grid_policy_r8998(rank,99)!=0
  for handle,names,expected in [(lib,['glm_layer_create_r5670','glm_layer_forward_r5670','modern_glm_dense_mode_r48_set','modern_glm_prefill_prefix_plan_counts_r8703'],proof['library']),(lib,['glm_grouped_down_r5123','glm_certified_repair_r5248'],proof['helper']),(helper,['cudaLaunchKernel','glm_prefill_grid_policy_r8998','glm_prefill_grid_counts_r8998'],proof['helper'])]:
   for sym in names:
    inf=DlInfo();assert process.dladdr(ctypes.cast(getattr(handle,sym),V),ctypes.byref(inf));actual=pathlib.Path(inf.fname.decode()).resolve();assert actual==pathlib.Path(expected).resolve();owners[sym]={'file':str(actual),'SHA256':sha(actual)}
  q['actual_symbol_owners']=owners;q['free_before_create']=torch.cuda.mem_get_info()[0]
  cfg=json.loads((model/'config.json').read_text());assert cfg['mlp_layer_types'][38]=='sparse'and cfg['indexer_types'][38]=='full';cfg['probe_layers_r5680']=[38]
  started=time.monotonic();uid=ctypes.create_string_buffer(unique,len(unique));ctx=lib.glm_layer_create_r5670(rank,json.dumps(cfg).encode(),uid);assert ctx,lib.glm_layer_error_r5670();q['create_seconds']=time.monotonic()-started
  keys=json.loads(lib.glm_layer_keys_r5670(ctx));index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert len(keys)==len(set(keys))and all(k.startswith('38.')and 'model.layers.'+k in index for k in keys)
  loadedbytes=0;started=time.monotonic()
  with contextlib.ExitStack()as stack:
   handles={}
   for first in range(0,len(keys),24):
    held=[];desc=[]
    for key in keys[first:first+24]:
     actual='model.layers.'+key;name=index[actual]
     if name not in handles:handles[name]=stack.enter_context(safe_open(str(model/name),framework='pt',device='cpu'))
     x=handles[name].get_tensor(actual)
     if x.dtype in[torch.float8_e4m3fn,torch.float8_e5m2]:x=x.view(torch.uint8)
     assert x.dtype in[torch.float32,torch.bfloat16,torch.uint8];x=x.contiguous();held.append(x);loadedbytes+=x.numel()*x.element_size()
     desc.append({'name':key,'shape':list(x.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[x.dtype],'pointer':x.data_ptr()})
    ck(lib.glm_layer_load_r5670(ctx,json.dumps(desc).encode()))
  ck(lib.glm_layer_finish_r5670(ctx));q.update(load_seconds=time.monotonic()-started,global_source_bytes_per_owner=loadedbytes,only_one_weight_load=True,free_after_load=torch.cuda.mem_get_info()[0]);agree(q['free_after_load']>=6*1024**3);save()
  generator=torch.Generator(device='cpu').manual_seed(8703);x=(torch.randn((4096,6144),generator=generator,dtype=torch.float32)*.1).to(torch.bfloat16).cuda();pos=torch.arange(512,dtype=torch.int64).repeat(8).cuda();guard=torch.full((4096*6144+128,),123,dtype=torch.bfloat16,device='cuda');out=guard[64:-64].view(4096,6144);desc=(Z*24)(*[v for row in range(8)for v in [row,0,512]]);saved_x=x.clone();saved_pos=pos.clone()
  q['fixture_provenance']={'producer':'Original585/573 seed8703 replicatedBF16 prefixN4096/B8/past0/count512; current561 chunk4096/routes32768','current_Model_capture':False,'weight_checkpoint_read_passes':1,'accepted_parent':'56168acfe all28objects unchanged; unique5386helper HOSTTU/6originalGPUobjects'}
  reference=None;own_reference={};q['trials']=[];modes=[0,1,2,0,1,2,2,1,0,2,0,1,1,0,2,0]
  for index,mode in enumerate(modes):
   os.environ['GLM_GROUPED_ROUTED_R5123']=str(int(mode>0));assert helper.glm_prefill_grid_policy_r8998(rank,int(mode==2))==0;ck(lib.glm_layer_reset_r5670(ctx));before=stats();torch.cuda.synchronize();tick=time.perf_counter();ck(lib.glm_layer_forward_r5670(ctx,4096,8,desc,V(x.data_ptr()),V(pos.data_ptr()),V(out.data_ptr()),0));wall=(time.perf_counter()-tick)*1000;report=json.loads(lib.glm_layer_report_r5670());after=stats();delta=[[b-a for a,b in zip(v,w)]for v,w in zip(before,after)];branch,grid,cert=delta;expected=[2,1,1,0,0]if mode else[2,0,0,2,0];healthy=branch==expected and grid[4]==0 and (grid==[0]*5 if mode==0 else grid[0]==1 and grid[1]>0 and grid[2]==grid[1]and grid[3]==(grid[1]if mode==2 else 0)) and (cert[2]>0 if mode else cert==[0]*4)
   actual=out.cpu();reference=actual.clone()if reference is None else reference;own_reference.setdefault(mode,actual.clone());diff=actual.float()-reference.float();od=actual.float()-own_reference[mode].float();finite=bool(torch.isfinite(actual.float()).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());own_exact=bool(torch.equal(actual.view(torch.int16),own_reference[mode].view(torch.int16)));readonly=bool(torch.equal(x.view(torch.uint8),saved_x.view(torch.uint8))and torch.equal(pos,saved_pos));agree(healthy and finite and canary and readonly and own_exact and report['profile']==0 and report['N']==4096 and report['B']==8 and report['layers']==[38]and not report['stages'])
   q['trials'].append({'index':index,'mode':mode,'label':'warm'if index<3 else'restore'if index==15 else'balanced','wholelayer_native_wall_ms':report['wall_ms'],'wholelayer_driver_CAPI_ms':wall,'owner_threadCPU_ms':report['threadCPU_ms'],'branch_delta':branch,'grid_delta':grid,'certificate_delta':cert,'observer_events':0,'finite':finite,'canary':canary,'inputs_readonly':readonly,'ownrepeat_exact_diagnostic':own_exact,'ownrepeat_maxabs':float(od.abs().max()),'ownrepeat_RMSE':float(od.square().mean().sqrt()),'crossarm_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),reference.view(torch.int16))),'maxabs_vs_A':float(diff.abs().max()),'RMSE_vs_A':float(diff.square().mean().sqrt()),'relative_RMS_vs_A':float(diff.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-30)),'output_SHA256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),'free_after':torch.cuda.mem_get_info()[0]});save()
  q['ordinary_summary']={str(mode):{metric:statistics.median(a[metric]for a in q['trials'][3:15]if a['mode']==mode)for metric in ['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms']}for mode in [0,1,2]};os.environ['GLM_GROUPED_ROUTED_R5123']='0';assert helper.glm_prefill_grid_policy_r8998(rank,0)==0;ck(lib.glm_layer_reset_r5670(ctx));expired=I();ck(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1;q['final_provider_stats']=stats();torch.cuda.synchronize();assert (torch.ones(32,device='cuda')+1).sum().item()==64;torch.cuda.synchronize();q['blas_policy_after']=json.loads(policy.modern_blas_policy_r243_json());assert q['blas_policy_after']['failures']==0;q.update(completed=True,Runtime_weak_expired=True,normal_reset_close=True,Torch_postclose=True,all_outputs_finite=True,inputs_readonly=True,Model_adoption=False);save()
 except BaseException as exc:q['error']=repr(exc);q['traceback']=traceback.format_exc();save();raise
 finally:
  if ctx and lib:
   expired=I();q['finally_destroy_attempted']=True
   try:q['finally_destroy_code']=int(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));q['finally_Runtime_expired']=bool(expired.value)
   except BaseException as exc:q['finally_error']=repr(exc)
   save()

if __name__=='__main__':
 assert torch.cuda.device_count()==8;assert sha(LIBRARY)==DSO_SHA;torch.cuda.set_device(0);lib=api();uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0;mp.spawn(worker,args=(bytes(uid),),nprocs=8,join=True)
