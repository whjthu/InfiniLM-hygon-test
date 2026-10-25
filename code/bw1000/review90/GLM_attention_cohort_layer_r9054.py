"""CPU-prepared finite layer38 component; no execution without root's concrete GPU GO.
One native context/weight load per owner. Synthetic replicated BF16 hidden.
Ordinary wholelayer wall includes producer/proof/fallback CPU work; no Model TPS.
"""
import pathlib,json,ctypes,os,time,hashlib,contextlib,statistics,traceback
import re
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');base=r/'candidates/glm-attention-cohort-r9054';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
LIBRARY=pathlib.Path(json.loads((base/'CPU-build.json').read_text())['library'])
DSO_SHA='45a3ad06b789152c532abcaa27a45db582bf55669ec4f7a56e6ba08b9de3003a'
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
 lib.glm_reduced_output_policy_r9032.argtypes=[I,I]
 lib.glm_reduced_output_counts_r9032.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]
 lib.glm_attention_cohort_policy_r9054.argtypes=[I,I]
 lib.glm_attention_cohort_counts_r9054.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]
 return lib

def worker(rank,unique):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8])
 p=r/'results'/f'GLM-attention_cohort-layer-r9054-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists()
 q={'completed':False,'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'library_SHA256':DSO_SHA,'layer':38,'API':'existing glm_layer_*_r5670','ordinary_metric':'wholelayer wall, not pureattention; includes proof/fallback CPU validation','source_input_scope':'Synthetic CPU-generated BF16 hidden replicated across TP8; real layer38 weights loaded once. No Model token/TPS/adoption claim.','CPU_affinity':sorted(os.sched_getaffinity(0)),'candidate':'Attention sameP cohortscale/Softmax/Cast only; QK/PV eachroworiginalshape/dtype/provider/defaultconfig; originalscale392BF16round; originalMoE; directPVfixedOFF; notModelqualified, originalSDK/ownership/finaldrains; observerOFF/vectorON/prefixON/GROUPED1/originaldrains','runtime_floor_bytes':512*1024**2,'floor_provenance':'561 runtime512MiB floor, distinct from6208MiB preModeladmission; original5736GiB singlelayer postloadadmission preserved','fixture_admission_bytes':6*1024**3,'trials':[],'decode_fallback':[],'irregular_partial_selection_GPU':'NOT_SUPPORTED_BY_EXISTING_API_CPU_ONLY_4006_DIFFERENTIAL'}
 ctx=None;lib=None
 def save():temp=p.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(p)
 def ck(code):
  if code:raise RuntimeError((code,lib.glm_layer_error_r5670().decode()))
 def stats():
  v=(ctypes.c_uint64*4)();ck(lib.glm_attention_cohort_counts_r9054(rank,v,4));return [v[0]+v[2],v[1]+v[3],0]
 def entry_stats():
  v=(ctypes.c_uint64*4)();ck(lib.glm_reduced_output_counts_r9032(rank,v,4));return [v[0],v[1],0,v[2],v[3],0]
 def agree(value):
  vote=torch.tensor([float(bool(value))],dtype=torch.float32,device='cuda');torch.cuda.synchronize();ck(lib.glm_layer_control_r5670(ctx,V(vote.data_ptr())));assert vote.item()==8,('unanimous_quality',rank,vote.item())
 try:
  proof=json.loads((base/'CPU-build.json').read_text());assert proof['completed']and proof['library_SHA256']==DSO_SHA==sha(proof['library']);assert proof['GPU_TUs_compiled_on_CPU']==0 and proof['HOST_TUs_compiled']==2 and len(proof['reused_objects_SHA256'])==26
  manifest=json.loads((r/'incoming/GLM-attention_cohort-layer-manifest-r9054.json').read_text())
  for rel,h in manifest['DSO_SHA256'].items():assert sha(r/rel)==h,('runtime_DSO_SHA',rel)
  lib=api();assert lib.glm_attention_cohort_policy_r9054(rank,99)!=0;ck(lib.glm_attention_cohort_policy_r9054(rank,0));assert lib.glm_reduced_output_policy_r9032(rank,99)!=0;assert lib.modern_glm_dense_mode_r48_set(2)==0;ck(lib.modern_glm_vector_policy_r8540(rank,1));ck(lib.glm_reduced_output_policy_r9032(rank,0));os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='1'
  policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p;assert json.loads(policy.modern_blas_policy_r243_json())['failures']==0
  class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
  process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];process.dladdr.restype=I
  owners={}
  for sym in ['glm_layer_create_r5670','glm_layer_forward_r5670','modern_glm_dense_mode_r48_set','modern_glm_prefill_prefix_plan_counts_r8703','modern_glm_shared_validation_counts_r5814','glm_vec_gate_r8528','glm_vec_down_r8528','glm_reduced_output_policy_r9032','glm_reduced_output_counts_r9032','glm_reduced_output_choose_r9032','glm_attention_cohort_policy_r9054','glm_attention_cohort_counts_r9054']:
   inf=DlInfo();assert process.dladdr(ctypes.cast(getattr(lib,sym),V),ctypes.byref(inf));actual=pathlib.Path(inf.fname.decode()).resolve();assert actual==pathlib.Path(proof['library']).resolve();owners[sym]={'file':str(actual),'SHA256':sha(actual)}
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
  generator=torch.Generator(device='cpu').manual_seed(8703);x=(torch.randn((4096,6144),generator=generator,dtype=torch.float32)*.1).to(torch.bfloat16).cuda();decode_x=(torch.randn((8,6144),generator=generator,dtype=torch.float32)*.1).to(torch.bfloat16).cuda();pos=torch.arange(512,dtype=torch.int64).repeat(8).cuda();decode_pos=torch.full((8,),512,dtype=torch.int64,device='cuda');out=torch.empty_like(x);decode_guard=torch.full((8*6144+128,),123,dtype=torch.bfloat16,device='cuda');decode_out=decode_guard[64:-64].view(8,6144);desc=(Z*24)(*[v for row in range(8)for v in [row,0,512]]);decode_desc=(Z*24)(*[v for row in range(8)for v in [row,512,1]])
  q['fixture_provenance']={'producer':'original8707 seed8703 BF16 prefix[4096,6144] + separate M8 hidden[8,6144]','past':512,'count':1,'current_Model_decode_capture':False,'weight_checkpoint_read_passes':1,'projection_quant_bytes_scales_direct_check':'not exposed by existing layerAPI; full layer output comparison is numerical acceptance'}
  reference=None;own_reference={};q['trials']=[]
  saved_inputs=[t.clone()for t in [x,decode_x,pos,decode_pos]];modes=[0,1,0,1,1,0,1,0,0,1,0] # 2warm + ABBA + BAAB + restore
  for index,mode in enumerate(modes):
   ck(lib.glm_attention_cohort_policy_r9054(rank,0));ck(lib.glm_reduced_output_policy_r9032(rank,0));ck(lib.glm_layer_reset_r5670(ctx));ck(lib.glm_layer_forward_r5670(ctx,4096,8,desc,V(x.data_ptr()),V(pos.data_ptr()),V(out.data_ptr()),0))
   ck(lib.glm_reduced_output_policy_r9032(rank,0));ck(lib.glm_attention_cohort_policy_r9054(rank,mode));before=stats();entry_before=entry_stats();torch.cuda.synchronize();tick=time.perf_counter();ck(lib.glm_layer_forward_r5670(ctx,8,8,decode_desc,V(decode_x.data_ptr()),V(decode_pos.data_ptr()),V(decode_out.data_ptr()),0));wall=(time.perf_counter()-tick)*1000;report=json.loads(lib.glm_layer_report_r5670());actual=decode_out.cpu();after=stats();entry_after=entry_stats();entry_delta=[a-b for a,b in zip(entry_after,entry_before)];q['latest_entry_delta']=entry_delta;delta=[a-b for a,b in zip(after,before)];hits=delta[1];q['latest_reduced_stats']=after
   if reference is None:reference=actual
   equal=torch.equal(actual.view(torch.int16),reference.view(torch.int16));own_reference.setdefault(mode,actual.clone());own_exact=torch.equal(actual.view(torch.int16),own_reference[mode].view(torch.int16));diff=actual.float()-reference.float();own_diff=actual.float()-own_reference[mode].float();rmse=float(diff.square().mean().sqrt());relrms=rmse/max(float(reference.float().square().mean().sqrt()),1e-30);own_rmse=float(own_diff.square().mean().sqrt());canary=bool((decode_guard[:64]==123).all()and(decode_guard[-64:]==123).all());q['actual_quality']={'index':index,'mode':mode,'finite':bool(torch.isfinite(actual.float()).all()),'exact':equal,'reduced_calls':hits};save()
   readonly=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip([x,decode_x,pos,decode_pos],saved_inputs))
   agree(readonly and canary and own_exact and bool(torch.isfinite(actual.float()).all()) and relrms<=0.005 and float(diff.abs().max())<=0.05*max(1.0,float(reference.float().abs().max()))and delta==([0,1,0]if mode else[1,0,0]) and entry_delta[:3]==[0,0,0] and entry_delta[3:]==[1,0,0] and report['profile']==0 and report['N']==8 and report['B']==8 and report['layers']==[38]and not report['stages'])
   q['trials'].append({'index':index,'mode':mode,'label':'warm'if index<2 else 'restore'if index==10 else 'ABBA'if index<6 else 'BAAB','route_fields_from_original_report':{k:report.get(k)for k in ['route_histogram','route_U','route_maxrows','route_readback_outside_wall']},'wholelayer_native_wall_ms':report['wall_ms'],'wholelayer_driver_CAPI_ms':wall,'owner_threadCPU_ms':report['threadCPU_ms'],'canary':canary,'inputs_readonly':readonly,'RMSE_vs_first_OFF':rmse,'relRMS_vs_first_OFF':relrms,'ownrepeat_bits_exact':own_exact,'ownrepeat_RMSE':own_rmse,'ownrepeat_maxabs':float(own_diff.abs().max()),'full_output_bits_exact':equal,'finite':bool(torch.isfinite(actual.float()).all()),'max_abs_vs_first_OFF':float((actual.float()-reference.float()).abs().max()),'reduced_calls':hits,'reduced_before':before,'reduced_after':after,'reduced_delta':delta,'entry_before':entry_before,'entry_after':entry_after,'entry_delta':entry_delta,'reduced_calls':delta[1],'ordinary_calls':delta[0],'launch_errors':delta[2],'output_SHA256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),'free_after':torch.cuda.mem_get_info()[0]});save()
  measurements=q['trials'][2:10];q['ordinary_summary']={str(mode):{metric:statistics.median([a[metric]for a in measurements if a['mode']==mode])for metric in ['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms']}for mode in [0,1]};q['wholelayer_native_speedup_OFF_over_ON']=q['ordinary_summary']['0']['wholelayer_native_wall_ms']/q['ordinary_summary']['1']['wholelayer_native_wall_ms'];q['gain_adoption']=False
  ck(lib.glm_attention_cohort_policy_r9054(rank,0));ck(lib.glm_reduced_output_policy_r9032(rank,0))
  ck(lib.glm_layer_reset_r5670(ctx));expired=I();ck(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1
  q['inputs_readonly']=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip([x,decode_x,pos,decode_pos],saved_inputs));assert q['inputs_readonly'];q['final_reduced_stats']=stats();assert q['final_reduced_stats'][0]>0 and q['final_reduced_stats'][1]>0 and q['final_reduced_stats'][2]==0;torch.cuda.synchronize();check=torch.ones(32,device='cuda');assert (check+1).sum().item()==64;torch.cuda.synchronize();q['blas_policy_after']=json.loads(policy.modern_blas_policy_r243_json());assert q['blas_policy_after']['failures']==0;q.update(completed=True,Runtime_weak_expired=True,all_outputs_finite=True,all_outputs_fullbits=all(t['full_output_bits_exact']for t in q['trials']),normal_reset_close=True,Torch_postclose=True,free_postclose=torch.cuda.mem_get_info()[0]);save()
 except BaseException as exc:q['error']=repr(exc);q['traceback']=traceback.format_exc();save();raise
 finally:
  if ctx and lib:
   expired=I();q['finally_destroy_attempted']=True
   try:q['finally_destroy_code']=int(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));q['finally_Runtime_expired']=bool(expired.value)
   except BaseException as exc:q['finally_error']=repr(exc)
   save()

if __name__=='__main__':
 assert torch.cuda.device_count()==8;assert sha(LIBRARY)==DSO_SHA;torch.cuda.set_device(0);lib=api();uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0;mp.spawn(worker,args=(bytes(uid),),nprocs=8,join=True)
