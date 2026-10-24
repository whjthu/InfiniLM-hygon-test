"""CPU-prepared finite layer38 component; no execution without root's concrete GPU GO.
One native context/weight load per owner. Synthetic replicated BF16 hidden.
Ordinary wholelayer wall includes producer/proof/fallback CPU work; no Model TPS.
"""
import pathlib,json,ctypes,os,time,hashlib,contextlib,statistics,traceback
import re
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');base=r/'candidates/glm-tile2-r8831';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
LIBRARY=pathlib.Path(json.loads((base/'CPU-build.json').read_text())['library'])
DSO_SHA='0e8a8354e8f6db9d672f4245534736cb5e3c2b9ae8e42371f372e0d26f87a0db'
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
 lib.modern_glm_tile2_policy_r8830.argtypes=[I,I]
 lib.modern_glm_tile2_hits_r8830.argtypes=[I];lib.modern_glm_tile2_hits_r8830.restype=ctypes.c_uint64
 for name in ["glm_vec_gate_r8528","glm_tile2_gate_r8830"]:getattr(lib,name).argtypes=[V]*9+[Z,Z,V]
 return lib

def worker(rank,unique):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8])
 p=r/'results'/f'GLM-prefill-prefix-component-r8834-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists()
 q={'completed':False,'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'library_SHA256':DSO_SHA,'layer':38,'API':'existing glm_layer_*_r5670','ordinary_metric':'wholelayer wall, not pureattention; includes proof/fallback CPU validation','source_input_scope':'Synthetic CPU-generated BF16 hidden replicated across TP8; real layer38 weights loaded once. No Model token/TPS/adoption claim.','CPU_affinity':sorted(os.sched_getaffinity(0)),'candidate':'tile2 gate-only routes64/defaultOFF; vectorON,downoriginal; prefixON fixed','trials':[],'decode_fallback':[],'irregular_partial_selection_GPU':'NOT_SUPPORTED_BY_EXISTING_API_CPU_ONLY_4006_DIFFERENTIAL'}
 ctx=None;lib=None
 def save():temp=p.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(p)
 def ck(code):
  if code:raise RuntimeError((code,lib.glm_layer_error_r5670().decode()))
 def agree(value):
  vote=torch.tensor([float(bool(value))],dtype=torch.float32,device='cuda');torch.cuda.synchronize();ck(lib.glm_layer_control_r5670(ctx,V(vote.data_ptr())));assert vote.item()==8,('unanimous_quality',rank,vote.item())
 try:
  proof=json.loads((base/'CPU-build.json').read_text());assert proof['completed']and proof['library_SHA256']==DSO_SHA==sha(proof['library']);assert len(proof['compiled'])==2 and len(proof['reused_objects_SHA256'])==26
  manifest=json.loads((r/'incoming/GLM-prefill-prefix-component-manifest-r8834.json').read_text())
  for rel,h in manifest['DSO_SHA256'].items():assert sha(r/rel)==h,('runtime_DSO_SHA',rel)
  lib=api();assert lib.modern_glm_dense_mode_r48_set(2)==0;ck(lib.modern_glm_vector_policy_r8540(rank,1));ck(lib.modern_glm_tile2_policy_r8830(rank,0));os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='1'
  policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p;assert json.loads(policy.modern_blas_policy_r243_json())['failures']==0
  class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
  process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];process.dladdr.restype=I
  owners={}
  for sym in ['glm_layer_create_r5670','glm_layer_forward_r5670','modern_glm_dense_mode_r48_set','modern_glm_prefill_prefix_plan_counts_r8703','modern_glm_shared_validation_counts_r5814']:
   inf=DlInfo();assert process.dladdr(ctypes.cast(getattr(lib,sym),V),ctypes.byref(inf));actual=pathlib.Path(inf.fname.decode()).resolve();assert actual==pathlib.Path(proof['library']).resolve();owners[sym]={'file':str(actual),'SHA256':sha(actual)}
  q['actual_symbol_owners']=owners;q['free_before_create']=torch.cuda.mem_get_info()[0]
  cfg=json.loads((model/'config.json').read_text());assert cfg['mlp_layer_types'][38]=='sparse'and cfg['indexer_types'][38]=='full';cfg['probe_layers_r5680']=[38]
  started=time.monotonic();uid=ctypes.create_string_buffer(unique,len(unique));ctx=lib.glm_layer_create_r5670(rank,json.dumps(cfg).encode(),uid);assert ctx,lib.glm_layer_error_r5670();q['create_seconds']=time.monotonic()-started
  keys=json.loads(lib.glm_layer_keys_r5670(ctx));index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert len(keys)==len(set(keys))and all(k.startswith('38.')and 'model.layers.'+k in index for k in keys)
  loadedbytes=0;projection_cpu={};started=time.monotonic()
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
     matched=re.fullmatch(r'38\.mlp\.experts\.(\d+)\.(gate_proj|up_proj)\.(weight|weight_scale_inv)',key)
     if matched:
      expert,proj,kind=matched.groups();v=x[rank*256:rank*256+256,:]if kind=='weight'else x
      projection_cpu[(int(expert),proj,kind)]=v.clone().contiguous()
     desc.append({'name':key,'shape':list(x.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[x.dtype],'pointer':x.data_ptr()})
    ck(lib.glm_layer_load_r5670(ctx,json.dumps(desc).encode()))
  ck(lib.glm_layer_finish_r5670(ctx));q.update(load_seconds=time.monotonic()-started,global_source_bytes_per_owner=loadedbytes,only_one_weight_load=True,free_after_load=torch.cuda.mem_get_info()[0]);agree(q['free_after_load']>=6*1024**3);save()
  # Projection and enclosing-layer fixtures share this one checkpoint read/load.
  banks=[torch.stack([projection_cpu[(e,proj,'weight')]for e in range(256)]).cuda()for proj in ['gate_proj','up_proj']]
  scales=[torch.stack([projection_cpu[(e,proj,'weight_scale_inv')]for e in range(256)]).float().cuda()for proj in ['gate_proj','up_proj']];projection_cpu.clear()
  generator=torch.Generator(device='cpu').manual_seed(8703);x=(torch.randn((4096,6144),generator=generator,dtype=torch.float32)*.1).to(torch.bfloat16).cuda();decode_x=(torch.randn((8,6144),generator=generator,dtype=torch.float32)*.1).to(torch.bfloat16).cuda();pos=torch.arange(512,dtype=torch.int64).repeat(8).cuda();decode_pos=torch.full((8,),512,dtype=torch.int64,device='cuda');out=torch.empty_like(x);decode_out=torch.empty_like(decode_x);desc=(Z*24)(*[v for row in range(8)for v in [row,0,512]]);decode_desc=(Z*24)(*[v for row in range(8)for v in [row,512,1]])
  q['fixture_provenance']={'producer':'original8707 seed8703 generated BF16 prefix[4096,6144] plus separately generated M8 decode input[8,6144]','past':512,'count':1,'current_Model_decode_capture':False,'projection_routes':'original8534 tokens8 U53 constructed routes, current actual layer38 TP shard weights; synthetic quant inputs','not_prefill_tail':True,'weight_checkpoint_read_passes':1}
  # Exact full projection outputs, current vec gate versus tile2; down unused/unchanged.
  qq=decode_x.float().view(8,48,128);xs=qq.abs().amax(-1).clamp_min(1e-10)/448.;packed=(qq/xs[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(8,6144).contiguous();xs=xs.contiguous()
  unique=[(j*37+rank*17)%256 for j in range(53)];rr=unique+[unique[j%53]for j in range(11)];ids=torch.tensor(rr,dtype=torch.int64,device='cuda');guard=torch.full((2,64*256+2),123.,dtype=torch.bfloat16,device='cuda');g=guard[0,1:-1].view(64,256);u=guard[1,1:-1].view(64,256)
  reference=None;q['projection_trials']=[]
  for index,mode in enumerate([0,1,0,1,1,0,1,0,0,1,0]):
   f=lib.glm_tile2_gate_r8830 if mode else lib.glm_vec_gate_r8528
   torch.cuda.synchronize();tick=time.perf_counter();ck(f(V(g.data_ptr()),V(u.data_ptr()),V(packed.data_ptr()),V(xs.data_ptr()),V(ids.data_ptr()),V(banks[0].data_ptr()),V(scales[0].data_ptr()),V(banks[1].data_ptr()),V(scales[1].data_ptr()),64,rank*256,V(torch.cuda.current_stream().cuda_stream)));torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
   actual=torch.cat([g,u],dim=1).cpu()
   if reference is None:reference=actual
   agree(bool(torch.isfinite(actual.float()).all())and torch.equal(actual.view(torch.int16),reference.view(torch.int16))and bool((guard[:,0]==123).all())and bool((guard[:,-1]==123).all()))
   q['projection_trials'].append({'index':index,'mode':mode,'wall_ms':wall,'full_output_bits_exact':True,'outputs':32768,'canary':True})
  del banks,scales,guard,g,u,packed,xs,qq,ids,actual,reference;torch.cuda.synchronize();torch.cuda.empty_cache();agree(torch.cuda.mem_get_info()[0]>=6*1024**3);save()
  reference=None;q['trials']=[]
  modes=[0,1,0,1,1,0,1,0,0,1,0] # 2warm + ABBA + BAAB + restore
  for index,mode in enumerate(modes):
   ck(lib.modern_glm_tile2_policy_r8830(rank,0));ck(lib.glm_layer_reset_r5670(ctx));ck(lib.glm_layer_forward_r5670(ctx,4096,8,desc,V(x.data_ptr()),V(pos.data_ptr()),V(out.data_ptr()),0))
   ck(lib.modern_glm_tile2_policy_r8830(rank,mode));before=int(lib.modern_glm_tile2_hits_r8830(rank));torch.cuda.synchronize();tick=time.perf_counter();ck(lib.glm_layer_forward_r5670(ctx,8,8,decode_desc,V(decode_x.data_ptr()),V(decode_pos.data_ptr()),V(decode_out.data_ptr()),0));wall=(time.perf_counter()-tick)*1000;report=json.loads(lib.glm_layer_report_r5670());actual=decode_out.cpu();hits=int(lib.modern_glm_tile2_hits_r8830(rank))-before
   if reference is None:reference=actual
   equal=torch.equal(actual.view(torch.int16),reference.view(torch.int16));q['actual_quality']={'index':index,'mode':mode,'finite':bool(torch.isfinite(actual.float()).all()),'exact':equal,'tile2_hits':hits};save()
   if not equal:
    path=p.with_suffix('.actual.bin');actual.view(torch.uint8).numpy().tofile(path);q['actual_quality']['actual_file']=str(path);q['actual_quality']['actual_SHA256']=sha(path);save()
   agree(equal and bool(torch.isfinite(actual.float()).all())and hits==mode and report['profile']==0 and report['N']==8 and report['B']==8 and report['layers']==[38]and not report['stages'])
   q['trials'].append({'index':index,'mode':mode,'label':'warm'if index<2 else 'restore'if index==10 else 'ABBA'if index<6 else 'BAAB','wholelayer_native_wall_ms':report['wall_ms'],'wholelayer_driver_CAPI_ms':wall,'owner_threadCPU_ms':report['threadCPU_ms'],'full_output_bits_exact':True,'tile2_hits':hits,'output_SHA256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),'free_after':torch.cuda.mem_get_info()[0]});save()
  measurements=q['trials'][2:10];q['ordinary_summary']={str(mode):{metric:statistics.median([a[metric]for a in measurements if a['mode']==mode])for metric in ['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms']}for mode in [0,1]};q['wholelayer_native_speedup_O_over_N']=q['ordinary_summary']['0']['wholelayer_native_wall_ms']/q['ordinary_summary']['1']['wholelayer_native_wall_ms'];q['gain_adoption']=False
  ck(lib.modern_glm_tile2_policy_r8830(rank,0))
  ck(lib.glm_layer_reset_r5670(ctx));expired=I();ck(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1
  torch.cuda.synchronize();check=torch.ones(32,device='cuda');assert (check+1).sum().item()==64;torch.cuda.synchronize();q['blas_policy_after']=json.loads(policy.modern_blas_policy_r243_json());assert q['blas_policy_after']['failures']==0;q.update(completed=True,Runtime_weak_expired=True,all_outputs_fullbits=True,normal_reset_close=True,Torch_postclose=True,free_postclose=torch.cuda.mem_get_info()[0]);save()
 except BaseException as exc:q['error']=repr(exc);q['traceback']=traceback.format_exc();save();raise
 finally:
  if ctx and lib:
   expired=I();q['finally_destroy_attempted']=True
   try:q['finally_destroy_code']=int(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));q['finally_Runtime_expired']=bool(expired.value)
   except BaseException as exc:q['finally_error']=repr(exc)
   save()

if __name__=='__main__':
 assert torch.cuda.device_count()==8;assert sha(LIBRARY)==DSO_SHA;torch.cuda.set_device(0);lib=api();uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0;mp.spawn(worker,args=(bytes(uid),),nprocs=8,join=True)
