"""Future singleGPU component only; root must separately authorize actualGPU launch.
Never imported or executed by CPU_build. No Model or service-rate claim.
"""
def run():
 import ctypes as C,pathlib,json,hashlib,time,statistics,traceback,os
 import torch
 from safetensors import safe_open
 r=pathlib.Path('/refactor');inc=r/'incoming/GLM-M15-component-r121';b=r/'candidates/GLM-M15-component-r121b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 gp=pathlib.Path(__file__).resolve().parent;gm=json.loads((gp/'GPU-manifest-r121b.json').read_bytes())
 for n,h in gm['source_SHA256'].items():assert sha(gp/n)==h,n
 assert sha(gp/'manifest-r121b.json')==gm['CPU_manifest_SHA256']
 expected_cpu=os.environ.get('GLM_M15_CPU_RECEIPT_SHA256_R121B','');assert len(expected_cpu)==64 and sha(b/'CPU-build.json')==expected_cpu
 m=json.loads((inc/'manifest-r121.json').read_bytes());cpu=json.loads((b/'CPU-build.json').read_bytes());assert cpu['completed'] and cpu['CPU_only'] and not cpu['dlopen'] and cpu['no_recompile_or_relink']
 assert cpu['repair_manifest_SHA256']==gm['CPU_manifest_SHA256'] and cpu['library_SHA256']==gm['original_DSO_SHA256']==sha(cpu['library'])
 assert os.environ['SLURM_JOB_ID']=='910376' and os.uname().nodename=='a02r4n12' and torch.cuda.device_count()==1
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['actual_binding_SHA256'].items():assert sha(r/n)==h,n
 device=0;rank=7;co=rank*2048;torch.set_num_threads(1);torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 V=C.c_void_p;Z=C.c_size_t;I=C.c_int;lib=C.CDLL(cpu['library']);lib.glm_m15_component_error_r121.restype=C.c_char_p;lib.glm_m15_component_create_r121.argtypes=[I,I];lib.glm_m15_component_create_r121.restype=V;lib.glm_m15_component_policy_r121.argtypes=[V,I];lib.glm_m15_component_forward_r121.argtypes=[V,V,V,V,V,V,C.POINTER(C.c_double)];lib.glm_m15_component_stats_r121.argtypes=[V,C.POINTER(C.c_uint64),Z];lib.glm_m15_component_clear_r121.argtypes=[V];lib.glm_m15_component_destroy_r121.argtypes=[V]
 def ck(x):
  if x:raise RuntimeError(lib.glm_m15_component_error_r121().decode())
 def ptr(x):return V(x.data_ptr())
 def stats():
  v=(C.c_uint64*16)();ck(lib.glm_m15_component_stats_r121(ctx,v,16));return list(v)
 report={'completed':False,'scope':'oneM15o_proj wholequant+projection+drain; syntheticBF16input/trainedcheckpointweights; notModel/notactual755activation','device':device,'logical_TP_rank':rank,'M':15,'N':6144,'K':2048,'scale_shape':[48,128],'column_offset':co,'CPU_receipt_SHA256':sha(b/'CPU-build.json'),'library_SHA256':cpu['library_SHA256'],'warm':[],'trials':[],'runtime_floor_bytes':536870912,'physical_peak':'UNKNOWN','GPU_F64':False,'only_one_weight_read':True};ctx=None
 out=r/'results'/('GLM-M15-component-r121b-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not out.exists()
 def save():out.write_text(json.dumps(report,indent=2))
 try:
  model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];wk='model.layers.38.self_attn.o_proj.weight';sk=wk.removesuffix('.weight')+'.weight_scale_inv'
  with safe_open(str(model/index[wk]),framework='pt',device='cpu') as f:w=f.get_slice(wk)[:,co:co+2048].contiguous().view(torch.uint8)
  with safe_open(str(model/index[sk]),framework='pt',device='cpu') as f:ws=f.get_tensor(sk).float().contiguous()
  assert tuple(w.shape)==(6144,2048) and tuple(ws.shape)==(48,128);report['trained_metadata']={'weight_key':wk,'scale_key':sk,'weight_SHA256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),'scale_SHA256':hashlib.sha256(ws.numpy().tobytes()).hexdigest()}
  w=w.cuda();ws=ws.cuda();g=torch.Generator(device='cpu').manual_seed(12115);x=(torch.randn((15,2048),generator=g)*.1).to(torch.bfloat16).cuda();readonly=[t.clone()for t in (x,w,ws)];guard=torch.full((15*6144+128,),123,dtype=torch.bfloat16,device='cuda');y=guard[64:-64].view(15,6144)
  ctx=lib.glm_m15_component_create_r121(device,rank);assert ctx,lib.glm_m15_component_error_r121();assert stats()[5]==0;report['default_OFF_verified']=True;previous={}
  for warm,mode in [(True,z)for z in (0,1,0,1)]+[(False,z)for z in (0,1,1,0)*2]:
   ck(lib.glm_m15_component_policy_r121(ctx,mode));y.fill_(float('nan'));torch.cuda.synchronize();before=stats();native=C.c_double();start=time.perf_counter();ck(lib.glm_m15_component_forward_r121(ctx,ptr(x),ptr(w),ptr(ws),ptr(y),V(torch.cuda.current_stream().cuda_stream),C.byref(native)));wall=1000*(time.perf_counter()-start);assert torch.isfinite(y).all().item() and (guard[:64]==123).all().item() and (guard[-64:]==123).all().item();free=torch.cuda.mem_get_info()[0];assert free>=536870912
   result=y.clone();own_exact=mode not in previous or torch.equal(result,previous[mode]);previous[mode]=result
   after=stats();assert after[0]==before[0]+1 and after[1]==before[1]+(mode==0) and after[2]==before[2]+(mode==1) and after[4]==after[15]==0 and after[5]==mode
   assert after[14]==before[14]+(mode==1)
   if mode==1:
    assert after[11]==1 and after[12]==31125504
    if before[11]==0:assert after[6]==before[6]+1 and after[9]==before[9]+1
    else:assert after[8]==before[8]+1
   else:assert after[6:16]==before[6:16]
   row={'mode':mode,'HOST_CAPI_ms':wall,'native_whole_projection_ms':native.value,'stats_before':before,'stats_after':after,'within_arm_repeat_exact_diagnostic':own_exact,'driver_free_bytes':free};report['warm'if warm else'trials'].append(row);assert all(torch.equal(t,b0)for t,b0 in zip((x,w,ws),readonly));save()
  a=previous[0].float();d=previous[1].float()-a;report['numerical']={'finite':True,'max_abs':d.abs().max().item(),'relative_RMS':(d.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-20)).item(),'tolerance_provenance':'existing3039CPUauditor component relativeerror0.005; noModelqualityclaim'};assert report['numerical']['relative_RMS']<.005
  trials=report['trials'];pairs=[]
  for i in range(0,8,2):
   u,v=trials[i:i+2];off=u if u['mode']==0 else v;on=u if u['mode']==1 else v;pairs.append(off['native_whole_projection_ms']/on['native_whole_projection_ms']-1)
  report['paired_speedup']=pairs;report['median_speedup']=statistics.median(pairs);report['component_positive']=all(v>0 for v in pairs);report['before_clear_stats']=stats();ck(lib.glm_m15_component_clear_r121(ctx));report['after_clear_stats']=stats();assert report['after_clear_stats'][6]==report['after_clear_stats'][7] and report['after_clear_stats'][11]==report['after_clear_stats'][12]==0;ck(lib.glm_m15_component_destroy_r121(ctx));ctx=None;torch.cuda.synchronize();report.update(completed=True,normal_close=True,no_Model_gain_claim=True)
 except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if ctx is not None:
   report['failure_clear_returncode']=lib.glm_m15_component_clear_r121(ctx);report['failure_destroy_returncode']=lib.glm_m15_component_destroy_r121(ctx)
  save()
if __name__=='__main__':run()
