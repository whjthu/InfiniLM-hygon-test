"""Future oneGPU/all8offset validation using existing20b240 DSO. No compilation.
Layer38 primary timing; layer39 freshweight/scales and clear/reuse are audits.
"""
def run():
 import pathlib,json,hashlib,ctypes as C,os,time,statistics,traceback
 import torch
 from safetensors import safe_open
 r=pathlib.Path('/refactor');gp=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();gm=json.loads((gp/'GPU-manifest-r127.json').read_bytes())
 for n,h in gm['source_SHA256'].items():assert sha(gp/n)==h,n
 cpu_path=r/'candidates/GLM-M15-component-r121b/CPU-build.json';expected=os.environ.get('GLM_M15_CPU_RECEIPT_SHA256_R121B','');assert sha(cpu_path)==expected==gm['CPU_receipt_SHA256'];cpu=json.loads(cpu_path.read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']and not cpu['dlopen']and cpu['no_recompile_or_relink'];assert cpu['repair_manifest_SHA256']==gm['CPU_manifest_SHA256'];assert sha(cpu['library'])==cpu['library_SHA256']==gm['DSO_SHA256']
 original=r/'incoming/GLM-M15-component-r121/manifest-r121.json';assert sha(original)==gm['original_manifest_SHA256'];original=json.loads(original.read_bytes())
 for n,h in original['actual_binding_SHA256'].items():assert sha(r/n)==h,n
 assert sha(r/'incoming/GLM-M15-component-r121b/manifest-r121b.json')==gm['CPU_manifest_SHA256'];assert os.environ['SLURM_JOB_ID']=='910376'and os.uname().nodename=='a02r4n12'and torch.cuda.device_count()==1
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 V=C.c_void_p;I=C.c_int;Z=C.c_size_t;lib=C.CDLL(cpu['library']);lib.glm_m15_component_error_r121.restype=C.c_char_p;lib.glm_m15_component_create_r121.argtypes=[I,I];lib.glm_m15_component_create_r121.restype=V;lib.glm_m15_component_policy_r121.argtypes=[V,I];lib.glm_m15_component_forward_r121.argtypes=[V,V,V,V,V,V,C.POINTER(C.c_double)];lib.glm_m15_component_stats_r121.argtypes=[V,C.POINTER(C.c_uint64),Z];lib.glm_m15_component_clear_r121.argtypes=[V];lib.glm_m15_component_destroy_r121.argtypes=[V]
 report={'completed':False,'normal_close':False,'scope':'OneGPU all8logicalTPoffsets; originaltrainedlayer38 primary, layer39 freshW/scales audit, constructedfullBF16input15x16384; notModel/TPS','device':0,'M':15,'N':6144,'K':2048,'scale_shape':[48,128],'CPU_receipt_SHA256':sha(cpu_path),'library_SHA256':cpu['library_SHA256'],'source_manifest_SHA256':sha(gp/'GPU-manifest-r127.json'),'weight_layers':[38,39],'weight_reads_per_layer':1,'offsets':[],'floor_bytes':536870912,'physical_peak':'UNKNOWN','GPU_F64':False,'no_Model_gain_claim':True}
 output=r/'results'/('GLM-M15-offset8-r127-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not output.exists();ctx=None;current=None
 def save():tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(output)
 def ck(code):
  if code:raise RuntimeError(lib.glm_m15_component_error_r121().decode())
 def ptr(x):return V(x.data_ptr())
 def stats():
  v=(C.c_uint64*16)();ck(lib.glm_m15_component_stats_r121(ctx,v,16));return list(v)
 def compare(value,reference):
  a=reference.float();d=value.float()-a;metric={'max_abs':d.abs().max().item(),'relative_RMS':(d.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-20)).item(),'exact_diagnostic':torch.equal(value,reference)};assert metric['relative_RMS']<.005,metric;return metric
 try:
  model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];weights={};report['trained_weights']={};load=time.perf_counter()
  for layer in (38,39):
   wk=f'model.layers.{layer}.self_attn.o_proj.weight';sk=wk.removesuffix('.weight')+'.weight_scale_inv'
   with safe_open(str(model/mapping[wk]),framework='pt',device='cpu')as f:w=f.get_tensor(wk)
   assert w.dtype==torch.float8_e4m3fn and tuple(w.shape)==(6144,16384);w=w.contiguous().view(torch.uint8)
   with safe_open(str(model/mapping[sk]),framework='pt',device='cpu')as f:ws=f.get_tensor(sk)
   assert ws.dtype==torch.float32 and tuple(ws.shape)==(48,128);ws=ws.contiguous();report['trained_weights'][str(layer)]={'weight_key':wk,'scale_key':sk,'weight_SHA256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),'scale_SHA256':hashlib.sha256(ws.numpy().tobytes()).hexdigest(),'single_weight_tensor_read':True};weights[layer]=(w.cuda(),ws.cuda())
  report['weight_load_seconds']=time.perf_counter()-load;assert report['trained_weights']['38']['weight_SHA256']!=report['trained_weights']['39']['weight_SHA256'];assert report['trained_weights']['38']['scale_SHA256']!=report['trained_weights']['39']['scale_SHA256']
  gen=torch.Generator(device='cpu').manual_seed(12716384);host=(torch.randn((15,16384),generator=gen)*.1).to(torch.bfloat16);report['full_input_SHA256']=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest();full=host.cuda();del host;save()
  for rank in range(8):
   co=rank*2048;x=full[:,co:co+2048].contiguous();slices={layer:(w[:,co:co+2048].contiguous(),ws)for layer,(w,ws)in weights.items()};readonly={layer:tuple(t.clone()for t in pair)for layer,pair in slices.items()};x_before=x.clone();guard=torch.full((15*6144+128,),123,dtype=torch.bfloat16,device='cuda');y=guard[64:-64].view(15,6144)
   current={'logical_rank':rank,'column_offset':co,'warm':[],'trials':[],'audits':[],'completed':False,'normal_close':False,'fresh_selected_weight_distinct':not torch.equal(slices[38][0],slices[39][0]),'fresh_selected_scale_blocks_distinct':not torch.equal(slices[38][1][:,co//128:co//128+16],slices[39][1][:,co//128:co//128+16])};report['offsets'].append(current)
   assert current['fresh_selected_weight_distinct'];ctx=lib.glm_m15_component_create_r121(0,rank);assert ctx,lib.glm_m15_component_error_r121();assert stats()[5]==0;current['default_OFF_verified']=True
   def one(layer,mode,kind):
    w,ws=slices[layer];assert x.dtype==torch.bfloat16 and w.dtype==torch.uint8 and ws.dtype==torch.float32 and tuple(w.shape)==(6144,2048)and tuple(ws.shape)==(48,128)
    ck(lib.glm_m15_component_policy_r121(ctx,mode));y.fill_(float('nan'));torch.cuda.synchronize();before=stats();native=C.c_double();start=time.perf_counter();ck(lib.glm_m15_component_forward_r121(ctx,ptr(x),ptr(w),ptr(ws),ptr(y),V(torch.cuda.current_stream().cuda_stream),C.byref(native)));outer=1000*(time.perf_counter()-start);after=stats()
    assert torch.isfinite(y).all().item()and(guard[:64]==123).all().item()and(guard[-64:]==123).all().item();assert torch.equal(x,x_before)and all(torch.equal(t,z)for lay,pair in slices.items()for t,z in zip(pair,readonly[lay]));free=torch.cuda.mem_get_info()[0];assert free>=536870912
    assert after[0]==before[0]+1 and after[1]==before[1]+(mode==0)and after[2]==before[2]+(mode==1)and after[4]==after[15]==0 and after[5]==mode and after[14]==before[14]+(mode==1)
    if mode:
     assert after[11]==1 and after[12]==31125504
     if before[11]==0:assert after[6]==before[6]+1 and after[9]==before[9]+1
     else:assert after[8]==before[8]+1 and after[6]==before[6]
    else:assert after[6:16]==before[6:16]
    row={'layer':layer,'mode':mode,'kind':kind,'native_whole_projection_ms':native.value,'HOST_CAPI_ms':outer,'stats_before':before,'stats_after':after,'free_bytes':free};return y.clone(),row
   reference=None
   for mode in (0,1,0,1):
    value,row=one(38,mode,'primary_warm')
    if mode==0:reference=value
    else:row['numerical']=compare(value,reference)
    current['warm'].append(row)
   for mode in (0,1,1,0)*2:
    value,row=one(38,mode,'primary_timing');row['numerical']=compare(value,reference);current['trials'].append(row)
   pairs=[]
   for i in range(0,8,2):
    a,b=current['trials'][i:i+2];off=a if a['mode']==0 else b;on=a if a['mode']==1 else b;pairs.append(off['native_whole_projection_ms']/on['native_whole_projection_ms'])
   current['rate_ratios']=pairs;current['median_rate_ratio']=statistics.median(pairs);current['positive_pairs']=sum(v>1 for v in pairs)
   before_fresh=stats();fresh_off,row=one(39,0,'fresh_trained_layer39_OFF');current['audits'].append(row);fresh_on,row=one(39,1,'fresh_trained_layer39_ON_same_cache');row['numerical']=compare(fresh_on,fresh_off);assert stats()[6]==before_fresh[6]and stats()[9]==before_fresh[9];current['audits'].append(row)
   restored,row=one(38,1,'restore_layer38_same_cache');row['numerical']=compare(restored,reference);current['audits'].append(row)
   ck(lib.glm_m15_component_clear_r121(ctx));cleared=stats();assert cleared[6]==cleared[7]and cleared[11]==cleared[12]==0;current['after_intermediate_clear']=cleared
   reused,row=one(38,1,'clear_then_reuse_layer38');row['numerical']=compare(reused,reference);assert stats()[6]==cleared[6]+1;current['audits'].append(row)
   ck(lib.glm_m15_component_clear_r121(ctx));current['after_clear_stats']=stats();assert current['after_clear_stats'][6]==current['after_clear_stats'][7]and current['after_clear_stats'][11]==current['after_clear_stats'][12]==0;ck(lib.glm_m15_component_destroy_r121(ctx));ctx=None;torch.cuda.synchronize();current.update(completed=True,normal_close=True,fresh_weight_audit=True,fresh_scale_audit=current['fresh_selected_scale_blocks_distinct'],clear_reuse_audit=True);save()
  report['all_offsets_validated']=len(report['offsets'])==8 and all(q['completed']and q['normal_close']and q['fresh_weight_audit']and q['fresh_scale_audit']and q['clear_reuse_audit']for q in report['offsets']);report['all_offsets_positive']=all(q['positive_pairs']==4 for q in report['offsets']);report['all_offsets_qualified']=report['all_offsets_validated']and report['all_offsets_positive'];report.update(completed=True,normal_close=True)
 except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if ctx is not None:
   current['failure_clear_returncode']=lib.glm_m15_component_clear_r121(ctx);current['failure_destroy_returncode']=lib.glm_m15_component_destroy_r121(ctx)
  save()
if __name__=='__main__':run()
