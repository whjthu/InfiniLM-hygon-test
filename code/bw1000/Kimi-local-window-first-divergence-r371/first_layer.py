"""Same firstlayer inputs: retained KV producers versus fixed-QKV attention crop.

Private function globals are copied, never monkeypatched. Capture operands stay
inside this one call and only scalar summaries may be serialized. No launcher.
"""
import types,time,hashlib
import torch
from private_attention import private_forward

THRESHOLD=2e-2

def summary(a,b):
 x=a.detach().float().cpu();y=b.detach().float().cpu()
 assert x.shape==y.shape
 finite=bool(torch.isfinite(x).all()and torch.isfinite(y).all())
 nr=float(torch.linalg.vector_norm(x-y)/(torch.linalg.vector_norm(x)+1e-30))if finite else None
 return dict(shape=list(x.shape),reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),finite=finite,NRMSE=nr,maxabs=float((x-y).abs().max())if finite else None,reference_RMS=float(torch.mean(x.square()).sqrt()),candidate_RMS=float(torch.mean(y.square()).sqrt()),above_original_02_diagnostic=(nr is None or nr>THRESHOLD))

def clone_globals(function,flex):
 env=dict(function.__globals__);env['compile_friendly_flex_attention']=flex
 out=types.FunctionType(function.__code__,env,function.__name__,function.__defaults__,function.__closure__)
 out.__kwdefaults__=function.__kwdefaults__
 return out

def first_layer(attention,draft_hidden,context,dp,cp,mask,save=lambda result:None,*,fence=lambda:None,original_output=None):
 L=context.shape[1];start=L-4095
 assert L==32638 and context.shape[0]==1 and draft_hidden.shape[1]==8
 assert dp.shape==(1,8)and cp.shape==(1,L)and mask.shape==(1,1,8,L+8)
 assert not bool(mask[...,:start].any())and bool(mask[...,L:].all())
 base_function=type(attention).forward
 original_flex=base_function.__globals__['compile_friendly_flex_attention']
 stages=['q_a_proj','q_a_layernorm','q_b_proj','kv_a_proj_with_mqa','kv_a_layernorm','kv_b_proj','o_proj']
 def rope_signature():
  return {n:dict(object_id=id(t),pointer=t.data_ptr(),shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest())for n,t in attention.rotary_emb.named_buffers()}
 rope_before=rope_signature()
 result=dict(completed=False,scope='Firstattention only; scalar summaries, no acceptance or performance qualification',original_threshold=THRESHOLD,producer_stages=[],runtime=dict(torch_version=torch.__version__,torch_HIP_version=torch.version.hip,default_dtype=str(torch.get_default_dtype()),input_dtype=str(draft_hidden.dtype),device=str(draft_hidden.device),SDPA_enabled_flags=dict(math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled()),actual_SDPA_algorithm='UNKNOWN; enabledflags do not identify selected backend',matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),RoPE_before=rope_before)
 def run(crop):
  captures={};held={};handles=[]
  def hook(name):
   def capture(module,args,out):
    # Full retainedcontext+all8draft rows, not a cherry-picked row sample.
    value=out[:,start:]if not crop and name.startswith('kv_')else out
    captures[name]=value.detach().clone()
   return capture
  for name in stages:handles.append(attention.get_submodule(name).register_forward_hook(hook(name)))
  def flex(**kwargs):
   held.update(kwargs);out=original_flex(**kwargs);held['output']=out;return out
  function=clone_globals(private_forward(base_function)if crop else base_function,flex)
  fence();tick=time.perf_counter()
  try:
   out=function(attention,draft_hidden,context[:,start:]if crop else context,dp,cp[:,start:]if crop else cp,mask[...,start:]if crop else mask,**({'absolute_span':L+8}if crop else {}))
   fence();wall=time.perf_counter()-tick
  finally:
   for handle in handles:handle.remove()
  return out,captures,held,wall
 full,ref,shared,wall=run(False);result['full_firstattention_wall_including_capture_s']=wall
 if original_output is not None:
  result['OFF_replay_to_original_attention_output']=summary(original_output,full);save(result)
 # SAME materialized Q/K/V: only remove all-False keys. Projection is absent.
 # Contiguous crop matches the ordinary candidate's dense K/V physical layout.
 fence();tick=time.perf_counter()
 cropped_attention=original_flex(query=shared['query'],key=shared['key'][:,:,start:].contiguous(),value=shared['value'][:,:,start:].contiguous(),block_mask=shared['block_mask'][...,start:],scale=shared['scale'])
 fence();result['same_QKV_crop_wall_including_copy_s']=time.perf_counter()-tick
 result['same_QKV_attention_crop']=summary(shared['output'],cropped_attention);save(result)
 cropped,candidate,on_shared,wall=run(True);result['recomputed_crop_firstattention_wall_including_capture_s']=wall
 for name in stages:
  row=dict(stage=name,**summary(ref[name],candidate[name]));result['producer_stages'].append(row);save(result)
  if not row['finite']:
   result.update(stop_reason='First nonfinite producer '+name,completed=True);save(result);return result
 result['first_nonzero_producer']=next((x['stage']for x in result['producer_stages']if x['NRMSE']>0),None)
 result['first_producer_above_02']=next((x['stage']for x in result['producer_stages']if x['above_original_02_diagnostic']),None)
 # Compare ALL retained aligned K/V after RoPE and all8 Q before SDPA.
 result['query_same_input']=summary(shared['query'],on_shared['query'])
 result['retained_key_recompute']=summary(shared['key'][:,:,start:],on_shared['key'])
 result['retained_value_recompute']=summary(shared['value'][:,:,start:],on_shared['value'])
 result['attention_output_recompute']=summary(shared['output'],on_shared['output'])
 result['projected_attention_output_recompute']=summary(full,cropped)
 result['RoPE_after']=rope_signature();result['RoPE_unchanged']=rope_before==result['RoPE_after']
 assert result['RoPE_unchanged']
 result.update(completed=True,stop_reason='Fixed firstlayer split complete; do not continue five-layer accumulation or ABBA',causal_limit='KV producer and fixedQKV crop are separate observed branches. Finiteprecision cause/backend algorithm is not inferred solely from cumulative hidden NR.')
 save(result);return result

def capture_original_query_input(runtime,sequence,epoch,save=lambda result:None):
 c=runtime.session.context;assert c.length==32638 and c.phase=='decode'
 attention=c.model.layers[0].self_attn;held={}
 def before(module,args):held['draft_hidden']=args[0].detach().clone()
 def after(module,args,out):held['original_output']=out.detach().clone()
 h=attention.q_a_proj.register_forward_pre_hook(before);o=attention.o_proj.register_forward_hook(after)
 c.crop_local=False
 try:
  tick=time.perf_counter();reply=runtime.session.dispatch(dict(cmd='query',seq=sequence,epoch=epoch,request=1,past=32638,anchor=163584,K=7),b'');wall=time.perf_counter()-tick
 finally:h.remove();o.remove()
 result=dict(original_complete_dispatch_wall_s=wall,handler_wall_s=reply['handler_wall_s'],proposals=reply['proposals'],hidden_finite=bool(torch.isfinite(c.last_debug['hidden']).all()),logits_finite=bool(torch.isfinite(c.last_debug['logits']).all()),profile=False,performance_qualified=False,scope='Original full query includes guard/pack/all5layers/head/selector/finite/proposals/fence, plus two small firstlayer input/output clones; no cost cropped away')
 save(result);return held,result,sequence+1
