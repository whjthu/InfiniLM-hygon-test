"""Bounded actual-firstlayer CPU consumer plus production-width BF16 projection sampling."""
from pathlib import Path
import json,sys,copy,hashlib,time,resource

def main():
 D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'incoming/Kimi-local-context-crop-r369';inc=R/'incoming'
 sys.path[:0]=[str(D),str(B),str(B/'gpu-v1'),str(inc),str(inc/'Kimi-long-draft-RPC-CPU-r333')]
 import torch
 from dflash2_creator_factory_r9007 import make_factory
 from window_context import WindowContext
 from long_rpc import DraftSession,TorchFeatureCodec
 from types import SimpleNamespace
 from first_layer import first_layer,capture_original_query_input,summary
 torch.set_num_threads(4);torch.manual_seed(371)
 raw=(inc/'config.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='aac0900105b9099b99a452c728ff9c4685889d264f7bed0101e57788df8f09c8'
 cfg=json.loads(raw);small=copy.deepcopy(cfg)
 small.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
 small['dflash_config']=dict(small['dflash_config'],selector_rank=8,selector_top_k=4)
 model=make_factory(small,inc)().to(torch.bfloat16).eval();head=torch.nn.Linear(32,cfg['vocab_size'],bias=False,dtype=torch.bfloat16).eval()
 before={k:v.detach().clone()for k,v in model.state_dict().items()};head_before=head.weight.detach().clone()
 start=time.perf_counter()
 with torch.inference_mode():
  context=WindowContext(model,'CPU-r371-diagnostic-only',enabled=True)
  session=DraftSession(context,head,TorchFeatureCodec('cpu',raw_dim=160),enabled=True);runtime=SimpleNamespace(session=session)
  seq=0
  def command(cmd,body=b'',**fields):
   nonlocal seq
   meta=dict(cmd=cmd,seq=seq)
   if cmd!='close':meta.update(epoch=1,request=1)
   meta.update(fields);answer=session.dispatch(meta,body);seq+=1;return answer
  command('begin',prompt_tokens=32638)
  raw_part=(torch.randn(1,512,160,dtype=torch.bfloat16)*.1).contiguous().view(torch.uint16).numpy().tobytes()
  for i,past in enumerate(range(0,32638,512)):
   n=min(512,32638-past);command('append_prompt',raw_part[:n*160*2],past=past,rows=n,capture_epoch=i+1)
  command('seal',past=32638)
  ctx_before=context.projected.clone();model_ids=[(n,id(v))for n,v in model.named_modules()]
  held,wholequery,seq=capture_original_query_input(runtime,seq,1)
  L=32638;cp=torch.arange(L).view(1,-1);dp=L+torch.arange(8).view(1,-1)
  keys=torch.cat([cp,dp],1);mask=((dp[:,:,None]-keys[:,None,:]).abs()<4096)[:,None]
  diagnostic=first_layer(model.layers[0].self_attn,held['draft_hidden'],context.projected,dp,cp,mask,fence=session.codec.fence,original_output=held['original_output'])
  assert diagnostic['completed']and all(v['finite']for v in diagnostic['producer_stages'])
  assert torch.equal(ctx_before,context.projected)and model_ids==[(n,id(v))for n,v in model.named_modules()]
  assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
  command('retire');command('close');assert session.closed and context.raw is None and context.projected is None
  assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())and torch.equal(head_before,head.weight)
  pc=copy.copy(model.config)
  for k in ['hidden_size','num_attention_heads','q_lora_rank','kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim']:setattr(pc,k,cfg[k])
  attn=type(model.layers[0].self_attn)(pc).to(torch.bfloat16).eval()
  cut=L-4095;source=torch.randn(1,L+8,7168,dtype=torch.bfloat16)*.1
  tick=time.perf_counter();full=attn.kv_a_proj_with_mqa(source);full_wall=time.perf_counter()-tick
  crop_source=source[:,cut:].contiguous()
  tick=time.perf_counter();narrow=attn.kv_a_proj_with_mqa(crop_source);crop_wall=time.perf_counter()-tick
  aligned=full[:,cut:];kva=summary(aligned,narrow);assert kva['finite']
  # Full latent/output widths, but only12 aligned rows for KV_B. This is not
  # a fullM GPU GEMM proxy, nor a productioncheckpoint/acceptance test.
  indices=torch.tensor([0,1,4087,4094]+list(range(4095,4103)))
  x=aligned.index_select(1,indices);y=narrow.index_select(1,indices)
  xn=attn.kv_a_layernorm(x[...,:512]);yn=attn.kv_a_layernorm(y[...,:512])
  rope=attn.rotary_emb;buffers={n:dict(id=id(t),pointer=t.data_ptr(),dtype=str(t.dtype),shape=list(t.shape))for n,t in rope.named_buffers()}
  dummy=torch.zeros(1,64,8,64,dtype=torch.bfloat16)
  ac,asi=rope(dummy,seq_len=L+8);bc,bsi=rope(dummy,seq_len=L+8)
  assert torch.equal(ac,bc)and torch.equal(asi,bsi)
  assert buffers=={n:dict(id=id(t),pointer=t.data_ptr(),dtype=str(t.dtype),shape=list(t.shape))for n,t in rope.named_buffers()}
  production=dict(random_parameters_only=True,hidden_width=7168,KV_A_output=576,latent=512,KV_B_output=16384,full_KV_A_M=L+8,crop_KV_A_M=4103,KV_A=kva,KV_A_full_wall_s=full_wall,KV_A_crop_wall_s=crop_wall,KV_norm_12_aligned_rows=summary(xn,yn),KV_B_12_aligned_rows=summary(attn.kv_b_proj(xn),attn.kv_b_proj(yn)),RoPE_buffers=buffers,RoPE_same_span_exact=True,limitations='CPU/randomparameters; KV_B only12rows. No trained/GPU backend, samefailed690 raw, acceptedtokens, or Model qualification.')
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_loaded=False,actual_original_query=wholequery,actual_same_firstlayer_diagnostic=diagnostic,protocol_commands=seq,protocol_closed=True,inputs_weights_modules_unchanged=True,production_width_sampling=production,total_wall_s=time.perf_counter()-start,CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))

if __name__=='__main__':main()
