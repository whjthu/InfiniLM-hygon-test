"""Real pinned small Torch consumer; no checkpoint loading and no GPU."""
from pathlib import Path
import hashlib,json,sys,copy

def main():
 import torch
 from torch import nn
 D=Path(__file__).resolve().parent;R=Path('/refactor');incoming=R/'incoming'
 sys.path[:0]=[str(incoming),str(incoming/'Kimi-long-draft-RPC-CPU-r333')]
 from dflash2_creator_factory_r9007 import make_factory
 from query_long_projected_r323 import query_long_projected
 from query_window_r369 import query_long_projected as query_window
 config_raw=(incoming/'config.json').read_bytes()
 assert hashlib.sha256(config_raw).hexdigest()=='aac0900105b9099b99a452c728ff9c4685889d264f7bed0101e57788df8f09c8'
 cfg=json.loads(config_raw)
 cfg.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
 cfg['dflash_config']=dict(cfg['dflash_config'],selector_rank=8,selector_top_k=4)
 torch.set_num_threads(1);torch.manual_seed(369)
 model=make_factory(cfg,incoming)().eval();head=nn.Linear(32,cfg['vocab_size'],bias=False).eval()
 before={k:v.detach().clone()for k,v in model.state_dict().items()}
 rows=[]
 class PlainCrop(nn.Module):
  def __init__(self,base,index):super().__init__();self.base=base;self.index=index
  def forward(self,x,ctx,dp,cp,mask):
   if self.index<4:
    L=ctx.shape[1];start=max(0,L-4095)
    ctx=ctx[:,start:];cp=cp[:,start:]
    mask=torch.cat([mask[... ,start:L],mask[...,L:]],-1)
   return self.base(x,ctx,dp,cp,mask)
 with torch.inference_mode():
  for L in [1,1024,4095,4096,8635,32638,32760]:
   cp=torch.arange(L).view(1,-1);dp=L+torch.arange(8).view(1,-1)
   keypos=torch.cat([cp,dp],1);mask=(dp[:,:,None]-keypos[:,None,:]).abs()<4096
   start=max(0,L-4095)
   assert not bool(mask[:,:,:start].any())
   assert torch.equal(mask[:,:,:L].any(1)[0],cp[0]>=max(0,L-4095))
   assert bool(mask[:,:,L:].all())
   ctx=torch.randn(1,L,32)*.1;ctx_before=ctx.clone()
   norm=model.context_norm(ctx)
   assert torch.equal(norm[:,start:],model.context_norm(ctx[:,start:]))
   off,debug=query_long_projected(model,head,[ctx],[163584],[7],enabled=True,return_debug=True,profile=False)
   assert torch.isfinite(debug['hidden']).all()
   old=model.layers;model.layers=nn.ModuleList([PlainCrop(layer,i)for i,layer in enumerate(old)])
   error=None
   try:
    cropped,cd=query_long_projected(model,head,[ctx],[163584],[7],enabled=True,return_debug=True,profile=False)
    assert L<=4095
    assert torch.equal(off,cropped)and torch.equal(debug['hidden'],cd['hidden'])
   except IndexError as e:
    error=str(e);assert L>4095
   finally:model.layers=old
   original_layers=[id(layer)for layer in model.layers]
   off_window,wd=query_window(model,head,[ctx],[163584],[7],enabled=True,return_debug=True,profile=False)
   cropped,cd=query_window(model,head,[ctx],[163584],[7],enabled=True,return_debug=True,profile=False,crop_local=True)
   assert torch.allclose(debug['hidden'],wd['hidden'],rtol=1e-6,atol=1e-6)
   diffs={}
   for name in ['hidden','logits']:
    a=wd[name].float();b=cd[name].float()
    nr=float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-30))
    assert nr<2e-4 and bool(torch.isfinite(b).all())
    diffs[name]=dict(NRMSE=nr,maxabs=float((a-b).abs().max()))
   assert original_layers==[id(layer)for layer in model.layers]
   assert torch.equal(ctx,ctx_before)
   rows.append(dict(L=L,crop_start=start,remaining_context_rows=L-start,global_absolute_dp=[L,L+7],crop_seq_len=L-start+8,absolute_span=L+8,all_removed_context_columns_false=True,all8_draft_columns_retained=True,normal_off_query_passed=True,plain_crop_IndexError=error,absolute_span_candidate_passed=True,numeric=diffs,proposal_equal=bool(torch.equal(off_window,cropped)),candidate_module_routing_restored=True))
 assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_loaded=False,actual_small_pinned_factory=True,actual_query_SHA256=hashlib.sha256((incoming/'Kimi-long-draft-RPC-CPU-r333/query_long_projected_r323.py').read_bytes()).hexdigest(),cases=rows,proof='K/V projections and RMSNorm are tokenwise; static YaRN value for an absolute position is independent of context extent. Direct unchanged layer call nevertheless slices RoPE table to cropped ctx+8 and indexes original absolute positions: invalid above4095. Candidate private forward only replaces seq_len with original absolute span L+8.',decision='CPU_ONLY_SOURCE_PROOF; real checkpoint/GPU speed/numerics and acceptance UNKNOWN',weights_and_input_unchanged=True)))

if __name__=='__main__':main()
