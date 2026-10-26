import sys,importlib.util
sys.path.append('/refactor/incoming')
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
import torch,pathlib,json
from query_flathead_r9188 import query_batched
class Model:
 def __init__(self):self.context_proj=lambda x:x[...,:4];self.context_norm=lambda x:x;self.final_norm=lambda x:x;self.layers=[self.layer for i in range(5)];self.candidate_selector=self;self.layer_calls=[];self.head_shapes=[];self.selects=0
 def embed_tokens(self,ids):return ids[:,:,None].float().expand(-1,-1,4)/163840
 def layer(self,x,ctx,dp,cp,mask):self.layer_calls.append((list(x.shape),list(dp.shape),list(cp.shape),list(mask.shape)));return x+.0001
 def head(self,x):self.head_shapes.append(list(x.shape));return x[...,:1].expand(*x.shape[:-1],16)
 def score_candidates(self,hidden,logits,prev):self.selects+=1;return torch.tensor([1.,0.]).expand(len(prev),-1),torch.stack([prev+1,prev+2],-1)
records=[]
with torch.inference_mode():
 for C in [1,2,4,8,16,32]:
  for Ks in [[k]*C for k in [1,3,7]]+[[[1,3,7][i%3]for i in range(C)]]:
   with torch.inference_mode(False):contexts=[torch.zeros((1,1+i%3,35840))for i in range(C)]
   versions=[x._version for x in contexts];anchor=list(range(100,100+C));K=max(Ks);m=Model();a,da=query_batched(m,m.head,contexts,anchor,Ks,enabled=True,return_debug=True,crop_head=True,flatten_head=False);oldlayers=list(m.layer_calls);assert m.selects==K and m.head_shapes==[[C,K,4]];m=Model();b,db=query_batched(m,m.head,contexts,anchor,Ks,enabled=True,return_debug=True,crop_head=True,flatten_head=True);assert b.shape==(C,K)and m.selects==K and m.head_shapes==[[C*K,4]]and m.layer_calls==oldlayers;assert all(x[0][1]==8 for x in m.layer_calls);assert torch.equal(a,b);assert versions==[x._version for x in contexts]
   x=torch.arange(C*8*4,dtype=torch.float32).reshape(C,8,4);prefix=x[:,1:1+K];flat=prefix.contiguous().view(C*K,4);assert torch.equal(flat.view(C,K,4),prefix);w=torch.randn(16,4);u=torch.nn.functional.linear(prefix,w);v=torch.nn.functional.linear(flat,w).view(C,K,16);err=float((u-v).abs().max());assert torch.allclose(u,v,rtol=1e-5,atol=1e-4);records.append({'C':C,'Ks':Ks,'fixed_positions':8,'head_positions':K,'same5layers_mask_positions':True,'3Dprefix_contiguous':prefix.is_contiguous(),'3Dprefix_stride':list(prefix.stride()),'flat_contiguous':flat.is_contiguous(),'row_mapping_exact':True,'F32CPU_linear_maxabs':err})
 for bad in [0,8,-1]:
  try:query_batched(Model(),Model().head,[torch.zeros((1,1,35840))],[100],[bad],enabled=True,crop_head=True,flatten_head=True);raise RuntimeError('badK')
  except AssertionError:pass
assert not torch.cuda.is_initialized();q={'CPU_only':True,'GPU':False,'actual_Torch_CPU_geometry':True,'actual_noncontiguous_row_mapping_and_2DFlinear_tested':True,'fixture_inference_context_versions_actual_checked':True,'trained_model_math_tested':False,'cases':records,'target_acceptance_qualified':False};pathlib.Path(__file__).with_name('CPU-contract-r9188.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
