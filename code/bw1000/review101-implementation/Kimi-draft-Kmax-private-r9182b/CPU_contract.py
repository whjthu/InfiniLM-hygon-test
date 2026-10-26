import sys,importlib.util
sys.path.append('/refactor/incoming')
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
assert callable(load_pinned_worker)and callable(ExternalTargetHead)
import torch,pathlib,json
from query_Kmax_r9182 import query_batched
class Model:
 def __init__(self):self.context_proj=lambda x:x[...,:4];self.context_norm=lambda x:x;self.final_norm=lambda x:x;self.layers=[self.layer for i in range(5)];self.candidate_selector=self;self.layer_calls=[];self.head_shapes=[];self.selects=0
 def embed_tokens(self,ids):return ids[:,:,None].float().expand(-1,-1,4)/163840
 def layer(self,x,ctx,dp,cp,mask):self.layer_calls.append((list(x.shape),list(dp.shape),list(cp.shape),list(mask.shape)));return x+.0001
 def head(self,x):self.head_shapes.append(list(x.shape));return x[...,:1].expand(-1,-1,16)
 def score_candidates(self,hidden,logits,prev):self.selects+=1;return torch.tensor([1.,0.]).expand(len(prev),-1),torch.stack([prev+1,prev+2],-1)
records=[]
with torch.inference_mode():
 for C in [1,2,4,8,16,32]:
  for Ks in [[k]*C for k in [1,3,7]]+[[[1,3,7][i%3]for i in range(C)]]:
   contexts=[torch.zeros((1,1+i%3,35840))for i in range(C)];anchor=list(range(100,100+C));m=Model();a,da=query_batched(m,m.head,contexts,anchor,Ks,enabled=True,return_debug=True,crop_head=False);oldlayers=list(m.layer_calls);assert m.selects==7 and m.head_shapes==[[C,7,4]];m=Model();b,db=query_batched(m,m.head,contexts,anchor,Ks,enabled=True,return_debug=True,crop_head=True);assert b.shape==(C,max(Ks))and m.selects==max(Ks)and m.head_shapes==[[C,max(Ks),4]]and m.layer_calls==oldlayers;assert all(x[0][1]==8 for x in m.layer_calls);assert torch.equal(a[:,:max(Ks)],b);assert not db['event_points'];records.append({'C':C,'Ks':Ks,'fixed_positions':8,'head_positions':max(Ks),'same5layers_mask_positions':True})
 for bad in [0,8,-1]:
  try:query_batched(Model(),Model().head,[torch.zeros((1,1,35840))],[100],[bad],enabled=True);raise RuntimeError('bad K accepted')
  except AssertionError:pass
assert not torch.cuda.is_initialized();q={'GPU_entry_top_import_closure_tested':True,'CPU_only':True,'GPU':False,'actual_Torch_CPU_geometry':True,'trained_model_math_tested':False,'cases':records,'bad_K_rejected':[0,8,-1],'target_acceptance_qualified':False};out=pathlib.Path(__file__).with_name('CPU-contract-r9182.json');out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
