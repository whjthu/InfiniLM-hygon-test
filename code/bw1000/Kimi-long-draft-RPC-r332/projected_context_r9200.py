"""Only rowwise target-feature projection cache; no draft attention/layer KV cache."""
import copy,threading
from draft_context_r9128 import ContextStore
class ProjectedContextStore(ContextStore):
 def __init__(self,model,weight_tag,max_requests=32):
  super().__init__(max_requests);assert isinstance(weight_tag,str)and weight_tag;self.model=model;self.weight_tag=weight_tag;self._sealed_tag=weight_tag;self.owner=threading.get_ident();self.projection_calls=0;self.projected_rows=0;self.signature=self._signature()
 def _signature(self):
  result=[]
  for prefix,module in [('projection',self.model.context_proj),('norm',self.model.context_norm)]:
   result.append((prefix,'module',id(module),type(module).__name__,bool(module.training),getattr(module,'variance_epsilon',None)))
   for name,p in module.named_parameters():
    try:version=p._version
    except RuntimeError:version=None
    result.append((prefix,name,id(p),p.data_ptr(),tuple(p.shape),str(p.dtype),str(p.device),version))
  assert result;return tuple(result)
 def _check(self):
  assert threading.get_ident()==self.owner,'foreign cache owner';assert self.weight_tag==self._sealed_tag,'weight tag changed';assert self._signature()==self.signature,'projection weights changed: resetcache/newweighttag required'
 def begin(self,epoch):self._check();return super().begin(epoch)
 def fork(self):
  self._check();x=copy.copy(self);x.values=self.values.copy();x.retired=self.retired.copy();return x
 def append_many(self,epoch,items,parts,concat=None,profile=False):
  self._check();self.validate_many(epoch,items);assert len(parts)==len(items)
  import torch
  weight=self.model.context_proj.weight;assert self.model.context_proj.bias is None;rawdim=int(weight.shape[1]);h=int(weight.shape[0]);assert rawdim==5*h
  for q,p in zip(items,parts):
   assert p.ndim==3 and tuple(p.shape)==(1,q['rows'],rawdim)and p.dtype==weight.dtype and p.device==weight.device
   assert q['past']==0 or q['rows']<=8,'append only committed target prefix1..8'
  points=[]
  def point(label):
   if profile:
    event=torch.cuda.Event(enable_timing=True);event.record();points.append((label,event))
  point('append_start')
  # One GEMM and one rowwise RMSNorm for every request append, never per-request GEMM.
  packed=torch.cat([p[0]for p in parts],dim=0);point('new_rows_packed');projected=self.model.context_norm(self.model.context_proj(packed));assert tuple(projected.shape)==(sum(q['rows']for q in items),h);point('new_rows_projected_norm')
  staged={};offset=0
  for q in items:
   key=(epoch,q['request']);old,value=self.values.get(key,(0,None));new=projected[offset:offset+q['rows']].unsqueeze(0);offset+=q['rows'];staged[key]=(old+q['rows'],new if value is None else(torch.cat([value,new],dim=1)if concat is None else concat(value,new)))
  self.values.update(staged);self.projection_calls+=1;self.projected_rows+=len(packed);point('appended_ready');self.last_profile=points;self.last_append={'requests':len(items),'packed_new_rows':len(packed),'projection_calls':1,'raw_dim':rawdim,'projected_dim':h};return[self.values[(epoch,q['request'])][1]for q in items]
 def get_many(self,epoch,ids):
  self._check();assert epoch==self.epoch and len(set(ids))==len(ids);return[self.values[(epoch,i)][1]for i in ids]
 def retire(self,epoch,request):self._check();return super().retire(epoch,request)
 def close(self):
  assert threading.get_ident()==self.owner,'foreign cache owner';return super().close()
