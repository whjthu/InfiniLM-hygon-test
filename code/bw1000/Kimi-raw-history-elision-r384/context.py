"""Default-OFF raw history elision; original projection/query and wire unchanged."""
from long_context_r323 import LongCommittedFeatureContext

class ProjectedOnlyContext(LongCommittedFeatureContext):
 def __init__(self,*args,discard_raw=False,**kwargs):
  super().__init__(*args,**kwargs)
  assert type(discard_raw)is bool
  self._discard_raw=discard_raw
 @property
 def discard_raw(self):return self._discard_raw
 def _append(self,epoch,request,past,part,fence,kind):
  if not self.discard_raw:return super()._append(epoch,request,past,part,fence,kind)
  self._check(epoch,request)
  assert type(past)is int and past==self.length and callable(fence)
  assert kind==self.phase and kind in ['prompt','decode']
  w=self.model.context_proj.weight
  assert self.model.context_proj.bias is None and int(w.shape[1])==5*int(w.shape[0])
  assert part.ndim==3 and part.shape[0]==1 and part.shape[2]==w.shape[1]
  assert part.device==w.device and part.dtype==w.dtype
  rows=int(part.shape[1]);assert 1<=rows<=(512 if kind=='prompt'else 8)
  assert past+rows<=(self.prompt_tokens if kind=='prompt'else 32768)
  import torch
  projected=self.model.context_norm(self.model.context_proj(part[0])).unsqueeze(0)
  assert tuple(projected.shape)==(1,rows,int(w.shape[0]))
  assert bool(torch.isfinite(projected).all()),'Nonfinite projected feature rows; do not publish'
  projected=projected if self.projected is None else torch.cat([self.projected,projected],dim=1)
  fence();self._check(epoch,request)
  self.raw=None;self.projected=projected;self.length=past+rows;self.projection_calls+=1
  self.last_append={'kind':kind,'past':past,'new_rows':rows,'length':self.length,'projection_calls':1}
  return self.projected
 def retire(self,epoch,request):
  if not self.discard_raw:return super().retire(epoch,request)
  self._check(epoch,request);removed=self.projected is not None
  self.raw=None;self.projected=None;self.phase='retired';self.last_append=None
  return removed
 def close(self):
  if not self.discard_raw:return super().close()
  self._owner();removed=self.projected is not None
  self.raw=None;self.projected=None;self.phase='closed';self.last_append=None
  return removed
 def payload_bytes(self):
  value=super().payload_bytes()
  # This is published input geometry, not wire traffic, allocator peak or HBM.
  value['published_raw_input_bytes']=self.length*int(self.model.context_proj.weight.shape[1])*self.model.context_proj.weight.element_size()
  value['raw_history_retained']=not self.discard_raw
  return value
