"""Explicit opt-in C1 prompt/committed feature consumer; no target or wire integration."""
import threading
from projected_context_r9200 import ProjectedContextStore

CAPACITY=32768
QUERY_POSITIONS=8

def plan_query(length,anchor,K):
 assert type(length)is int and 1<=length<=CAPACITY
 assert type(anchor)is int and anchor>=0 and type(K)is int and 1<=K<=7
 if length+QUERY_POSITIONS>CAPACITY:
  return {'mode':'ar_tail','reason':'fixed8_draft_positions_exceed_context','context_tokens':length,'draft_positions':0,'proposals':0}
 return {'mode':'draft','context_tokens':length,'draft_positions':QUERY_POSITIONS,'proposals':K,'last_draft_position':length+QUERY_POSITIONS-1}

class LongCommittedFeatureContext:
 """One logical request. Raw/projected tensors publish together after the caller fence."""
 def __init__(self,model,weight_tag,*,enabled=False):
  if enabled is not True:raise RuntimeError('C1 long draft consumer is default OFF')
  self.owner=threading.get_ident();self.model=model
  self.guard=ProjectedContextStore(model,weight_tag,max_requests=1)
  self.epoch=0;self.request=None;self.phase='closed';self.raw=None;self.projected=None
  self.length=0;self.prompt_tokens=0;self.last_append=None;self.projection_calls=0
 def _owner(self):assert threading.get_ident()==self.owner,'foreign long draft owner'
 def _check(self,epoch,request):
  self._owner();self.guard._check()
  assert type(epoch)is int and epoch==self.epoch
  assert type(request)is int and request==1 and request==self.request
  assert self.phase in ['prompt','decode'],'closed or retired request'
 def begin(self,epoch,request,prompt_tokens):
  self._owner();self.guard._check()
  assert self.phase in ['closed','retired'] and self.raw is None and self.projected is None
  assert type(epoch)is int and epoch>self.epoch and type(request)is int and request==1
  assert type(prompt_tokens)is int and 1<=prompt_tokens<=CAPACITY
  self.epoch=epoch;self.request=request;self.prompt_tokens=prompt_tokens
  self.phase='prompt';self.length=0;self.last_append=None;self.projection_calls=0
 def _append(self,epoch,request,past,part,fence,kind):
  self._check(epoch,request)
  assert type(past)is int and past==self.length and callable(fence)
  assert kind==self.phase and kind in ['prompt','decode']
  w=self.model.context_proj.weight
  assert self.model.context_proj.bias is None and int(w.shape[1])==5*int(w.shape[0])
  assert part.ndim==3 and part.shape[0]==1 and part.shape[2]==w.shape[1]
  assert part.device==w.device and part.dtype==w.dtype
  rows=int(part.shape[1]);assert 1<=rows<=(512 if kind=='prompt'else 8)
  assert past+rows<=(self.prompt_tokens if kind=='prompt'else CAPACITY)
  import torch
  projected=self.model.context_norm(self.model.context_proj(part[0])).unsqueeze(0)
  assert tuple(projected.shape)==(1,rows,int(w.shape[0]))
  assert bool(torch.isfinite(projected).all()),'Nonfinite projected feature rows; do not publish'
  raw=part if self.raw is None else torch.cat([self.raw,part],dim=1)
  projected=projected if self.projected is None else torch.cat([self.projected,projected],dim=1)
  # No metadata/tensor/counter publication on projection, concat, or fence failure.
  fence();self._check(epoch,request)
  self.raw=raw;self.projected=projected;self.length=past+rows;self.projection_calls+=1
  self.last_append={'kind':kind,'past':past,'new_rows':rows,'length':self.length,'projection_calls':1}
  return self.projected
 def append_prompt(self,epoch,request,past,part,fence):return self._append(epoch,request,past,part,fence,'prompt')
 def seal_prompt(self,epoch,request):
  self._check(epoch,request);assert self.phase=='prompt' and self.length==self.prompt_tokens
  self.phase='decode'
 def append_committed(self,epoch,request,past,part,fence):return self._append(epoch,request,past,part,fence,'decode')
 def query_plan(self,epoch,request,anchor,K=7):
  self._check(epoch,request);assert self.phase=='decode'
  return plan_query(self.length,anchor,K)
 def query(self,epoch,request,head,anchor,K=7,*,return_debug=False):
  plan=self.query_plan(epoch,request,anchor,K)
  if plan['mode']=='ar_tail':return dict(plan,values=None)
  from query_long_projected_r323 import query_long_projected
  values=query_long_projected(self.model,head,[self.projected],[anchor],[K],enabled=True,return_debug=return_debug,profile=False)
  return dict(plan,values=values)
 def retire(self,epoch,request):
  self._check(epoch,request);removed=self.raw is not None
  self.raw=None;self.projected=None;self.phase='retired';self.last_append=None
  return removed
 def close(self):
  self._owner();removed=self.raw is not None
  self.raw=None;self.projected=None;self.phase='closed';self.last_append=None
  return removed
 def payload_bytes(self):
  self._owner()
  return {'raw':0 if self.raw is None else self.raw.numel()*self.raw.element_size(),'projected':0 if self.projected is None else self.projected.numel()*self.projected.element_size()}
