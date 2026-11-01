"""Atomic raw/projected committed-target feature caches; per-cohort defaultOFF."""
import threading
from draft_context_r9128 import ContextStore
from projected_context_r9200 import ProjectedContextStore
class CommittedFeatureContexts:
 def __init__(self,model,weight_tag,max_requests=32):
  self.owner=threading.get_ident();self.model=model;self.raw=ContextStore(max_requests);self.projected=ProjectedContextStore(model,weight_tag,max_requests);self.enabled=False;self.last_append=None
 def check_owner(self):assert threading.get_ident()==self.owner,'foreign cache owner'
 def begin(self,epoch,projected_cache=False):
  self.check_owner();assert type(projected_cache)is bool and type(epoch)is int and epoch>self.raw.epoch and epoch>self.projected.epoch;assert not self.raw.values and not self.projected.values;self.projected._check();self.raw.begin(epoch);self.projected.begin(epoch);self.enabled=projected_cache;self.last_append=None
 def validate_many(self,epoch,items):
  self.check_owner();self.raw.validate_many(epoch,items)
  if self.enabled:self.projected.validate_many(epoch,items);self.projected._check()
  assert all(q['past']==0 or q['rows']<=8 for q in items),'only committed prefix1..8 after prefill'
 def _raw_fork(self):
  x=ContextStore(self.raw.max_requests);x.epoch=self.raw.epoch;x.values=self.raw.values.copy();x.retired=self.raw.retired.copy();return x
 def append_many(self,epoch,items,parts,fence,concat=None):
  self.validate_many(epoch,items);assert len(parts)==len(items);w=self.model.context_proj.weight
  for q,part in zip(items,parts):assert tuple(part.shape)==(1,q['rows'],int(w.shape[1]))and part.device==w.device and part.dtype==w.dtype
  import torch
  cat=concat if concat is not None else lambda a,b:torch.cat([a,b],dim=1)
  raw=self._raw_fork();projected=self.projected.fork();before=projected.projection_calls
  raw_contexts=raw.append_many(epoch,items,parts,cat)
  contexts=projected.append_many(epoch,items,parts)if self.enabled else raw_contexts
  # Publish both dictionaries only once all construction AND caller's stream fence pass.
  # Rejected verify suffix must have been cropped by the target commit protocol before this API.
  fence();self.raw=raw;self.projected=projected;self.last_append={'projected_cache':self.enabled,'new_rows':sum(q['rows']for q in items),'packed_projection_calls':projected.projection_calls-before,'lengths':[q['past']+q['rows']for q in items],'logical_requests':[q['request']for q in items]};assert self.last_append['packed_projection_calls']==int(self.enabled);return contexts
 def retire(self,epoch,request):
  self.check_owner();key=(epoch,request)
  for s in [self.raw,self.projected]:assert epoch==s.epoch and type(request)is int and 1<=request<=s.max_requests and key not in s.retired
  if self.enabled:assert self.raw.values.keys()==self.projected.values.keys()
  removed=self.raw.retire(epoch,request);self.projected.retire(epoch,request);return removed
 def close(self):
  self.check_owner();raw=self.raw.close();projected=self.projected.close();assert not self.enabled or set(raw)==set(projected);return raw
 def lengths(self):return {str(k[1]):v[0]for k,v in self.raw.values.items()}
