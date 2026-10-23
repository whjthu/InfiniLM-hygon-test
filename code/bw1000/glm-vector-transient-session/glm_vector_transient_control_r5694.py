"""CPU-import-safe vector+transient owner wrapper. Not GPU qualified."""
import ctypes,json,pathlib
from glm_transient_control_r5692 import TransientControl as BaseControl,CandidateRejected
class VectorTransientControl(BaseControl):
 def __init__(self,lib,ctx,stage,rank,control,check,report,save):
  lib.modern_glm_vector_policy_r8540.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_vector_get_r8540.argtypes=[ctypes.c_int];lib.modern_glm_vector_counts_r8540.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  self.vector_requested=False;super().__init__(lib,ctx,stage,rank,control,check,report,save)
  cfg=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_text());self.sparse=sum(v=='sparse'for v in cfg['mlp_layer_types'][(0,38)[stage]:(38,78)[stage]])
  assert self.snapshot()['vector_mode']==0
 def configure(self,on):self.vector_requested=bool(on)
 def snapshot(self):
  q=super().snapshot();a=(ctypes.c_uint64*8)();self.check(self.lib.modern_glm_vector_counts_r8540(self.rank,a,8));mode=self.lib.modern_glm_vector_get_r8540(self.rank);assert mode in[0,1];q['vector_mode']=mode;q['vector_counts']=list(a);return q
 def vector_OFF(self):return self.lib.modern_glm_vector_policy_r8540(self.rank,0)
 def apply(self,label,on,trace):
  metadata=self.all_rows([self.epoch+1,int(on),int(self.vector_requested),int(trace)]);assert all(v==metadata[0]for v in metadata),'allowner vector intent differs'
  codes=self.all_rows([int(self.vector_OFF()==0)]);assert all(v[0]==1 for v in codes),'allowner vectorOFF before policy'
  try:
   super().apply(label,on,trace)
   code=self.lib.modern_glm_vector_policy_r8540(self.rank,int(self.vector_requested));status=self.all_rows([int(code==0)])
   if not all(v[0]==1 for v in status):raise CandidateRejected('vector setter allowner rejection while EMPTY')
  except CandidateRejected:
   reset=self.all_rows([int(self.lib.modern_glm_model_reset(self.ctx)==0)]);assert all(v[0]==1 for v in reset),'allowner rollback reset failed';t=self.setter(False);v=self.vector_OFF();rollback=self.all_rows([int(t==0),int(v==0)]);assert all(row==[1,1]for row in rollback),'allowner EMPTY rollback failed';raise
  q=self.snapshot();assert q['vector_mode']==int(self.vector_requested);receipt=self.report['transient_policy_phases'][-1];receipt.update(vector_on=self.vector_requested,vector_intent=metadata,native=q);self.save();return q
 def finish(self,label,before,events,prompts):
  q=super().finish(label,before,events,prompts);d=[b-a for a,b in zip(before['vector_counts'],q['vector_counts'])];on=bool(before['vector_mode']);hit=[e for e in events if on and e['flat_tokens']in[4,8]];n=self.sparse*len(hit);fallback=self.sparse*(len(events)-len(hit));rows=self.sparse*sum(e['flat_tokens']for e in hit);expected=[n,fallback,n,fallback,0,rows,rows,0]
  assert q['vector_mode']==before['vector_mode']and d==expected,('vector exact actual-shape dispatch',label,d,expected)
  self.report['transient_phase_after'][-1].update(vector_counter_delta=d,vector_counter_expected=expected);self.save();return q
