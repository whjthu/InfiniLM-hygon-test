"""ROWS32 resident owner wrapper. Imports are CPU-only; instantiated inside GPU worker."""
import ctypes,json,pathlib
from glm_crossrequest_resident_base_r8866 import ResidentControl as BaseControl,CandidateRejected,BOUNDS,RESERVE,ALLOWANCE,FLOOR,BANK_BYTES
class CrossrequestControl(BaseControl):
 def __init__(self,lib,ctx,stage,rank,control,check,report,save):
  lib.modern_glm_vector_policy_r8540.argtypes=[ctypes.c_int,ctypes.c_int]
  lib.modern_glm_vector_get_r8540.argtypes=[ctypes.c_int]
  lib.modern_glm_vector_counts_r8540.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  lib.modern_glm_projected_host_counts_r4504.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  lib.modern_glm_model_transient_policy_r5688.argtypes=[ctypes.c_void_p,ctypes.c_int]
  lib.modern_glm_model_transient_report_r5688.argtypes=[ctypes.c_void_p];lib.modern_glm_model_transient_report_r5688.restype=ctypes.c_char_p
  super().__init__(lib,ctx,stage,rank,control,check,report,save)
  q=self.snapshot();assert not q['transient']['enabled'] and not q['transient']['busy'] and not q['transient']['poison'] and q['transient']['GPU_logical_bytes']==0
  assert (q['transient']['rows'],q['transient']['capacity'])==(32,640)and q['vector_mode']==0
  cfg=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_text());self.sparse=sum(x=='sparse'for x in cfg['mlp_layer_types'][BOUNDS[stage]:BOUNDS[stage+1]])
  report.update(transient_fixed_OFF_no_Scratch_alloc=True,vector_fixed_ON_in_all_paired_arms=True,constructor_ROWS=32)
 def vector_OFF(self):return self.lib.modern_glm_vector_policy_r8540(self.rank,0)
 def snapshot(self):
  q=super().snapshot()
  a=(ctypes.c_uint64*7)();self.check(self.lib.modern_glm_projected_host_counts_r4504(self.rank,a,7));q['host_counters']=list(a)
  b=(ctypes.c_uint64*8)();self.check(self.lib.modern_glm_vector_counts_r8540(self.rank,b,8));q['vector_counts']=list(b)
  q['vector_mode']=self.lib.modern_glm_vector_get_r8540(self.rank)
  q['transient']=json.loads(self.lib.modern_glm_model_transient_report_r5688(self.ctx));return q
 def apply(self,label,on,trace):
  try:
   super().apply(label,on,trace)
   code=self.lib.modern_glm_vector_policy_r8540(self.rank,1)
   if not self.vote(code==0):raise CandidateRejected('fixed vectorON setter rejected while EMPTY')
  except CandidateRejected:
   assert self.vote(self.lib.modern_glm_model_reset(self.ctx)==0),'rollback reset allowners'
   a=self.setter([0]*78,False);b=self.vector_OFF()
   assert self.vote(a==0 and b==0),'rollback resident/vector OFF allowners'
   raise
  q=self.snapshot();assert q['vector_mode']==1 and not q['transient']['enabled']and q['transient']['GPU_logical_bytes']==0
  self.report['resident_policy_phases'][-1]['native']=q;self.save();return q
 def finish(self,label,before,events,prompts):
  super().finish(label,before,events);q=self.snapshot();L=BOUNDS[self.stage+1]-BOUNDS[self.stage];host=L-self.count
  delta=[b-a for a,b in zip(before['host_counters'],q['host_counters'])]
  expected=[0,0,0,2*host*sum(len(e['rows'])for e in events),8192*host*sum(len(prompts[i])+e['round']for e in events for i in e['rows']),2*host*sum(len(e['rows'])for e in events),8192*host*sum(e['flat_tokens']for e in events)]
  assert delta==expected,('host4504 resident suppression',label,delta,expected)
  hit=[e for e in events if e['flat_tokens']in(4,8,16)];n=self.sparse*len(hit);fallback=self.sparse*(len(events)-len(hit));rows=self.sparse*sum(e['flat_tokens']for e in hit)
  vd=[b-a for a,b in zip(before['vector_counts'],q['vector_counts'])];ve=[n,fallback,n,fallback,0,rows,rows,0]
  assert vd==ve and q['vector_mode']==1,('fixed vector dispatch',label,vd,ve)
  assert q['transient']==before['transient'],'ROWS32 transient must remain disabled/noalloc'
  self.report['resident_phase_after'][-1].update(native=q,host_counter_delta=delta,expected_host_counter_delta=expected,vector_counter_delta=vd,vector_counter_expected=ve)
  self.save();return q
