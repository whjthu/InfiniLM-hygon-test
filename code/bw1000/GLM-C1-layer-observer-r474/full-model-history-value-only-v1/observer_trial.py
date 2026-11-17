"""Explicit typed observer adapter and frozen33request schedule; no DSO load."""
import ctypes as C,json
NAMES={'prepare':('modern_glm_global_sparse_prepare_r474',[C.c_void_p],C.c_int),'policy':('modern_glm_global_sparse_policy_r474',[C.c_void_p,C.c_int],C.c_int),'next_decode':('modern_glm_global_sparse_next_decode_r474',[C.c_void_p],C.c_int),'read':('modern_glm_global_sparse_read_r474',[C.c_void_p],C.c_char_p)}
def schedule():
 return[dict(index=0,case_index=0,diagnostic=True,screen_index=None,owner_cores=4,prefill_chunk=512,phase='case0')]+[dict(index=i+1,case_index=i,diagnostic=False,screen_index=i,owner_cores=4,prefill_chunk=512,phase='case'+str(i+1))for i in range(32)]
def phases():return['created','loaded']+[a['phase']for a in schedule()]+['quiescent','closed']
class Observer474:
 def __init__(self,library,context):
  self.ctx=context;self.prepared=False;self.enabled=False;self.mark_calls=0;self.f={}
  for k,(name,args,ret)in NAMES.items():
   f=getattr(library,name);f.argtypes=args;f.restype=ret;self.f[k]=f
 def prepare(self):
  self.ctx.owned();assert self.ctx.ready and self.ctx.past==0 and not self.prepared
  self.ctx.check(self.f['prepare'](self.ctx.ctx));self.prepared=True
 def policy(self,enabled):
  self.ctx.owned();assert self.prepared and self.ctx.past==0 and type(enabled)is bool
  self.ctx.check(self.f['policy'](self.ctx.ctx,int(enabled)));self.enabled=enabled
 def before_AR_continuation(self,full_prompt_tokens,full_prompt_complete):
  self.ctx.owned();assert self.prepared and self.enabled and full_prompt_complete is True and type(full_prompt_tokens)is int and full_prompt_tokens>0 and self.ctx.past>=full_prompt_tokens
  if self.mark_calls>=3:return False
  self.ctx.check(self.f['next_decode'](self.ctx.ctx));self.mark_calls+=1;return True
 def read(self):
  self.ctx.owned();assert self.prepared
  raw=self.f['read'](self.ctx.ctx)
  if not raw:raise RuntimeError(self.ctx.error())
  q=json.loads(raw);assert q['global_tp_rank']==self.ctx.rank and q['frame_cap']==3 and q['event_cap_per_frame']==512 and 0<=q['total_frames_used']<=3 and not q['pending_next_decode'];return q
