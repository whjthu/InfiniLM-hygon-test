"""Owner-only bounded window. ARM votes complete before original forward."""
import ctypes,json,threading,torch
class Window:
 def __init__(self,lib,ctx,stage,rank,control,report,save):
  self.lib,self.ctx,self.stage,self.rank,self.control,self.report,self.save=lib,ctx,stage,rank,control,report,save;self.owner=threading.get_ident()
  lib.modern_glm_window_prepare_r5797.argtypes=[ctypes.c_void_p];lib.modern_glm_window_arm_r5795.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int];lib.modern_glm_window_read_r5795.argtypes=[ctypes.c_void_p];lib.modern_glm_window_read_r5795.restype=ctypes.c_char_p
 def vote(self,rc):
  assert threading.get_ident()==self.owner
  x=torch.tensor([float(rc==0)],device='cuda');self.control(x);assert x.item()==16,('5799 all16 ARM/prepare failed before forward',rc,x.item())
 def prepare(self):self.vote(self.lib.modern_glm_window_prepare_r5797(self.ctx)if self.stage==1 else 0)
 def before(self,step,group):
  if step not in [63,64,65]:return
  cohort=0 if list(group)==list(range(8))else 1 if list(group)==list(range(8,16))else -1
  assert cohort>=0
  self.vote(self.lib.modern_glm_window_arm_r5795(self.ctx,step,cohort,1)if self.stage==1 else 0)
 def after(self,label,step,group):
  if step not in [63,64,65]or self.stage!=1:return
  assert threading.get_ident()==self.owner
  raw=self.lib.modern_glm_window_read_r5795(self.ctx);assert raw,'5799 window getter failed after originalCAPI'
  q=json.loads(raw);cohort=0 if group[0]==0 else 1
  assert(q['layer'],q['stage'],q['rank'],q['round'],q['cohort'],q['past'])==(38,1,self.rank,step,cohort,512+step-1)
  assert 2<=q['marker_count']<=512
  self.report.setdefault('selected_decode_window_r5799',[]).append({'phase':label,**q});self.save()
