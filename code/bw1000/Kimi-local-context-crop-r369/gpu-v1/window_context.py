"""Private C1 query selection; all base append/commit/retire ownership is unchanged."""
from long_context_r323 import LongCommittedFeatureContext
from query_window_r369 import query_long_projected

class WindowContext(LongCommittedFeatureContext):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.crop_local=False;self.last_debug=None
 def query(self,epoch,request,head,anchor,K=7,*,return_debug=False):
  plan=self.query_plan(epoch,request,anchor,K)
  if plan['mode']=='ar_tail':return dict(plan,values=None)
  values,debug=query_long_projected(self.model,head,[self.projected],[anchor],[K],enabled=True,return_debug=True,profile=False,crop_local=self.crop_local)
  self.last_debug=debug
  return dict(plan,values=(values,debug)if return_debug else values)
 def retire(self,*args):
  self.last_debug=None;return super().retire(*args)
 def close(self):
  self.last_debug=None;return super().close()
