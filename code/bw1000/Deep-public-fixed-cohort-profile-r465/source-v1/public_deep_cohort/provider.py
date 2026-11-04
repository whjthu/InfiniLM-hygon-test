"""Actual public engine factory; a queue is never mistaken for a ready Model."""
from pathlib import Path
from .actors import Actors

class DeepProvider:
 def __init__(self,config,prepared):
  self.config,self.prepared=dict(config),prepared;self.group=Actors(config,prepared)
  self._used=self._ready=self._stopped=False
 def __call__(self,**settings):
  if self._used:raise RuntimeError('Deep provider factory consumed twice')
  assert Path(settings['model_path']).resolve()==Path(self.config['model_path']).resolve()
  expected=dict(device='cuda',dtype='bfloat16',tensor_parallel_size=8,pipeline_parallel_size=1,pipeline_parallel_stage=0,cache_type='static',max_batch_size=4,max_cache_len=32768,max_tokens=4096,temperature=0.,top_p=1.,top_k=1,enable_graph=False,skip_load=False,enable_prefix_caching=False,attn_backend='default',use_mla=False,moe_ep_backend='disabled',moe_ep_size=1,weight_load_mode='async',kv_transfer_config=None,pre_transpose=False)
  for key,value in expected.items():assert settings[key]==value,(key,settings[key],value)
  self._used=True;return self
 def __getattr__(self,name):return getattr(self.prepared['engine'],name)
 def start(self):
  if self._ready or self._stopped:raise RuntimeError('Deep provider cannot restart')
  self.prepared['engine'].start()
  try:
   self.group.start();row=self.group.wait_ready();self.group.run.write('MODEL_READY.json',dict(Model_READY=True,Native_constructed_loaded_by_provider=True,Native_internal_owner_count=8,TP=8,PP=1,HC_mode=0,Native_SHA256=self.config['Native_SHA256'],prepared_identity=self.prepared['identity'],fixed_profile=row['fixed_profile'],rows=4,capacity_per_row=32768,standard_public_C1=False));self._ready=True
  except BaseException as error:
   if self.group._claimed:
    self.group._trip(error)
    try:self.group.stop()
    except BaseException as cleanup:
     if hasattr(error,'add_note'):error.add_note('Deep startup cleanup: '+repr(cleanup))
   raise
 def is_healthy(self):return self._ready and not self._stopped and not self.group.failed.is_set()and self.prepared['engine'].is_healthy()
 def stop(self):
  if self._stopped:return
  self._stopped=True;self._ready=False;self.group.stop()
