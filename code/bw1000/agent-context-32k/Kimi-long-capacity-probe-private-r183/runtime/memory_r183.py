"""Actual global mem_get_info sampling. Native pool reserved is UNKNOWN."""
import time,threading
class MemorySampler:
 def __init__(self,torch,interval=.01):
  self.torch=torch;self.interval=interval;self.lock=threading.Lock();self.stop=threading.Event();self.minimum=None;self.latest=None;self.samples=0;self.error=None;self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def sample(self):
  values=[list(self.torch.cuda.mem_get_info(i))for i in range(8)]
  with self.lock:self.latest=values;self.samples+=1;self.minimum=values if self.minimum is None else[[min(a[0],b[0]),b[1]]for a,b in zip(self.minimum,values)]
 def loop(self):
  while not self.stop.wait(self.interval):
   try:self.sample()
   except BaseException as error:self.error=type(error).__name__;return
 def reset(self):
  self.sample()
  with self.lock:self.minimum=[list(x)for x in self.latest];self.samples=1
 def snapshot(self):
  self.sample()
  with self.lock:return {'observed_min_memory':[list(x)for x in self.minimum],'latest_memory':[list(x)for x in self.latest],'sample_count':self.samples,'interval_s':self.interval,'sampler_error':self.error,'exact_transient_peak':'UNKNOWN','Native_pool_reserved':'UNKNOWN','Torch_reserved_only':[self.torch.cuda.memory_reserved(i)for i in range(8)]}
 def close(self):self.stop.set();self.thread.join(5)
