import time,threading
class MemorySampler:
 def __init__(self,torch,interval=.25):
  self.torch=torch;self.interval=interval;self.lock=threading.Lock();self.stop=threading.Event();self.minimum=None;self.latest=None;self.samples=0;self.error=None;self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def sample(self):
  vals=[list(self.torch.cuda.mem_get_info(d))for d in range(8)]
  with self.lock:
   self.latest=vals;self.samples+=1;self.minimum=vals if self.minimum is None else[[min(a[0],b[0]),b[1]]for a,b in zip(self.minimum,vals)]
 def loop(self):
  while not self.stop.wait(self.interval):
   try:self.sample()
   except BaseException as e:self.error=repr(e);return
 def reset(self):
  self.sample()
  with self.lock:self.minimum=[list(v)for v in self.latest];self.samples=1
 def snapshot(self):
  self.sample()
  with self.lock:return {'memory':[list(v)for v in self.minimum],'latest_memory':[list(v)for v in self.latest],'sample_count':self.samples,'interval_s':self.interval,'sampler_error':self.error,'scope':'read_only HIP mem_get_info sampling viaTorch; notownerRuntime, sampledmin notexactinternaltransientpeak','Torch_peak_allocated_lifetime_not_native_total':[self.torch.cuda.max_memory_allocated(d)for d in range(8)]}
 def close(self):self.stop.set();self.thread.join(5)
