"""Original346 absolute deadline/guard, with ordinary-only Settings in bootstrap."""
from pathlib import Path
import os,json,time
class Gate:
 def __init__(self,settings,clock=time.monotonic,wall=time.time):
  self.settings=settings;self.clock=clock;self.deadline=clock()+max(0.,settings.deadline-wall());self.counts={};self.signal_cancel=False
 def cancelled(self):return self.signal_cancel or self.settings.cancel.exists()
 def check(self,label):
  self.counts[label]=self.counts.get(label,0)+1
  if self.clock()>=self.deadline:raise TimeoutError('Original Native deadline before '+label)
  if self.cancelled():raise RuntimeError('Original controller cancellation before '+label)
 def remaining(self):return max(0.,self.deadline-self.clock())

class GuardedFunction:
 def __init__(self,raw,gate,name):object.__setattr__(self,'raw',raw);object.__setattr__(self,'gate',gate);object.__setattr__(self,'name',name)
 def __getattr__(self,name):return getattr(self.raw,name)
 def __setattr__(self,name,value):setattr(self.raw,name,value)
 def __call__(self,*args,**kwargs):
  if self.name not in ['modern_kimi_engine_destroy_r2475','modern_kimi_engine_error_r2475']:self.gate.check('Native:'+self.name)
  return self.raw(*args,**kwargs)

class GuardedLibrary:
 def __init__(self,raw,gate):self.raw=raw;self.gate=gate;self.functions={}
 def __getattr__(self,name):
  if name not in self.functions:self.functions[name]=GuardedFunction(getattr(self.raw,name),self.gate,name)
  return self.functions[name]

def guarded_safe_open(function,gate):
 def call(*args,**kwargs):gate.check('safe_open');return function(*args,**kwargs)
 return call

def publish_once(path,value):
 with Path(path).open('x',encoding='utf8',newline='\n')as f:
  f.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
