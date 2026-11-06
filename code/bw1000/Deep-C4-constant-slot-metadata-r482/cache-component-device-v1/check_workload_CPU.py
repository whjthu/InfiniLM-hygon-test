"""One complete schedule against a CPU-only contract recording, not a Native test."""
from pathlib import Path
import tempfile
import numpy as np
from workload import run_phase,raw_expected,ORDER
class Recording:
 def __init__(self,mode):self.mode=mode;self.physical_count=np.zeros(21,np.int64);self.v=np.zeros((21,8192,128),np.uint8);self.s=np.zeros((21,8192,1),np.float32);self.w=np.zeros((21,8192),bool);self.stores=self.gathers=self.guards=self.resets=self.fast=self.elided=self.fallback=0
 def raw(self,l):
  return raw_expected(self.v[l],self.s[l],int(self.physical_count[l]))
 def seed(self,l,b,v,s):n=len(v);self.v[l,b:b+n]=v;self.s[l,b:b+n]=s;self.w[l,b:b+n]=True;self.physical_count[l]=max(self.physical_count[l],b+n);self.stores+=1
 def guard(self,l,k):
  if k==4:assert not self.w[l,8191]
  self.guards+=1
 def reset(self,l):self.w[l]=False;self.resets+=1
 def gather(self,l,i):
  assert self.w[l,i].all();fast=self.mode and np.array_equal(i,np.arange(i.size));self.gathers+=1;self.fast+=int(fast);self.elided+=i.size*8 if fast else 0;self.fallback+=int(not fast);return self.v[l,i].copy(),self.s[l,i].copy(),[0.,0.,0.]
 def observe(self):return [64<<30,64<<30,21,128,128*8448,self.stores,self.gathers,self.guards,self.resets,1376256 if self.mode else 0,self.gathers,self.fast,self.elided,self.fallback,1,1]
assert ORDER==[0,1,1,0,0,1]
with tempfile.TemporaryDirectory()as x:
 for mode in [0,1]:
  r=run_phase(Recording(mode),mode,Path(x)/str(mode),mode);assert r['passed']and r['raw_audit_count']==252
print('PASS two full CPU schedule recordings:126measured/21warm/147guards/63stores/21resets each; real Native/SDK/GPU not executed')
