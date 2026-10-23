"""Shared CPU inputs and exact integer oracle for the isolated MMAC probe."""
import numpy as np
def patterns(device,op):
 assert 0<=device<8 and op in (0,1)
 K=32 if op==0 else 64;lower,upper=(-128,128)if op==0 else(-8,8)
 rng=np.random.default_rng(840500+device*2+op)
 aa=rng.integers(lower,upper,size=(16,16,K),dtype=np.int8);bt=rng.integers(lower,upper,size=(16,16,K),dtype=np.int8)
 assert set(aa.reshape(-1).tolist())==set(range(lower,upper))
 def pack(v):
  if op==0:return v.view(np.uint8).reshape(16,512)
  q=(v.astype(np.int16)&15).astype(np.uint8)
  return(q[...,0::2]|(q[...,1::2]<<4)).reshape(16,512)
 return aa,bt,pack(aa),pack(bt),aa.astype(np.int64)@bt.astype(np.int64).swapaxes(-1,-2),(lower,upper-1)
def expected_output(gold,blocks,chains,loops):
 assert blocks in (1,80,640,2560)and chains in (1,4)and 1<=loops<=8192
 groups=blocks*chains;indices=np.arange(groups)%16
 assert np.max(np.abs(gold[indices]))*loops+chains<2**31
 return(gold[indices]*loops+(np.arange(groups)%chains)[:,None,None]).astype(np.int32).reshape(-1)
