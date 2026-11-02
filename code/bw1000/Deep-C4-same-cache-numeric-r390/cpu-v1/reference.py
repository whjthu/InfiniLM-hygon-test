"""Independent CPU layout/FP8 score oracle for pinned current Hygon provider.

Scores keep the provider's 16-lane dot and 128-contribution addition trees.
All arithmetic arrays are F32, with separate multiply/add; no GPU F64.
"""
import numpy as np
PAGE_SLOTS=64
KEY_BYTES=128
PAGE_BYTES=8448
SCALE_OFFSET=8192

def gather_paged(raw,slots,written):
 raw=np.asarray(raw,dtype=np.uint8).reshape(-1,PAGE_BYTES);slots=np.asarray(slots,dtype=np.int64)
 assert slots.ndim==1 and len(slots)>0 and np.all(slots>=0)and np.all(slots<len(raw)*64)
 assert all(bool(written[int(s)])for s in slots),'Unwritten slot must fail before scoring'
 values=np.empty((len(slots),128),np.uint8);scales=np.empty((len(slots),1),np.float32)
 for i,slot in enumerate(slots):
  page=int(slot)//64;row=int(slot)%64;values[i]=raw[page,row*128:(row+1)*128]
  scales[i,0]=raw[page,8192+row*4:8192+(row+1)*4].copy().view(np.float32)[0]
 assert np.isfinite(scales).all()
 return values,scales

def geometry(past):
 assert type(past)is int and 0<=past<32768
 return dict(T=1,heads=8,dim=128,before=past//4,after=(past+1)//4,topk=512,pruning_required=(past+1)//4>512)


def decode_fp8(code):
 import math
 if type(code)is not int or not 0<=code<=255 or(code&127)==127:
  raise ValueError('Invalid current E4M3FN code')
 exponent=(code>>3)&15;mantissa=code&7
 value=math.ldexp(1+mantissa/8,exponent-7)if exponent else math.ldexp(mantissa,-9)
 return -value if code&128 else value


def score_cpu(query,weights,keys,scales,visible):
 query=np.asarray(query,dtype=np.uint8);keys=np.asarray(keys,dtype=np.uint8)
 weights=np.asarray(weights,dtype=np.float32);scales=np.asarray(scales,dtype=np.float32)
 if query.shape!=(8,128)or keys.ndim!=2 or keys.shape[1]!=128:
  raise ValueError('Only pinned T1/H8/D128 geometry')
 count=keys.shape[0]
 if weights.shape!=(8,)or scales.shape!=(count,)or type(visible)is not int or not 0<=visible<=count:
  raise ValueError('Invalid score count/visibility')
 out=np.full(count,-np.inf,dtype=np.float32)
 if visible==0:return out
 if np.any((query&127)==127)or np.any((keys[:visible]&127)==127):
  raise ValueError('FP8 invalid code')
 if not np.isfinite(weights).all()or not np.isfinite(scales[:visible]).all()or np.any(scales[:visible]<=0):
  raise ValueError('Invalid head weights/key scale')
 # All valid values are exactly representable binary F32 values.
 lut=np.array([decode_fp8(i)if(i&127)!=127 else 0 for i in range(256)],dtype=np.float32)
 q=lut[query];k=lut[keys[:visible]]
 partial=np.zeros((8,visible,16),dtype=np.float32)
 for offset in range(8):
  ds=np.arange(16)+16*offset
  product=np.multiply(q[:,None,ds],k[None,:,ds],dtype=np.float32)
  partial=np.add(partial,product,dtype=np.float32)
 for span in [8,4,2,1]:
  partial[:,:,:span]=np.add(partial[:,:,:span],partial[:,:,span:2*span],dtype=np.float32)
 scaled=np.multiply(partial[:,:,0].T,scales[:visible,None],dtype=np.float32)
 contribution=np.zeros((visible,128),dtype=np.float32)
 contribution[:,:8]=np.multiply(np.maximum(scaled,np.float32(0)),weights[None,:],dtype=np.float32)
 for span in [64,32,16,8,4,2,1]:
  contribution[:,:span]=np.add(contribution[:,:span],contribution[:,span:2*span],dtype=np.float32)
 out[:visible]=contribution[:,0]
 return out
