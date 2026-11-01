"""CPU byte-layout oracle. FP8 arithmetic awaits the exact current provider."""
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
