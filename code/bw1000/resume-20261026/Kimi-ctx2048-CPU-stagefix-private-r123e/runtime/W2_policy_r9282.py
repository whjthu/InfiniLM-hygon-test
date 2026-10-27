import ctypes,os
class W2Counter:
 def __init__(self,lib):
  self.fn=lib.modern_kimi_W2_pair_counts_r9280;self.fn.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];self.fn.restype=ctypes.c_int
 def read(self):
  rows=[]
  for device in range(8):
   a=(ctypes.c_uint64*2)();assert self.fn(device,a,2)==0;rows.append(list(a))
  return rows
 def configure(self,enabled):
  assert type(enabled)is bool;os.environ['INFINILM_KIMI_W2_PAIR_R9280']='1'if enabled else'0'
 def delta(self,before,after,enabled):
  rows=[[y-x for x,y in zip(a,b)]for a,b in zip(before,after)];assert all(all(x>=0 for x in row)and row[0 if enabled else 1]==0 for row in rows);return rows
