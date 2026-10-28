"""CPU-callable metadata contracts; no Runtime/GPU/network/secret handling."""
import ctypes as C
import os

def validate_loaded_registry(report):
 assert os.environ.get('INFINILM_LOADING_BINDINGS_R9212')=='1','r162a fixed cacheON'
 rows=report['registry_total_8x8_r159'];flushes=report['loader_flushes']
 assert type(flushes)is int and flushes>0 and len(rows)==8
 for row in rows:
  assert len(row)==8 and all(type(v)is int and v>=0 for v in row)
  assert [row[0],row[2],row[4],row[6]]==[2,flushes,1,flushes], 'r162a cache first+finish/lookup/load counts not qualified'
 return {'passed':True,'local_owners':8,'flushes':flushes,'state_dict_builds_each_owner':2,'cache_lookup_each_owner':flushes,'finish_each_owner':1,'fixed_cache_ON':True,'whole_load_speedup_causal_claim':False}

class W13OffProof:
 def __init__(self,lib):
  assert os.environ.get('INFINILM_KIMI_W13_WAVEPARTIAL_R155')=='0','r162a W13 staysOFF'
  self.api=lib.modern_kimi_W13_wavepartial_counts_r155;self.api.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t];self.api.restype=C.c_int
  self.before=self.snapshot()
 def snapshot(self):
  result=[]
  for device in range(8):
   row=(C.c_uint64*2)();assert self.api(device,row,2)==0;result.append(list(row))
  return result
 def finish(self):
  assert os.environ.get('INFINILM_KIMI_W13_WAVEPARTIAL_R155')=='0'
  after=self.snapshot();delta=[[b-a for a,b in zip(x,y)]for x,y in zip(self.before,after)]
  assert all(x>=0 and y==0 for x,y in delta),'r162a W13 candidate branch unexpectedly executed'
  return {'passed':True,'fixed_W13_OFF':True,'counts_delta8x2':delta,'baseline_zero_allowed_if_no_I192_shape':True}
