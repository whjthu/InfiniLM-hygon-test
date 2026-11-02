"""Actual eight Runtime-owner HOST counters; no GPU readback and no timing claim."""
import ctypes
FIELDS=('enabled_decoder_scopes','borrowed_compressor_copies','borrowed_bytes','enabled_mismatch_fallbacks','disabled_readthrough_copies')
class PositionsModel:
 def __init__(self,lib,layers):
  assert layers==43;self.lib=lib
  self.set=lib.deep_position_proof_set_r387;self.set.argtypes=[ctypes.c_int];self.set.restype=ctypes.c_int
  self.get=lib.deep_position_proof_get_r387;self.get.argtypes=[];self.get.restype=ctypes.c_int
  self.read=lib.deep_position_proof_stats_r387;self.read.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];self.read.restype=ctypes.c_int
  assert self.get()==0,'Actual Native positions default must be OFF'
  self.initial=self.snapshot();assert self.initial==[[0]*5 for _ in range(8)],'No preforward proof counts expected'
 def snapshot(self):
  rows=[]
  for device in range(8):
   out=(ctypes.c_uint64*5)();assert self.read(device,out,5)==0;rows.append(list(map(int,out)))
  return rows
 def select(self,mode):
  assert type(mode)is int and mode in [0,1]and self.set(mode)==0 and self.get()==mode
  return self.snapshot()
 def verify_forward(self,before,after,tokens,mode,kind):
  assert type(tokens)is int and tokens>=1 and mode in [0,1]and kind in ['prefill','decode']
  expected=[43,62,62*tokens*8,0,0]if mode else[0,0,0,0,62]
  delta=[[b-a for a,b in zip(old,new)]for old,new in zip(before,after)]
  assert len(delta)==8 and all(row==expected for row in delta),('Position387 actual owner coverage differs',kind,tokens,mode,delta,expected)
  return dict(kind=kind,input_tokens=tokens,mode=mode,per_owner_delta=delta,counter_fields=FIELDS,
   actual_Runtime_owner_HOST_branch_counts=True,not_hardware_GPU_hit_counter=True)
