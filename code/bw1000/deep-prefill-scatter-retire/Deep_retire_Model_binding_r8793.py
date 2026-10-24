"""CPU-importable helper for a future necessary Model load. No launch or DSO load."""
import ctypes,os
PACK='INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750'
RETIRE='INFINILM_DEEP_SCATTER_RETIRE_R8785'

def bind_counters(lib):
 lib.deep_prefill_retire_counts_r8785.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 lib.deep_prefill_retire_counts_r8785.restype=ctypes.c_int
 lib.deep_prefill_pack_counts_r8767.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 lib.deep_prefill_pack_counts_r8767.restype=ctypes.c_int

def configure_after_blocking_reset(lib,ctx,mode):
 """Use the original Engine reset's completed all-owner barrier, then set flags.

 The caller must retain the original single-controller/no-concurrent-forward
 contract from 8774. This helper adds no reset implementation or stream.
 """
 if type(mode)is not int or mode not in (0,1):raise ValueError('retire mode must int0/1')
 rc=lib.modern_native_engine_reset(ctx)
 if rc:raise RuntimeError('original Model blocking reset failed: '+str(rc))
 os.environ[PACK]='1'
 os.environ[RETIRE]=str(mode)

def read_counts(lib):
 result=[]
 for device in range(8):
  pack=(ctypes.c_uint64*12)();retire=(ctypes.c_uint64*4)()
  if lib.deep_prefill_pack_counts_r8767(device,pack,12)or lib.deep_prefill_retire_counts_r8785(device,retire,4):raise RuntimeError('read-only counters failed')
  result.append({'pack':list(pack),'retire':list(retire)})
 return result

def validate_full128_delta(before,after,mode):
 """Exact existing B32/M32/I512/O128 geometry; actual sumU from counters."""
 if type(mode)is not int or mode not in (0,1):raise ValueError('retire mode must int0/1')
 if len(before)!=8 or len(after)!=8:raise ValueError('requires all8owners')
 for b,a in zip(before,after):
  if any(len(x['pack'])!=12 or len(x['retire'])!=4 for x in (b,a)):raise ValueError('counter ABI extent')
  p=[y-x for x,y in zip(b['pack'],a['pack'])];v=[y-x for x,y in zip(b['retire'],a['retire'])]
  if len(p)!=12 or len(v)!=4:raise ValueError('counter ABI extent')
  u=p[3]
  if p!=[0,344,0,u,0,688,0,344,0,67633152,5461,0]or u<=0:raise ValueError('PACK ON actual fullphase counters differ')
  expected=[344,0,u,0]if mode==0 else[0,344,0,u]
  if v!=expected:raise ValueError('retire phase counters differ')
 return True
