"""One changed-entry recording invocation, success + failed guard branches only."""
import ctypes as C,json,time
from types import SimpleNamespace
import numpy as np
import model_driver as d
from node import first_prefill_affinity_guard

def consume(bad=False):
 events=[];resets=[];native=[];keep=[];ctx=0x12345678abcdef01;lib=SimpleNamespace()
 def forward(pointer,kind,tokens,count,flags,*args):
  assert pointer==ctx and count==1 and flags==3
  native.append((kind,tokens));events.append('Native-return:'+str(len(native)))
  np.ctypeslib.as_array((C.c_uint16*163840).from_address(args[6])).fill(0x3f80)
  C.cast(args[7],C.POINTER(C.c_int64))[0]=17 if kind==0 else 163585
  C.cast(args[8],C.POINTER(C.c_int32))[0]=3
  return 0
 f=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int,C.c_size_t,C.c_size_t,C.c_int,*([C.c_void_p]*9))(forward);setattr(lib,d.SYMBOL,f);keep.append(f)
 def reset(pointer):resets.append(pointer);events.append('reset');return 0
 for n in ['reset','finish']:
  fn=C.CFUNCTYPE(C.c_int,C.c_void_p)(reset);setattr(lib,'modern_kimi_engine_'+n+'_r2475',fn);keep.append(fn)
 for n in ['modern_kimi_engine_load_r2475','modern_kimi_shard_head_expected_r9183','modern_kimi_shard_head_error_r9183']:setattr(lib,n,SimpleNamespace())
 lib.modern_kimi_engine_error_r2475=lambda:b'recording failure'
 raw={'owners':[dict(device=i,changes=0 if bad and i==0 else 1,verified_mask=','.join(map(str,range(4*i,4*i+4))))for i in range(8)]}
 lib.modern_owner_affinity_report_r2392=lambda:json.dumps(raw).encode()
 class Control:
  def collect(self,label,p):
   events.append('collect:'+label)
   return[dict(p,node=i)for i in range(4)]
 control=Control()
 def record(event,payload):events.append(event)
 import os
 old=os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392');os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']='4'
 error=None;result=None
 try:
  result=d.generate(lib,ctx,0,control,[17],4096,time.monotonic()+30,record,after_first_prefill=lambda:first_prefill_affinity_guard(lib,control,record,0,list(range(64))))
 except AssertionError as e:error=repr(e)
 finally:
  if old is None:os.environ.pop('INFINILM_OWNER_CPU_WIDTH_R2392')
  else:os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=old
 assert len(resets)==2
 assert events.index('Native-return:1')<events.index('first_prefill_affinity_raw')<events.index('collect:first-real-prefill-affinity-all32')
 assert events.count('first_prefill_affinity_raw')==1
 if bad:
  assert error and native==[(0,1)]and 'emit'not in events and result is None
 else:
  assert error is None and native==[(0,1),(1,1)]and result['normal_reset']and result['natural_EOS']
  assert events.index('first_prefill_affinity_all32')<events.index('Native-return:2')and events.index('first_prefill_affinity_all32')<events.index('emit')
 return dict(passed=True,recording_bad_binding=bad,native_calls=native,reset_calls=len(resets),events=events,error_expected=error,planned_denominator=256,execution_model_PASS=False,recording_scope='real changed generate + real first_prefill_affinity_guard; Native/4peer/IDs leaves recording only; no actual Runtime masks or cleanup qualification')
def main():return dict(passed=True,GPU=0,new_jobs=0,Native_dlopen=False,SDK_execution=False,old_suites_reexecuted=False,changed_entry_recording=[consume(),consume(True)])
if __name__=='__main__':print(json.dumps(main(),indent=2))
