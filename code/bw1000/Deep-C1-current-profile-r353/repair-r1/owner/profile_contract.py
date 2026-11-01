"""Sparse existing Native observers; no numerical or dispatch change."""
import pathlib,time
from deep_owner_consumer_r168a import bind_inner,inner_owner_forward_once,native_expected_kinds

ARMS=(1024,)*6
OBSERVER=(False,False,False,True,True,False)
SAMPLES=(8,87)
GPU_HARD_SECONDS=1800
WORK_SECONDS=1680
MEMORY_FLOOR_BYTES=16*1024**3

def descriptor_for_index(base,index):
 if type(index)is not int or not 0<=index<6:raise ValueError('Fixed six requests exhausted')
 if base.get('rows')!=1 or base.get('capacity')!=32768:raise ValueError('Binding geometry differs')
 return dict(base,prefill_chunk=1024,arm_index=index,warmup=index<2,output_budget=128,observer_enabled=OBSERVER[index],sample_decode_calls=list(SAMPLES))

class SparseObserver:
 def __init__(self,lib,check,cfg,root):
  self.lib=lib;self.check=check;self.kinds=native_expected_kinds(cfg);self.root=pathlib.Path(root);bind_inner(lib)
  check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0))
 def forward(self,forward,args,meta,save):
  decode=bool(meta['backend_token_monotonic']);index=meta['Native_forward_host_summary']['decode_calls']+1 if decode else None
  captured=decode and meta['prefill_binding']['observer_enabled'] and index in SAMPLES
  geometry=dict(tokens=len(args[0]),requests=len(args[2]),rows=list(args[2]),past=list(args[3]),sizes=[len(args[0])],decode_call_1based=index)
  timing=dict(kind='decode'if decode else'prefill',decode_call_1based=index,past=args[3][0],tokens=len(args[0]),observer_enabled=captured)
  began=time.perf_counter();returned=[]
  def call():
   value=forward(*args);returned.append(value);return value[0]
  if captured:
   assert geometry['tokens']==geometry['requests']==1 and geometry['past']==[8489+index-1]
   prefix=self.root/f"request{meta['admitted_index']}-decode{index}"
   try:
    _,ended,started,profile=inner_owner_forward_once(self.lib,call,self.check,1,self.kinds,geometry,prefix)
   except BaseException as exc:
    if hasattr(exc,'profile'):meta.setdefault('profile_failures',[]).append(exc.profile);save()
    raise
   meta.setdefault('owner_profiles',[]).append(profile)
   timing.update(native_forward_wrapper_s=ended-started,observer_arm_s=profile['arm_wall_s'],observer_read_copy_s=profile['read_native_and_copy_wall_s'],observer_packet_io_validation_s=profile['Python_validation_and_packet_io_wall_s'],observer_total_s=profile['observer_total_wall_s'])
  else:
   started=time.perf_counter();call();timing['native_forward_wrapper_s']=time.perf_counter()-started
  timing['whole_wrapper_with_observer_s']=time.perf_counter()-began
  meta.setdefault('forward_timings',[]).append(timing)
  return returned[0]
 def complete(self,meta):
  expected=list(SAMPLES)if meta['prefill_binding']['observer_enabled']else[]
  rows=meta.get('owner_profiles',[])
  assert [v['geometry']['decode_call_1based']for v in rows]==expected,'Natural EOS before selected samples or missing capture'
  assert all(v['native_window_closed']and v['owner_event_validation_passed']and v['inner_event_validation_passed']for v in rows)
  self.check(self.lib.deep_owner_arm_r8900(0));self.check(self.lib.deep_attention_inner_arm_r9193(0))
