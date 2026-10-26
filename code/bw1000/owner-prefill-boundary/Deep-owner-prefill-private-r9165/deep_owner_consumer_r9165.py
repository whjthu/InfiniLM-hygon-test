"""Default-OFF existing owner-event CAPI consumer. No Runtime/device creation."""
import ctypes as C,json,math,time,statistics,hashlib
STAGES=['decoder_envelope','metadata_validation_readback','attention_HC_pre','attention_norm','attention','attention_HC_post','FFN_envelope','FFN_HC_pre','FFN_norm_sync','MoE_envelope','FFN_HC_post','router','routed_experts','shared_experts','TP_reduction','output_cast']

def bind_owner(lib):
 lib.deep_owner_arm_r8900.argtypes=[C.c_int];lib.deep_owner_arm_r8900.restype=C.c_int
 lib.deep_owner_read_r8900.argtypes=[];lib.deep_owner_read_r8900.restype=C.c_char_p

def validate_owner(raw,expected_window):
 q=json.loads(raw);assert q['schema']==1 and q['window']==expected_window and len(q['rows'])==344
 assert {(x['device'],x['layer'])for x in q['rows']}=={(d,l)for d in range(8)for l in range(43)}
 perstage={name:[0.0]*8 for name in STAGES}
 for row in q['rows']:
  dev=row['device'];assert row['markers']==32 and row['owner']and isinstance(row['stream'],int)and row['stream']>=0;assert row['event_create_during_device']==row['event_query_during_device']==dev
  spans=row['spans'];assert len(spans)==16 and[x['stage']for x in spans]==list(range(16))
  for span in spans:
   assert all(math.isfinite(span[k])and span[k]>=0 for k in ['start_ms','end_ms','stream_ms','host_wall_ms','thread_cpu_ms']);assert span['end_ms']>=span['start_ms'];assert abs(span['stream_ms']-(span['end_ms']-span['start_ms']))<0.01
   perstage[STAGES[span['stage']]][dev]+=span['stream_ms']
 return {'raw_owner_events':q,'owner_rows':344,'markers_each_row':32,'spans_each_row':16,'perstage_perdevice_sum_layers_stream_elapsed_ms':perstage,'scope':'16 nested spans MUST NOT be added across stages; device sums MUST NOT be added across devices. GPU event stream elapsed includes launch gaps/dependency/communication waits, NOT pure GPU busy time. Only each named stage individually summed over sequential layers.','raw_activation_captured':False}

class OwnerWindowStillArmed(RuntimeError):
 def __init__(self,profile):self.profile=profile;super().__init__('owner observation failed and native window not closed; no further forward permitted')

def owner_forward_once(lib,forward,check,window=1):
 assert window in [1,2];arm_start=time.perf_counter();check(lib.deep_owner_arm_r8900(window));arm_end=time.perf_counter();tick=time.perf_counter();rc=forward();ended=time.perf_counter();check(rc)
 read_start=time.perf_counter();raw=lib.deep_owner_read_r8900();read_end=time.perf_counter();raw=bytes(raw)if raw else None
 try:
  assert raw,'owner read failed/incomplete';v=validate_owner(raw,window);v['owner_event_validation_passed']=True
 except (AssertionError,ValueError,TypeError,KeyError)as e:
  v={'owner_event_validation_passed':False,'observer_error':repr(e),'read_bytes':len(raw)if raw else 0,'unqualified_read_SHA256':hashlib.sha256(raw).hexdigest()if raw else None,'raw_activation_captured':False}
 checked=time.perf_counter();closed=lib.deep_owner_arm_r8900(0)==0;validated=time.perf_counter();v.update(arm_wall_s=arm_end-arm_start,native_forward_wall_s=ended-tick,read_native_and_copy_wall_s=read_end-read_start,Python_validation_wall_s=checked-read_end,window_closed_check_wall_s=validated-checked,observer_total_wall_s=validated-arm_start,read_bytes=len(raw)if raw else 0,forward_start=tick,forward_end=ended,window=window,native_window_closed=closed)
 if not closed:raise OwnerWindowStillArmed(v)
 return rc,ended,tick,v
