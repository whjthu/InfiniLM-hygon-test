"""CPU packets exercise the actual existing CAPI consumer, never performance data."""
import json
from types import SimpleNamespace as NS

CFG=dict(vocab_size=129280,num_hidden_layers=43,num_nextn_predict_layers=1,compress_ratios=[0,0]+[4 if i%2==0 else 128 for i in range(41)]+[4])

def library(reset):
 kinds=[{0:0,4:2,128:1}[r]for r in CFG['compress_ratios'][:43]]
 owner=[];inner=[]
 for d in range(8):
  for l in range(43):
   owner.append(dict(device=d,layer=l,markers=32,owner=100+d,stream=0,drain_host_ms=.00754,event_create_before_device=0,event_query_before_device=d,event_create_during_device=d,event_query_during_device=d,spans=[dict(stage=i,start_ms=float(i),end_ms=i+.1,stream_ms=.1,host_wall_ms=.1,thread_cpu_ms=.1)for i in range(16)]))
   inner.append(dict(device=d,layer=l,kind=kinds[l],tokens=1,requests=1,owner=100+d,thread=100+d,stream=0,events_created=7,events_queried=7,events_destroyed=7,create_during_device=d,query_during_device=d,destroy_during_device=d,whole_stream_ms=.6,original_final_drain_host_ms=.01,segments=[dict(segment=i,start_ms=i*.1,end_ms=(i+1)*.1,stream_ms=.1,host_wall_ms=.1,thread_cpu_ms=.1)for i in range(6)]))
 rawowner=json.dumps(dict(schema=1,window=1,rows=owner)).encode()
 rawinner=json.dumps(dict(schema=1,profile='attention-inner-r9193',GPU_handles_retained=0,scope='CPU_SYNTHETIC_PACKET_NOT_PERFORMANCE',rows=inner)).encode()
 state={'owner':0,'inner':0,'arm1':0}
 def arm(kind,value):
  assert value in [0,1];state[kind]=value
  if value:state['arm1']+=1
  return 0
 def owner_arm(v):return arm('owner',v)
 def inner_arm(v):return arm('inner',v)
 def owner_read():assert state['owner']==1;return rawowner
 def inner_read():assert state['inner']==1;return rawinner
 return NS(modern_native_engine_reset=reset,deep_owner_arm_r8900=owner_arm,deep_attention_inner_arm_r9193=inner_arm,deep_owner_read_r8900=owner_read,deep_attention_inner_read_r9193=inner_read,CPU_state=state)

def profiles_for(index):
 return [dict(geometry={'decode_call_1based':v},native_window_closed=True,owner_event_validation_passed=True,inner_event_validation_passed=True)for v in ([8,87]if index in [3,4]else[])]
