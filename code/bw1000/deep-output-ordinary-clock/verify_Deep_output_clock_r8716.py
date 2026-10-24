"""CPU-only ordinary layer clock qualification and noncausal arrival envelopes."""
import json,pathlib,sys,math,statistics
d=json.loads(pathlib.Path(sys.argv[1]).read_text());assert d['completed']and len(d['owners'])==8
assert sorted(x['dev']for x in d['owners'])==list(range(8))
samples=[]
for o in d['owners']:
 assert o['normal_owner_closed']and len(o['arms'])==4
 for a in o['arms']:
  assert a['observer']==int(a['arm']in[1,2])and a['decode_forward_count']==35 and len(a['forwards'])==43
  clocks=[f for f in a['forwards']if f['observer_clock']is not None]
  assert [f['step']for f in clocks]==([25,35]if a['arm']in[1,2]else[])
  for f in a['forwards']:
   assert math.isfinite(f['CAPI_wall_seconds'])and f['CAPI_wall_seconds']>0
   if f['step']in[25,35]:assert {s['name']for s in f['stages']}=={'attn_value','hidden'}and all(s['numeric_guard']for s in f['stages'])
 for arm,step,seq in [(1,25,1),(1,35,2),(2,25,3),(2,35,4)]:
  f=next(f for f in o['arms'][arm]['forwards']if f['step']==step);v=f['observer_clock'];assert len(v)==8 and all(math.isfinite(x)for x in v)and v[6]==seq and v[7]==32 and v[5]>0 and all(v[i]<=v[i+1]for i in range(4))
for arm in [1,2]:
 for step in [25,35]:
  fs=[next(f for f in o['arms'][arm]['forwards']if f['step']==step)for o in d['owners']];vs=[f['observer_clock']for f in fs]
  samples.append({'arm':arm,'step':step,'CPU_enter_SUM_spread_s':max(v[2]for v in vs)-min(v[2]for v in vs),'CPU_after_drain_spread_s':max(v[4]for v in vs)-min(v[4]for v in vs),'CPU_first_enter_to_last_drain_s':max(v[4]for v in vs)-min(v[2]for v in vs),'CPU_last_enter_to_last_drain_s':max(v[4]for v in vs)-max(v[2]for v in vs),'owner_records':[{'device':o['dev'],'CPU_before_SUM_submission_s':v[2]-v[0],'CPU_SUM_enqueue_s':v[3]-v[2],'CPU_enqueue_return_to_original_drain_s':v[4]-v[3],'GPU_queue_event_span_s':v[5],'CAPI_full_layer_s':f['CAPI_wall_seconds']}for o,v,f in zip(d['owners'],vs,fs)]})
out={'passed':True,'actual_handle':d['step'],'all8normal':True,'sampled_numeric_outputs_passed':True,'clocks':samples,'wholelayer_observer_controls_s':d['arm_max_owner_seconds'],'scope':'Same actual checkpoint r7003 layer0/8owner TP8; CPU enqueue != GPUready. Eventspan includes wait/arrival/dependencies, not puretransport; no Model TPS/gain claim.','final_tokens':'NOT_MEASURED'}
print(json.dumps(out))
