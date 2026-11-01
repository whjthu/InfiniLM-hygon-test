"""Summarize observer perturbation and current inclusive buckets; never speed gain."""
import pathlib,json,hashlib,statistics

def analyze(q,verify_packets=True):
 assert q['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 assert q['completed']and q['passed']and q['normal_server_shutdown']and q['normal_Native_shutdown']
 assert q['owner_exit_observation']['all8_unique_threads_exited']and q['actual_HTTP']==q['actual_Native']==6
 rows=q['requests'];assert len(rows)==6
 for i,r in enumerate(rows):
  assert r['prompt_tokens']==8489 and r['max_new_tokens']==128 and r['native_finish_reason']=='eos_token'
  assert r['prefill_binding']['prefill_chunk']==1024 and r['prefill_binding']['observer_enabled']==(i in [3,4])
  assert [v['geometry']['decode_call_1based']for v in r.get('owner_profiles',[])]==([8,87]if i in [3,4]else[])
  for v in r.get('owner_profiles',[]):
   assert v['native_window_closed']and v['owner_event_validation_passed']and v['inner_event_validation_passed']
   if verify_packets:
    for p in v['scalar_packets'].values():assert hashlib.sha256(pathlib.Path(p['path']).read_bytes()).hexdigest()==p['SHA256']
 def pair(values):
  a=[values[2],values[5]];b=[values[3],values[4]]
  return dict(OFF=a,ON=b,ON_over_OFF=statistics.mean(b)/statistics.mean(a),OFF_endpoint_drift=a[1]/a[0]-1,ON_endpoint_drift=b[1]/b[0]-1)
 sampled={}
 for index in [8,87]:
  times=[]
  for r in rows:
   v=next(v for v in r['forward_timings']if v['kind']=='decode'and v['decode_call_1based']==index)
   assert v['past']==8489+index-1;times.append(v['native_forward_wrapper_s'])
  sampled[str(index)]=pair(times)
 return dict(passed=True,scope='Current8K C1 attribution only; two warmups excluded; observer ON/OFF is not an optimization.',sampled_forward_wrapper=sampled,decode_wrapper=pair([sum(v['native_forward_wrapper_s']for v in r['forward_timings']if v['kind']=='decode')for r in rows]),profiles=[dict(request_index=i,profiles=r.get('owner_profiles',[]))for i,r in enumerate(rows)if i in [3,4]],timing_limit='Nativewrapper includes argument packaging/CAPI/drain; owner16 spans nested, sixattention segments siblings. Never sum nested stages or differentowners. Stream intervals include launch gaps, dependencies, communication and sync; no GPUbusy or pureNIC attribution.',boundary_limit='call8 past8496 is noncompletion; call87 past8575 completes C4/C128. Neighbor7/9,86/88 differ in state and boundary, so neighboring time difference alone is not observer overhead.',raw_packet_policy='Only paths/SHA/selected aggregate metadata; raw packets retained on experimental host.')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('output');a=p.parse_args();out=pathlib.Path(a.output);assert not out.exists();out.write_text(json.dumps(analyze(json.loads(pathlib.Path(a.report).read_bytes())),indent=2)+'\n')
