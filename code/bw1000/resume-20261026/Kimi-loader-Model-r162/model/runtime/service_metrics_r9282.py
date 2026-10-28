def service_metrics(parts,arrival,request_count=32):
 assert len(parts)==2 and sum(len(p['request_metrics'])for p in parts)==request_count
 users=[]
 for part in parts:
  for item in part['request_metrics']:
   u=dict(item);u['service_arrival_native_monotonic']=arrival;u['service_queue_to_native_admission_s']=part['admission_native_monotonic']-arrival;first=u['first_ready_monotonic'];u['service_TTFT_s']=first-arrival if first is not None else None;u['service_completion_s']=u['completion_monotonic']-arrival;users.append(u)
 end=max(u['completion_monotonic']for u in users);wall=end-arrival;assert wall>0
 useful=sum(u['useful_tokens']for u in users);correctEOS=sum(u['quality']['correct']and u['EOS']for u in users)
 return {'frontend_C':32,'native_max_resident':16,'native_cohorts_serial':2,'actual_native_C32':False,'arrival_scope':'32 simultaneous arrivals at firstcohort command entry, one local emitter clock, includes preparation/reset/queue/prefill/caseboundaryreport overhead; not HTTP benchmark','arrival_native_monotonic':arrival,'end_native_monotonic':end,'service_common_wall_s':wall,'service_useful_output_tokens':useful,'service_output_tok_s':useful/wall,'service_output_tok_s_per_all33card':useful/wall/33,'correct_naturalEOS_tasks':correctEOS,'correct_naturalEOS_tasks_per_service_s':correctEOS/wall,'all_serving_GPU_cards':33,'cohort_decode_metrics':[{'label':p['label'],'decode_common_window_s':p['decode_common_window_s'],'total_useful_decode_tokens':p['total_useful_decode_tokens'],'aggregate_decode_tok_s':p['aggregate_decode_tok_s']}for p in parts],'request_metrics':users}
def screen_gate(cases):
 import re
 chosen=[c for c in cases if c.get('command',{}).get('category')=='main-C2' and re.fullmatch(r'main-C2-g0[0-3]-[01]',c.get('label',''))]
 if len(chosen)<8:return None
 assert len(chosen)==8
 by={False:{},True:{}}
 for c in chosen:
  assert c['configured_C']==2 and len(c['request_metrics'])==2
  arm=c['command']['W2pair'];assert type(arm)is bool
  for u in c['request_metrics']:
   key=u['quality']['id'];assert key not in by[arm];by[arm][key]=u
 assert len(by[False])==len(by[True])==8 and set(by[False])==set(by[True]),'fixedfirst8 mustpairbyqualityid'
 stats={str(k):{'wrong':sum(not u['quality']['correct']for u in v.values()),'nonEOS_or_cap':sum(not u['EOS']or u['truncated']for u in v.values())}for k,v in by.items()}
 stop=stats['True']['wrong']-stats['False']['wrong']>=2 or stats['True']['nonEOS_or_cap']-stats['False']['nonEOS_or_cap']>=2
 return {'stop':stop,'rule':'candidate excess >=2wrong or >=2nonEOS/cap versus matched baseline; equal caps retained not automatic veto','scope':'first4 main-C2 cohorts/8unique matchedrequests, no extraC1 or datasetselection','matched_quality_ids':sorted(by[False]),'arms':stats}
