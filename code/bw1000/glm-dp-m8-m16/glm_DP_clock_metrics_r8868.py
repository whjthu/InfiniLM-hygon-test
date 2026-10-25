"""Saved successful packet arrivals on one CPU clock; no Model or GPU calls."""
import statistics,math
def summary(values):return {'min':min(values),'mean':statistics.mean(values),'max':max(values)}if values else None
def percentile(values,p):
 if not values:return None
 v=sorted(values);z=(len(v)-1)*p;i=int(z);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
def common_metrics(data,reports,eos):
 assert data['finished']and data['failure']is None
 assert data['phase_arrived']==data['phase_done']==dict.fromkeys(('jointM8','jointM16'),32)
 result={}
 for phase in ('jointM8','jointM16'):
  replicas=('A','B');events=[e for e in data['records']if e['phase']==phase];assert set(e['replica']for e in events)==set(replicas)
  service={a:data['service_start_monotonic_ns'][phase+':'+a]for a in replicas};prefill={a:data['prefill_complete_monotonic_ns'][phase+':'+a]for a in replicas};anchor=max(prefill.values());start=min(service.values());release=data['gate_release_monotonic_ns'][phase];assert release<=start<anchor
  sequences={};times={};decode_intervals={};peruser=[];all_itl=[];useful_total=0;useful_after=0;input_tokens=0
  for a in replicas:
   case=next(c for c in reports[a]if c['phase']==f'perf-I512-DP-{a}-{phase}-N-r0');assert case['batch']==16 and case['micro']==(16 if phase=='jointM16'else 8)
   input_tokens+=sum(case['input_lengths'])
   arr=[e['arrival_monotonic_ns']for e in events if e['replica']==a and e['token_index']>0];decode_intervals[a]=[min(arr),max(arr)]
   for row,gold_ids in enumerate(case['generated_ids']):
    seen=[]
    for e in events:
     if e['replica']!=a or row not in e['rows']:continue
     i=e['rows'].index(row);assert e['global_rows'][i]==row+(0 if a=='A'else 16);seen.append((e['token_index'],e['ids'][i],e['arrival_monotonic_ns']))
    seen.sort();assert [v[0]for v in seen]==list(range(len(gold_ids)))and[v[1]for v in seen]==gold_ids
    assert all(seen[i][2]<=seen[i+1][2]for i in range(len(seen)-1))
    key=a+':'+str(row);sequences[key]=[v[1]for v in seen];times[key]=[v[2]for v in seen]
    useful=[v for v in seen if v[1]not in eos];useful_total+=len(useful);useful_after+=sum(i>0 and t>=anchor for i,token,t in useful)
    gaps=[(useful[i+1][2]-useful[i][2])/1e9 for i in range(len(useful)-1)];all_itl.extend(gaps)
    rate=(len(useful)-1)/((useful[-1][2]-useful[0][2])/1e9)if len(useful)>1 else None
    peruser.append({'replica':a,'local_row':row,'global_row':row+(0 if a=='A'else 16),'nonEOS_tokens':len(useful),'first_last_TPS':rate,'service_TTFT_s':(seen[0][2]-service[a])/1e9,'including_gate_setup_TTFT_s':(seen[0][2]-release)/1e9,'ITL_P95_s':percentile(gaps,.95),'ITL_P99_s':percentile(gaps,.99),'maximum_gap_s':max(gaps,default=0)})
  terminal=max(v[-1]for v in times.values());decode_s=(terminal-anchor)/1e9;E2E_s=(terminal-start)/1e9;cards=16*len(replicas);assert decode_s>0 and E2E_s>0
  overlap=max(0,min(v[1]for v in decode_intervals.values())-max(v[0]for v in decode_intervals.values()))/1e9
  result[phase]={'cards':cards,'replicas':list(replicas),'useful_postallprefill_decisions':useful_after,'common_decode_interval_s':decode_s,'decode_useful_TPS':useful_after/decode_s,'decode_useful_TPS_per_card':useful_after/decode_s/cards,'prefill_input_tokens':input_tokens,'prefill_common_interval_s':(anchor-start)/1e9,'prefill_input_TPS':input_tokens/((anchor-start)/1e9),'common_service_E2E_s':E2E_s,'service_E2E_useful_TPS':useful_total/E2E_s,'gate_setup_then_service_E2E_s':(terminal-release)/1e9,'actual_decode_overlap_s':overlap,'all_recorded_IDs_match_Model_reports':True,'request_firstlast_TPS_summary':summary([v['first_last_TPS']for v in peruser if v['first_last_TPS']is not None]),'service_TTFT_s_summary':summary([v['service_TTFT_s']for v in peruser]),'including_gate_setup_TTFT_s_summary':summary([v['including_gate_setup_TTFT_s']for v in peruser]),'ITL_allgap_P95_s':percentile(all_itl,.95),'ITL_allgap_P99_s':percentile(all_itl,.99),'maximum_gap_s':max(all_itl,default=0),'per_request':peruser,'scope':'Same-process collector successful packet arrivals; includes transport/observer; original localemitter metrics separate. NonEOS(n-1)/(last-first); not aggregate/B. Gate setup and service-start clocks explicitly separate.'}
 for phase in('jointM8','jointM16'):result[phase]['actual_joint_overlap_proved']=result[phase]['actual_decode_overlap_s']>0
 result['comparison']={'total_requests_fixed':32,'M16_over_M8_decode_ratio':result['jointM16']['decode_useful_TPS']/result['jointM8']['decode_useful_TPS'],'scope':'Matched actualjoint/commonclock;same32distinct;twoB16replicas,notB32x2'}
 return result
