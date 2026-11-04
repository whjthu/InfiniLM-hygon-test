"""Stdlib-only existing1417 nested-clock arithmetic. No Native or job imports."""
from pathlib import Path
import argparse,base64,collections,gzip,hashlib,json,math,statistics
PIN='a0459317d11f7668db714d9ae6d75ec67466130a13091b819a1cf88d7a551393'
PREFIX='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/Deep458-fixed-cohort-private/fixed-c4-20261104-a/'
def stats(v):
 s=sorted(v);n=len(s)
 def p(q):
  x=(n-1)*q;i=int(x);return s[i]+(s[min(i+1,n-1)]-s[i])*(x-i)
 return dict(n=n,sum_s=sum(s),mean_s=statistics.mean(s),median_s=statistics.median(s),min_s=s[0],P95_s=p(.95),max_s=s[-1])if n else None
def recompute(path):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN;archive=json.loads(raw);files={}
 for f in archive['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64']));assert len(b)==f['bytes']and hashlib.sha256(b).hexdigest()==f['SHA256'];files[f['path']]=b
 report=json.loads(files[PREFIX+'entry-report.json']);journal=[json.loads(line)for line in files[PREFIX+'calls-and-events-private.jsonl'].splitlines()]
 assert len(journal)==8142;groups={};index=None
 for e in journal:
  if e['event']=='cohort_begin':index=e['ordinal'];groups[index]=[]
  elif e['event']=='CAPI_return':groups[index].append(e)
 trials=[];formal=collections.defaultdict(list);user=[]
 for i in range(6):
  t=json.loads(files[PREFIX+'cohort-%d-private.json'%i]);calls=t['calls'];capi=groups[i];assert len(calls)==len(capi)
  start=t['all_arrival_monotonic'];end=t['completion_monotonic'];whole=end-start;assert math.isclose(whole,t['whole_cohort_seconds'])
  cs=[v['return_monotonic']-v['start_monotonic']for v in calls];ns=[v['CAPI_return_monotonic']-v['CAPI_start_monotonic']for v in capi]
  for c,n in zip(calls,capi):assert c['start_monotonic']<=n['CAPI_start_monotonic']<=n['CAPI_return_monotonic']<=c['return_monotonic']
  before=calls[0]['start_monotonic']-start;after=end-calls[-1]['return_monotonic'];gaps=[b['start_monotonic']-a['return_monotonic']for a,b in zip(calls,calls[1:])]
  assert min(gaps)>=0;outside=sum(gaps)+before+after;assert math.isclose(whole,sum(cs)+outside,abs_tol=1e-7)
  by={}
  for kind in['prefill','decode']:
   for rows in[1,2,3,4]:
    selected=[(c,n)for c,n in zip(calls,capi)if c['kind']==kind and len(c['rows'])==rows]
    if not selected:continue
    key='%s_rows%d'%(kind,rows);by[key]=dict(caller=stats([c['seconds']for c,n in selected]),ctypes_CAPI=stats([n['CAPI_seconds']for c,n in selected]),caller_minus_CAPI=stats([c['seconds']-n['CAPI_seconds']for c,n in selected]),past_range=[min(min(c['past'])for c,n in selected),max(max(c['past'])for c,n in selected)])
    if i>=2:formal[(t['mode'],key)]+=selected
  peruser=[]
  for state in t['states']:
   v=[e for e in state['token_events']if e['visible']];first,last=v[0]['monotonic'],v[-1]['monotonic'];itl=[b['monotonic']-a['monotonic']for a,b in zip(v,v[1:])]
   between=[(c,n)for c,n in zip(calls,capi)if c['start_monotonic']>=first and c['return_monotonic']<=last]
   native=sum(n['CAPI_seconds']for c,n in between);outer=last-first;assert native<=outer+1e-6
   row=dict(row=state['row'],visible_tokens=len(v),first_visible=first,last_visible=last,TTFT_s=first-start,decode_tok_s=(len(v)-1)/outer,first_ITL_s=itl[0],ITL=stats(itl),visible_interval_s=outer,ctypes_CAPI_enclosures_in_visible_interval_s=native,non_CAPI_visible_interval_s=outer-native,if_only_non_CAPI_interval_removed_tok_s=(len(v)-1)/native,other_row_prefill_CAPI_before_first_decode_s=sum(n['CAPI_seconds']for c,n in between if c['kind']=='prefill'and state['row']not in c['rows']))
   if i>=2 and t['mode']=='serial':
    assert all(c['rows']==[state['row']]and c['kind']=='decode'for c,n in between)
    row['required_enclosed_latency_reduction_fraction_for_15']=1-((len(v)-1)/15)/native
   peruser.append(row)
   if i>=2:user.append(dict(cohort=i,mode=t['mode'],**row))
  trials.append(dict(ordinal=i,mode=t['mode'],warm=i<2,whole_cohort_s=whole,caller_sum_s=sum(cs),ctypes_CAPI_sum_s=sum(ns),caller_minus_CAPI_s=sum(cs)-sum(ns),outside_caller_s=outside,begin_bracket_s=before,inter_call_gaps=stats(gaps),end_bracket_s=after,whole_minus_CAPI_s=whole-sum(ns),grade_enclosure_difference_s=t['whole_trial_seconds_including_decode_grade']-whole,full_trial_including_grade_s=t['whole_trial_seconds_including_decode_grade'],calls=len(calls),phase_counts=by,users=peruser))
 summary={}
 for (mode,key),selected in formal.items():summary[mode+'/'+key]=dict(caller=stats([c['seconds']for c,n in selected]),ctypes_CAPI=stats([n['CAPI_seconds']for c,n in selected]),caller_minus_CAPI=stats([c['seconds']-n['CAPI_seconds']for c,n in selected]))
 assert sum(v['calls']for v in trials)==1821
 totals={k:sum(t[k]for t in trials)for k in['whole_cohort_s','caller_sum_s','ctypes_CAPI_sum_s','caller_minus_CAPI_s','outside_caller_s','whole_minus_CAPI_s','grade_enclosure_difference_s']}
 return dict(archive_SHA256=PIN,rawfiles=len(files),raw_decoded_bytes=sum(map(len,files.values())),JSONL_events=len(journal),CAPI_calls=1821,totals=totals,cohorts=trials,formal_phase_summary=summary,formal_user_clocks=user,journal_measured_write_enclosure_s=report['journal']['append_flush_seconds'],journal_write_calls=report['journal']['writes'],exclusive_GPU_kernel_s='UNKNOWN',exclusive_CCL_s='UNKNOWN',GPU_busy_fraction='UNKNOWN',expert_route_repetition='UNKNOWN_NOT_RECORDED',counter_scope='HC/layout/FP32 dispatch verification is not phase timing or selected-expert IDs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=recompute(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');print(json.dumps(q['totals']))
