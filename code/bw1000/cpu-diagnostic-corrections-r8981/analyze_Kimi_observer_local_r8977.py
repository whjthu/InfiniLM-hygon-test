"""CPU-only comparison of retained 639 same-case adjacent calls; no trace reload."""
from pathlib import Path
import hashlib,json,re,statistics
r=Path.home()/'hygon-inference-refactor-20261014'
run='currentprofile8670-2780c25f37791711'
out={'CPU_only':True,'GPU':False,'build':False,'original_handle':'910375.639','run':run,'nodes':[]}
for node in range(4):
 p=r/'results'/f'kimi-current-profile-r8670-{run}-node{node}.json'
 raw=p.read_bytes();q=json.loads(raw);w=q['profiling_windows'];assert len(w)==1;w=w[0]
 assert q['completed'] and q['all8_owned_workers_shutdown_completed']
 assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
 case=w['label'];samples=[]
 for v in q['local_forward_samples']:
  m=re.fullmatch(re.escape(case)+r'-decode(\d+)-cohort(\d+)',v['label'])
  if m:
   assert v['N']==v['B']==16
   a=dict(v);a['step']=int(m[1]);a['cohort']=int(m[2]);a['fully_marked']=w['start_realtime_ns']<=a['CAPI_start_realtime_ns']<a['CAPI_end_realtime_ns']<=w['stop_realtime_ns'];samples.append(a)
 inside=[a for a in samples if a['fully_marked']];assert len(inside)==8
 steps=sorted(set(a['step']for a in inside));assert len(steps)==2
 lo,hi=min(steps),max(steps);chosen=[a for a in samples if lo-1<=a['step']<=hi+1]
 assert len(chosen)==16 and all(len([a for a in chosen if a['step']==s])==4 for s in range(lo-1,hi+2))
 ratios=[];cohort=[]
 for c in range(4):
  z={a['step']:a for a in chosen if a['cohort']==c}
  base=(z[lo-1]['CAPI_seconds']+z[hi+1]['CAPI_seconds'])/2
  active=(z[lo]['CAPI_seconds']+z[hi]['CAPI_seconds'])/2
  ratios.append(active/base);cohort.append({'cohort':c,'before_seconds':z[lo-1]['CAPI_seconds'],'marked_seconds':[z[s]['CAPI_seconds']for s in steps],'after_seconds':z[hi+1]['CAPI_seconds'],'marked_over_adjacent_mean':active/base})
 out['nodes'].append({'node':node,'source_report_path':str(p),'source_report_SHA256':hashlib.sha256(raw).hexdigest(),'window':w,'actual_marked_steps':steps,'adjacent_unmarked_steps':[lo-1,hi+1],'matched_cohorts':cohort,'ratio_median':statistics.median(ratios),'step_summaries':[{'step':s,'fully_marked':all(a['fully_marked']for a in chosen if a['step']==s),'median_CAPI_seconds':statistics.median(a['CAPI_seconds']for a in chosen if a['step']==s),'sum_CAPI_seconds':sum(a['CAPI_seconds']for a in chosen if a['step']==s)}for s in range(lo-1,hi+2)],'all_same_case_step_medians':[{'step':s,'median_CAPI_seconds':statistics.median(a['CAPI_seconds']for a in samples if a['step']==s)}for s in sorted(set(a['step']for a in samples))],'selected_sample_extra_fields':sorted(set().union(*(set(a)for a in chosen))-{'label','N','B','CAPI_seconds','CAPI_start_realtime_ns','CAPI_end_realtime_ns','pipeline','step','cohort','fully_marked'}),'report_worker_profile_keys':[k for k in q if any(t in k.lower()for t in ('worker','sample','timing','profile'))]})
out['limitations']=['All same-case calls retain hipprof interposer; active-window effect only, not interposer-vs-production control.','CAPI_seconds includes owner runtime/PP/feedback dependency effects; adjacent steps are observational, not randomized.','local_forward_samples has one owner CAPI per stage, not eight independent rank timing samples.','Past is not stored per call; source increments each row exactly one after each decode and actual start_pasts binds marked start.']
print(json.dumps(out,ensure_ascii=False))
