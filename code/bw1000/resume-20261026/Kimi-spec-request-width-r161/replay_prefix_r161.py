"""Read only root's whitelist trace; never predicts tok/s from token counts."""
from pathlib import Path
import sys,json,re,hashlib,collections
from request_width_r161 import request_width

p=Path(sys.argv[1]);d=json.loads(p.read_bytes());cases={}
for r in d['rows']:
 loc=r['location'];v=r['data']
 m=re.fullmatch(r'batch_cases/(\d+)/request_metrics/(\d+)/cycles/\d+',loc)
 if m:
  case,user=map(int,m.groups());c=cases.setdefault(case,{'users':{},'rounds':{}});c['users'].setdefault(user+1,{})[v['step']]=v
  continue
 m=re.fullmatch(r'batch_cases/(\d+)/cycles/\d+',loc)
 if m:
  case=int(m.group(1));cases.setdefault(case,{'users':{},'rounds':{}})['rounds'][v['step']]=v

tot=collections.Counter();records=[]
for ci,c in sorted(cases.items()):
 histories={i:[]for i in c['users']};count=collections.Counter()
 for step,rd in sorted(c['rounds'].items()):
  ids=rd['logical'];widths=rd['widths'];assert len(ids)==len(widths)and len(set(ids))==len(ids)
  chosen={i:min(w,request_width(histories[i],len(ids),8,enabled=True))for i,w in zip(ids,widths)}
  assert sum(chosen.values())<=64
  for i,w in zip(ids,widths):
   actual=c['users'][i][step];a=actual['accepted_proposals'];assert 0<=a<=w-1
   effective=chosen[i];accepted=min(a,effective-1)
   useful=actual['useful_emitted'];projected=min(useful,accepted+1)
   count.update(baseline_T=w,projected_T=effective,baseline_useful=useful,projected_same_cycle_useful=projected,request_cycles=1,shortened=int(effective<w),high_accept_shortened=int(effective<w and a>=4))
   # Only AFTER choosing width, publish this cycle's observable feedback.
   # Narrowed acceptance is censored, not an oracle for its omitted suffix.
   histories[i].append({'proposed':effective-1,'accepted_proposals':accepted,'terminal':actual['terminal']})
 tot.update(count);records.append({'case_index':ci,**dict(count),'T_reduction_fraction':1-count['projected_T']/count['baseline_T']})

reduction=1-tot['projected_T']/tot['baseline_T'];useful_loss=1-tot['projected_same_cycle_useful']/tot['baseline_useful']
result={'CPU_only':True,'model':False,'GPU':False,'source_whitelist_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'actual_report_SHA256':d['SHA256'],'actual_report_path':d['report_path'],'cases':records,'totals':dict(tot),'T_reduction_fraction':reduction,'same_cycle_useful_prefix_loss_fraction':useful_loss,'high_accept_shrink_fraction_of_shortened':tot['high_accept_shortened']/tot['shortened']if tot['shortened']else 0,'T_per_samecycle_useful_baseline':tot['baseline_T']/tot['baseline_useful'],'T_per_samecycle_useful_projected':tot['projected_T']/tot['projected_same_cycle_useful'],'wall_speedup':'UNKNOWN','quality':'UNKNOWN; no new generated outputs','causal_rollout':False,'past_only_decisions':True,'current_cycle_future_outcome_used_for_decision':False,'threshold_predeclared_min_T_reduction':.10,'decision':'STOP_NO_DEDICATED_GPU_LOAD'if reduction<.10 or useful_loss>=reduction else'CONDITIONAL_SAME_LOAD_ABBA_ONLY','limitations':['Same observed request arrivals/termination and teacher-forced tokens are held fixed; after a narrower bonus diverges, later model states are not a real rollout.','Only counterfactual prefix-work accounting. Native wall cost is nonlinear in T, MoE routes, PP and batch, so no tok/s prediction.','Historical narrow cases C1W2/W4 were slower, so C1..C3 stayW8. Existing C16W4/C32W2 remain unchanged.','The12cases repeat the same32 questions/profile arms, not96independent questions.','Trained draft block8/K<=7 and currentNative per-rowwidth<=8; no length16 claim.']}
out=Path(sys.argv[2]);assert not out.exists();out.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:result[k]for k in ['totals','T_reduction_fraction','same_cycle_useful_prefix_loss_fraction','high_accept_shrink_fraction_of_shortened','T_per_samecycle_useful_baseline','T_per_samecycle_useful_projected','decision']}))
