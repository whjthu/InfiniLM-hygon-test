from pathlib import Path
import json,statistics,sys,hashlib
rows=[json.loads(Path(p).read_bytes())for p in sys.argv[1:3]]
assert len(rows)==2 and {x['node']for x in rows}=={0,1} and all(x['completed']and x['normal_CUDA_cleanup']for x in rows)
result={'component_only':True,'GPU_math':'FP32 reduction order differs; no GPU F64','current449_routes':False,'experiments':[],'evidence_SHA256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in sys.argv[1:3]}}
for layer in [1,47]:
 owners=[o for n in rows for e in n['experiments']if e['layer']==layer for o in e['native']['reports']]
 assert len(owners)==16 and all(o['normal_Runtime_close']and o['module_closed']for o in owners)
 for w in [0,1]:
  blocks=[]
  for k in range(3):
   arms=[4+3*k+j for j in range(3)];critical=[];phase=[];hist=[]
   for arm in arms:
    q=[next(a for a in o['arms']if a['arm']==arm and a['workload_index']==w)for o in owners]
    assert all(a['input_readonly_and_tail_canary']and a['forward_finite']and a['NRMSE_vs_first_warm']<=.005 and len(a['uninstrumented_full_MoE_complete_wall_ms'])==3 for a in q)
    critical.append(statistics.median(max(a['uninstrumented_full_MoE_complete_wall_ms'][r]for a in q)for r in range(3)))
    ps=[next(p for p in o['profile']if p['arm']==arm and p['workload_index']==w)for o in owners]
    assert all(sum(p['expert_hist'])==1024 and len(p['event_segment_ms'])==15 for p in ps)
    assert all(p['candidate_wavepartial']==(arm==arms[1])for p in ps)
    phase.append(statistics.median(p['event_segment_ms'][6]for p in ps))
    hist.append([{'global_rank':o['global_tp_rank'],'U':p['expert_unique'],'J':p['TT2_jobs'],'partial':p['TT2_partial_jobs']}for o,p in zip(owners,ps)])
   base=(critical[0]+critical[2])/2;pb=(phase[0]+phase[2])/2
   blocks.append({'arms':arms,'complete_MoE_critical_max_owner_ms':critical,'complete_gain_fraction':base/critical[1]-1,'OFF_bracket_drift_fraction':abs(critical[2]/critical[0]-1),'W13_median_owner_ms':phase,'W13_gain_fraction':pb/phase[1]-1,'route_counts':hist})
  result['experiments'].append({'layer':layer,'workload_index':w,'blocks':blocks,'eligible_for_later_model_proposal':all(b['complete_gain_fraction']>=.05 and b['W13_gain_fraction']>=.05 and b['OFF_bracket_drift_fraction']<=.03 for b in blocks)})
result['all_four_controlled_cases_promotable']=all(x['eligible_for_later_model_proposal']for x in result['experiments'])
result['limitations']='No automatic Model GO. Controlled historical16-repeat/Gaussian T64, not current449 route distribution; max owner call durations are not whole model throughput; actual counts are controlled only. New HIP codegen and actual GPU validation required.'
p=Path(sys.argv[3]);assert not p.exists();p.write_text(json.dumps(result,indent=2))
