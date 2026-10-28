from pathlib import Path
import json,hashlib,statistics,sys
rows=[json.loads(Path(p).read_bytes())for p in sys.argv[1:5]]
assert len(rows)==4 and {r['node']for r in rows}==set(range(4))and all(r['completed']and r['normal_CUDA_cleanup']for r in rows)
owners=[o for r in rows for o in r['native']['reports']];assert len(owners)==32 and {o['global_tp_rank']for o in owners}==set(range(32))
result={'component_only':True,'TP':32,'PP':1,'local_I':96,'cases':[],'evidence_SHA256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in sys.argv[1:5]},'whole_Model_TPS':'UNKNOWN','full_Model_fit':'UNKNOWN_STOP'}
for workload in range(4):
 blocks=[]
 for base in [4,8]:
  times=[];errors=[]
  for arm in range(base,base+4):
   values=[next(a for a in o['arms']if a['workload']==workload and a['arm']==arm)for o in owners]
   assert all(a['actual_I96_branch_hits']==int(a['candidate_I96'])and a['input_readonly_tail_canary']and a['nonfinite']==0 for a in values)
   times.append(max(a['whole_MoE_including_outer_drain_ms']for a in values));errors.append(max(a['repeat_relative_RMS_diagnostic']for a in values))
  off=(times[0]+times[3])/2;on=(times[1]+times[2])/2;blocks.append({'max_owner_ms_ABBA':times,'MoE_speedup':off/on,'OFF_endpoint_drift':abs(times[3]/times[0]-1),'max_NRMSE_each_arm':errors})
 result['cases'].append({'workload':workload,'scope':'historicalT8/T16/T64 orcontrolledGaussian64, notcurrent449traffic','blocks':blocks,'eligible_for_later_design':all(b['MoE_speedup']>=1.10 and b['OFF_endpoint_drift']<=.03 for b in blocks)})
result['no_automatic_Model_GO']=True;result['limitations']='Matched TP32 component only; no comparison to TP16/PP2 as2x; no Model/quality/userTPS/capacity claim. Max owner duration not synchronized global latency.'
p=Path(sys.argv[5]);assert not p.exists();p.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
