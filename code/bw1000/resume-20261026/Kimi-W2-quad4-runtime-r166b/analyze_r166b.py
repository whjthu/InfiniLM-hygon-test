from pathlib import Path
import json,statistics,hashlib,sys
paths=[Path(p)for p in sys.argv[1:3]];rows=[json.loads(p.read_bytes())for p in paths];assert len(rows)==2 and{q['node']for q in rows}=={0,1}and all(q['completed']and q['normal_CUDA_cleanup']for q in rows)
owners=[o for q in rows for e in q['experiments']for o in e['native']['reports']];assert len(owners)==16 and all(o['normal_Runtime_close']and o['module_closed']for o in owners)
out={'scope':'L1onlyTP16T8historygeometry/T64history/Gaussian;notC1flow/notModel','schedule':'4warmOFFOFFONON+2ABBA,3uninstrumentedcallsand1separatephaseperarm','component_only':True,'evidence_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},'experiments':[]}
for w,T in [(0,8),(1,64),(2,64)]:
 blocks=[]
 for start in [4,8]:
  full=[];phase=[];error=[]
  for arm in range(start,start+4):
   qs=[next(a for a in o['arms']if a['workload_index']==w and a['arm']==arm)for o in owners];assert all(a['T']==T and a['forward_finite']and a['input_readonly_and_tail_canary']and a['candidate_quad4']==((arm-4)%4 in [1,2])and len(a['uninstrumented_full_MoE_complete_wall_ms'])==3 for a in qs)
   full.append(statistics.median(max(a['uninstrumented_full_MoE_complete_wall_ms'][r]for a in qs)for r in range(3)));ps=[next(p for p in o['W2_event_samples']if p['workload_index']==w and p['arm']==arm)for o in owners];phase.append(statistics.median(p['W2_stream_ms']for p in ps));error.append(max(a['NRMSE_vs_first_warm']for a in qs))
  base=(full[0]+full[3])/2;candidate=(full[1]+full[2])/2;pb=(phase[0]+phase[3])/2;pc=(phase[1]+phase[2])/2
  blocks.append({'arms':list(range(start,start+4)),'fullMoE_ABBA_criticalmax_ms':full,'fullMoE_rate_gain_percent':100*(base/candidate-1),'OFF_drift_percent':100*abs(full[3]/full[0]-1),'W2_ABBA_medianowner_stream_ms':phase,'W2_rate_gain_percent':100*(pb/pc-1),'NRMSE_max':max(error),'passed_predeclared_component_gate':base/candidate>=1.05 and pb/pc>=1.05 and abs(full[3]/full[0]-1)<=.03})
 out['experiments'].append({'workload_index':w,'T':T,'blocks':blocks,'both_blocks_passed':all(b['passed_predeclared_component_gate']for b in blocks)})
out['all3controlled_cases_passed']=all(e['both_blocks_passed']for e in out['experiments']);out['Model_promotion']='NONE;newquality/serving qualification required evenifcomponentpositive';out['normal_owners']=16
p=Path(sys.argv[3]);assert not p.exists();p.write_text(json.dumps(out,indent=2)+chr(10))
