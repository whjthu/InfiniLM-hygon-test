"""Audit existing metadata only; never imports GPU libraries or launches jobs."""
from pathlib import Path
import json, hashlib, base64, statistics, math

G=Path(__file__).resolve().parents[3]
L=G.parent
O=G/'results/bw1000/2026-11-02'
sha=lambda b:hashlib.sha256(b).hexdigest()
median=statistics.median
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
 transfer=json.loads((L/'Deep359-raw-sealed-transfer.json').read_bytes())
 names=['Deep359-GPU684-report-sealed.json','Deep359-keeper727-watch-sealed.json']
 refs=[]
 for row,name in zip(transfer,names):
  b=base64.b64decode(row['base64']);assert sha(b)==row['SHA256']
  p=O/name;p.write_bytes(b);refs.append(dict(local=str(p.relative_to(G)),remote=row['path'],SHA256=sha(b),bytes=len(b)))
 r=json.loads((O/names[0]).read_bytes());w=json.loads((O/names[1]).read_bytes())
 assert r['completed']and r['normal']and w['completed']and w['passed']
 assert r['actual_step']==w['actual_GPU']=='910371.684'and w['CPU_keeper']=='910407.727'
 assert w['srun_exit_code']==0 and len(r['hardware'])==8
 assert len({h['PCI']for h in r['hardware']})==8 and all('gfx936'in h['arch']for h in r['hardware'])
 owners=r['owners'];assert sorted(o['rank']for o in owners)==list(range(8))
 assert all(len(o['calls'])==52 and o['normal_close']and not o.get('error')for o in owners)
 cpus=[o['affinity_after_submit']for o in owners]
 assert all(len(v)==4 for v in cpus)and len(set(sum(cpus,[])))==32
 assert all(o['expected_affinity']==o['affinity_after_submit']for o in owners)
 assert all(len(o['initial_affinity'])==32 for o in owners)
 allcalls=[c for o in owners for c in o['calls']]
 for o in owners:
  assert len(o['nextstep_commit_checks'])==6 and all(v['passed']for v in o['nextstep_commit_checks'])
  for t,c in enumerate(o['calls']):
   assert c['trial']==t and all(c[k]==r['plan'][t][k]for k in ['past','phase','mode'])
   assert c['candidate_hits']==int(c['mode']==1)
   n=c['numeric'];assert n['finite']and n['TopK512_unique_valid']and n['CPU_FP8_score_NRMSE']<2e-4
 results={}
 for past in [8496,8575]:
  rows=r['matched_measurements'][str(past)]
  assert len(rows)==20 and [v['mode']for v in rows]==[0,1,1,0]*5
  for row in rows:
   cs=[o['calls'][row['trial']]for o in owners]
   assert row['max8_CAPI_wall_s']==max(c['CAPI_wall_s']for c in cs)
   assert row['max8_Cpp_prepare_commit_s']==max(c['Cpp_prepare_s']+c['Cpp_commit_s']for c in cs)
   assert row['same_phase_checked_enclosure_s']==max(c['measurement_end_s']for c in cs)-min(c['measurement_begin_s']for c in cs)
  bymetric={}
  for key in ['max8_CAPI_wall_s','max8_Cpp_prepare_commit_s','same_phase_checked_enclosure_s']:
   rounds=[]
   for i in range(0,20,4):
    a,b,c,d=rows[i:i+4];off=(a[key]+d[key])/2;on=(b[key]+c[key])/2
    rounds.append(dict(trials=[a['trial'],b['trial'],c['trial'],d['trial']],OFF_s=off,ON_s=on,ratio=on/off,OFF_endpoint_ratio=d[key]/a[key]))
   bymetric[key]=dict(rounds=rounds,round_ratio_median=median(v['ratio']for v in rounds),round_ratio_min=min(v['ratio']for v in rounds),round_ratio_max=max(v['ratio']for v in rounds),OFF_round_median_s=median(v['OFF_s']for v in rounds),ON_round_median_s=median(v['ON_s']for v in rounds),OFF_endpoint_drift_min=min(v['OFF_endpoint_ratio']for v in rounds),OFF_endpoint_drift_max=max(v['OFF_endpoint_ratio']for v in rounds))
  samecard=[]
  for o in owners:
   cs=[v for v in o['calls']if v['past']==past and v['phase']=='measurement'];ratios=[]
   for i in range(0,20,4):
    a,b,c,d=cs[i:i+4];ratios.append((b['CAPI_wall_s']+c['CAPI_wall_s'])/(a['CAPI_wall_s']+d['CAPI_wall_s']))
   samecard.append(dict(rank=o['rank'],round_ratios=ratios,median=median(ratios)))
  results[str(past)]=dict(metrics=bymetric,samecard_CAPI=samecard,original_cold_QA_max8_prefix_s=max(c['prefix_reseed_wall_s']for c in allcalls if c['past']==past and c['phase']=='baseline_QA'),all_trials_max8_prefix_local_wall_s=[max(o['calls'][i]['prefix_reseed_wall_s']for o in owners)for i,p in enumerate(r['plan'])if p['past']==past])
 post=json.loads((L/'Deep359-independent-post684-exact.json').read_bytes())
 assert post['CPU_only']and not post['GPU_initialized']
 assert post['node16']['returncode']==post['keepernode']['returncode']==0
 p=post['node16']['rows'][0];q=post['keepernode']['rows'][0]
 assert p['KFD_available']and not p['kfd_pids']and not p['matching_component_processes']
 assert not q['matching_controllers']and not q['original_srun_PID_exists']
 write(O/'Deep359-GPU684-independent-post.json',dict(exact_post=post,first_post_probe_note='First keeper process substring scan matched its own python -c literal; exact argv-suffix follow-up found no controller. First metadata remains private L/Deep359-independent-post684.json; it was not a leftover worker.',original_post=w['post'],Slurm_GPU=w['terminal'],Slurm_keeper='910407.727 COMPLETED 0:0 00:01:02',independent_CPU_srun_returncodes=[0,0]))
 nums=[c['numeric']for c in allcalls]
 bymode={}
 for past in [8496,8575]:
  for mode in [0,1,2]:
   ns=[c['numeric']for c in allcalls if c['past']==past and c['mode']==mode]
   bymode[str((past,mode))]=dict(records=len(ns),original_score_maxabs_max=max(n.get('original_score_maxabs',0)for n in ns),original_TopK_overlap_min=min(n.get('TopK_overlap',512)for n in ns))
 out=dict(completed=True,component_GPU_execution_passed=True,Model_loaded=False,Model_gain='UNKNOWN',decision='COMPONENT_TIMING_POSITIVE_WITH_CROSS_TRIAL_SCORE_DRIFT_CAVEAT; no Model promotion or further GPU authorized',GPU='910371.684',CPU_keeper='910407.727',GPU_elapsed_s=59,CPU_elapsed_s=62,actual_GPU_card_seconds=472,source_publication_commit=w['publication_commit'],GPU_manifest_SHA256=w['manifest_SHA256'],raw=refs,hardware=r['hardware'],owner_affinity=[dict(rank=o['rank'],initial=o['initial_affinity'],actual=o['affinity_after_submit'],TID=o['TID'])for o in owners],records=416,trials=52,measurement_trials=40,nextstep_commit_CAPI_success_checks=48,nextstep_numerical_comparison='NOT_PERFORMED; success is real subsequent transaction and canary, not independently compared subsequent output',all8_normal_Runtime_and_CCL_destroy=True,all_candidate_hits_exact=True,all_input_canary_checks_passed=True,numeric=dict(independent_CPU_FP8_score_NRMSE_max=max(n['CPU_FP8_score_NRMSE']for n in nums),independent_CPU_maxabs_max=max(n['CPU_maxabs']for n in nums),TopK512_unique_valid_all=True,cross_baseline_relative_RMS_score_scale_and_cutoff_tie_margin='UNKNOWN; raw scores and cross-trial caches not persisted',baseline_snapshot_score_and_TopK_diagnostics=bymode,committed_fields='[0,1,2,3,7] compared byte-exact against per-past original baseline by unchanged driver',limitation='CPU reference is recomputed from each trial own raw cache field6. Raw field6 is not byte-compared across original/private or repeated resets and is not saved. Repeated original mode0 itself has score drift/occasional overlap511, so recorded cross-trial differences cannot be attributed solely to candidate arithmetic. Exact same-cache original-vs-candidate numerical equivalence remains UNKNOWN; no additional GPU or reset replay.'),timing=results,timing_scope=r['timing_scope'],checked_enclosure_clock='Single host shared perf_counter, max(owner end)-min(owner begin) per trial; includes scheduling skew/H2D/clone/canary. It is not GPUevent and no cross-host clock subtraction.',prefix_scope='Reset/reseed1024 is outside target single-token prepare+commit. Per-owner prefix local wall and cold original baseline reported separately; max of local durations is descriptive, not a synchronized full-prefix makespan.',checkpoint=r['checkpoint'],input_kind=r['input_kind'],CPU_receipt_SHA256=r['CPU_receipt_SHA256'],DSO_SHA256=r['DSO_SHA256'],driver_seconds=r['driver_seconds'],Torch_peak_scope='Only Torch allocator, excludes InfiniRT; not whole-component memory peak',Torch_peaks=[dict(rank=o['rank'],allocated=o['torch_peak_allocated'],reserved=o['torch_peak_reserved'])for o in owners],cleanup='Original watch passed and post685 clean; independent CPU688 node16 KFD/PIDs empty and CPU729 keeper controller absent/original srun PID gone; both independent srun clients returned0. Original GPU/keeper terminal0, no retry/no additional request.')
 write(O/'Deep359-GPU684-component-audit.json',out)
 task=G/'coordination/tasks/Deep-C4-paged-score-r359.json';t=json.loads(task.read_bytes());t.update(state='SEALED_COMPONENT_PASS_TIMING_POSITIVE_NUMERICAL_SAMECACHE_EQUIVALENCE_UNKNOWN_NO_MODEL',GPU_GO=False,new_GPU_submitted=True,actual_GPU='910371.684',actual_CPU_keeper='910407.727',GPU_terminal='COMPLETED0/59s',CPU_keeper_terminal='COMPLETED0/62s',whole_Runtime_or_GPU='52trials/416ownerrecords passed; no Model',final_result=str((O/'Deep359-GPU684-component-audit.json').relative_to(G)),final_raw_report_SHA256=refs[0]['SHA256'],normal8=True,no_further_GPU=True)
 write(task,t)
 print(json.dumps(dict(completed=True,results={p:{k:dict(ratio=v['round_ratio_median'],OFF_ms=v['OFF_round_median_s']*1000,ON_ms=v['ON_round_median_s']*1000,range=[v['round_ratio_min'],v['round_ratio_max']])for k,v in x['metrics'].items()}for p,x in results.items()},audit=str(O/'Deep359-GPU684-component-audit.json'))))

if __name__=='__main__':main()
