"""Read immutable full-Delta reports; recompute same-trial max16 wall. No execution/GPU."""
from pathlib import Path
import argparse,json,statistics,hashlib

def analyze(watch,nodes):
 assert watch['completed']and not watch['passed']
 assert all(p['passed']and not p['KFD_pids']and not p['expected_PIDs_still_exist']and not p['listeners']for p in watch['failure_post'])
 workers=[w for n in nodes for w in n['workers']];assert len(workers)==16 and sorted(w['rank']for w in workers)==list(range(16))
 assert all(n['Math']==0 and not n['capture']and len(n['workers'])==8 for n in nodes)
 common=min(len(w['samples'])for w in workers);prefix_rounds=common//40;prefix_calls=prefix_rounds*40
 assert 0<prefix_calls<320 and common==273
 assert all(w['normal_close']and len(w['samples'])==common and w['attempted']==w['CAPI_returned']==common+1 for w in workers)
 assert all(w['samples'][-1]['counters_after']['fallback']==0 and w['samples'][-1]['counters_after']['chosen']==140 and w['samples'][-1]['counters_after']['consumer_counts']==[140]*4 and w['samples'][-1]['counters_after']['epochs']==274 and w['pack_stats']['packed_bytes']==45961216 and w['pack_stats']['packs']==1 for w in workers)
 # All 273 completed calls checked; only the entire chronological four-arm-round prefix is used for timing.
 all_numeric_passed=all(t['passed']and t['counter_passed']and t['call_completed']and t['numerical']['passed']for w in workers for t in w['samples'])
 assert all_numeric_passed
 pts=[];max_NRMS={};maxabs={}
 for index in range(prefix_calls):
  r=index//40;block=(index//10)%4;trial=index%10;on=([False,True,True,False]if r%2==0 else[True,False,False,True])[block];arm='ON'if on else'OFF'
  rows=[w['samples'][index]for w in workers]
  assert all(t['index']==index and t['round']==r and t['trial']==trial and t['arm']==arm and t['warmup']==(trial<2)and t['call_completed']and t['passed']and t['counter_passed']and t['numerical']['passed']and t['numerical']['source_state_rows_unchanged']for t in rows)
  for t in rows:
   for name in ['output','conv_written_row2','ssm_written_row2']:
    v=t['numerical'][name];assert v['finite']and v['passed'];max_NRMS[name]=max(max_NRMS.get(name,0),v['NRMS']);maxabs[name]=max(maxabs.get(name,0),v['max_abs'])
  pts.append(dict(index=index,round=r,block=block,trial=trial,fixture=trial%2,arm=arm,warmup=trial<2,max16_API_wall_ms=max(t['numerical']['API_wall_ms']for t in rows)))
 formal=[p for p in pts if not p['warmup']];assert len(formal)==prefix_rounds*32
 med=lambda v:statistics.median(v)
 off=med([p['max16_API_wall_ms']for p in formal if p['arm']=='OFF']);on=med([p['max16_API_wall_ms']for p in formal if p['arm']=='ON']);gain=off/on-1
 endpoints=[med([p['max16_API_wall_ms']for p in formal if p['round']==r and p['arm']=='OFF'])for r in [0,prefix_rounds-1]];drift=endpoints[1]/endpoints[0]-1
 adjacent=[[],[]];rounds=[]
 for r in range(prefix_rounds):
  block_ms=[med([p['max16_API_wall_ms']for p in formal if p['round']==r and p['block']==b])for b in range(4)]
  armorder=['OFF','ON','ON','OFF']if r%2==0 else['ON','OFF','OFF','ON']
  rates=[]
  for pair,(a,b) in enumerate([(0,1),(2,3)]):
   offv=block_ms[a]if armorder[a]=='OFF'else block_ms[b];onv=block_ms[b]if armorder[a]=='OFF'else block_ms[a];rate=offv/onv-1;adjacent[pair].append(rate);rates.append(rate)
  rounds.append(dict(round=r,arm_order=armorder,formal_calls_per_block=8,block_max16_median_ms=block_ms,adjacent_rate_gain=rates))
 pairrates=[med(x)for x in adjacent]
 numerical=dict(all_observed_4368_C1_checks_passed=all_numeric_passed,max_NRMS_all_observed={name:max(t['numerical'][name]['NRMS']for w in workers for t in w['samples'])for name in max_NRMS},max_NRMS_complete_prefix=max_NRMS,max_abs_complete_prefix=maxabs)
 gate=bool(gain>abs(drift)and all(x>0 for x in pairrates))
 setup=[w['pack_stats']['setup_ms']for w in workers];destroy=[w['destroy_wall_ms']for w in workers]
 api_sum_max16_s=sum(max(w['samples'][i]['numerical']['API_wall_ms']for w in workers)for i in range(common))/1000
 observed_warm=sum(t['warmup']for t in workers[0]['samples'])
 return dict(completed=True,full_plan_passed=False,full_plan_failure='300s measurement deadline at273/320 C1 perowner; no qualification or replay',partial_numeric_checks_passed=True,partial_prefix_performance_gate_PASS=gate,decision='NO_FULL_QUALIFICATION_PARTIAL_GAIN_BELOW_OFF_DRIFT',scope='complete Delta C1 TP16 component prefix only, historical teacher-normalized two rows; no Model/HTTP',actual_GPU=watch['actual_step'],source_manifest_SHA256=watch['GPU_source_manifest_SHA256'],actual_calls=dict(owners=16,planned_C1_per_owner=320,observed_C1_per_owner=common,observed_C1_total=common*16,observed_C1_warm_total=observed_warm*16,observed_C1_formal_total=(common-observed_warm)*16,T512_initialization_total=16,all_CAPI_returned_total=sum(w['CAPI_returned']for w in workers),complete_prefix_rounds=prefix_rounds,prefix_calls_per_owner=prefix_calls,prefix_formal_max16_points=len(formal),prefix_formal_C1_total=len(formal)*16,partial_round_omitted_from_performance=common-prefix_calls),OFF_max16_median_ms=off,ON_max16_median_ms=on,rate_gain=gain,latency_reduction=1-on/off,OFF_endpoint_medians_ms=endpoints,OFF_endpoint_drift=drift,adjacent_pair_median_rate_gain=pairrates,rounds=rounds,numerical=numerical,per_fixture={str(fi):{arm:med([p['max16_API_wall_ms']for p in formal if p['fixture']==fi and p['arm']==arm])for arm in ['OFF','ON']}for fi in [0,1]},setup=dict(pack_bytes_per_rank=45961216,pack_bytes_16rank=45961216*16,pack_setup_ms_min=min(setup),pack_setup_ms_max=max(setup),finish_wall_ms_max=max(w['finish_setup_wall_ms']for w in workers),destroy_wall_ms_max=max(destroy),extra_persistent_memory_also_present_in_OFF=True,Model_memory_fit='UNKNOWN'),cost=dict(GPUstep_elapsed_s=364,original_measurement_limit_s=300,max16_API_wall_sum_all273_s=api_sum_max16_s,API_sum_includes_cold_warm=True,perrank_API_sum_range_s=[min(sum(t['numerical']['API_wall_ms']for t in w['samples'])/1000 for w in workers),max(sum(t['numerical']['API_wall_ms']for t in w['samples'])/1000 for w in workers)],API_fraction_of_measurement_limit=api_sum_max16_s/300,API_excluded_enclosure='state restore/readback, counters, numeric/readonly checks, barriers, JSON deepcopy/full-report rewrite; not individually timed, exclusive save cost UNKNOWN',source_report_full_rewrites_per_node=2202,final_node_report_bytes='input_refs bytes',no_overlapping_stage_subtraction=True),state_restore=dict(excluded_from_API_wall=True,median_ms=med([v for w in workers for v in w['restore_wall_ms']]),max_ms=max(v for w in workers for v in w['restore_wall_ms'])),normal_Runtime_close_16=True,Slurm_FAILED1=True,srun_exit=watch['last_observation']['srun_exit'],post=watch['failure_post'],Model_gain=False,Model_GO=False,new_GPU_GO=False,points=pts)

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',required=True);p.add_argument('--nodes',nargs=2,required=True);p.add_argument('--output',required=True);a=p.parse_args();paths=[Path(a.watch),*[Path(v)for v in a.nodes]];data=[json.loads(p.read_bytes())for p in paths];q=analyze(data[0],data[1:]);q['input_refs']=[dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())for p in paths];Path(a.output).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q[k]for k in ['decision','OFF_max16_median_ms','ON_max16_median_ms','rate_gain','OFF_endpoint_drift','partial_prefix_performance_gate_PASS']}))
if __name__=='__main__':main()
