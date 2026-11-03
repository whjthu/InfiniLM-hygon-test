"""Stdlib-only original report/40 sidecar/journal reader; no device operations."""
from pathlib import Path,PurePosixPath
import json,hashlib,argparse,statistics,math
LABELS=['QueryProjection_through_RoPE_sync','Features_through_Query_status','CacheGather_through_Gather_status','Scores_through_Scores_status','TPsum_TopK_through_prepare_final_sync']
def analyze(path):
 b=Path(path).read_bytes();a=json.loads(b);files=a['raw_files']
 for n,v in files.items():assert hashlib.sha256(v['text'].encode()).hexdigest()==v['SHA256']and len(v['text'].encode())==v['bytes'],n
 watch=json.loads(next(v['text']for n,v in files.items()if n.endswith('-watch.json')));q=json.loads(files[watch['report_path']]['text']);assert files[watch['report_path']]['SHA256']==watch['report_SHA256']
 assert watch['passed']and watch['srun_exit_code']==0 and q['passed']and q['completed']and q['normal']and q['Model_loaded']is False
 assert [x['timing']for x in q['plan']]==[0,1,0,1,0]and len(q['owners'])==8 and len({x['PCI']for x in q['hardware']})==8
 matched=[];clock_reads=0;nummax=0.;allrows=[]
 for i,item in enumerate(q['plan']):
  rows=[]
  for owner in q['owners']:
   assert owner['normal_close']and owner['destroy_returned']and owner['Runtime_expired']and owner['actual_affinity']==owner['expected_affinity']
   assert len(owner['calls'])==5;r=owner['calls'][i];s=r['stats'];rows.append(r);allrows.append((owner['rank'],r))
   assert r['index']==i and s['counts']==[1,1,1,0,0]and s['errors']==[0]*5 and s['clocks_valid']and s['phase_counts']==[1]*5 and s['phase_errors']==[0]*5
   assert s['clock_reads']==(32 if item['timing']else 0);clock_reads+=s['clock_reads']
   n=r['numeric'];assert n['finite']and n['TopK512_unique_valid']and n['state_finite']and n['state_prepare_readonly']and n['valid_cache_prepare_commit_unchanged']and n['CPU_FP8_score_NRMSE']<2e-4;nummax=max(nummax,n['CPU_FP8_score_NRMSE'])
   if not item['timing']:assert s['phase_wall_ns']==s['phase_thread_CPU_ns']==s['status_wall_ns']==s['status_thread_CPU_ns']==[0]*5
   else:
    assert all(x>0 for x in s['phase_wall_ns'])and all(x>0 for x in s['phase_thread_CPU_ns'])
    assert sum(s['phase_wall_ns'])/1e9<=s['prepare_s']+1e-8
    for child,parent in [(0,1),(1,2),(2,3)]:assert s['status_wall_ns'][child]<=s['phase_wall_ns'][parent]
   proof=next(x for x in q['raw_archives']if x['query']==i and x['rank']==owner['rank'])
   meta=next(v for name,v in files.items()if name.endswith('/'+PurePosixPath(proof['metadata']).name)and('/Deep413-phases-'+watch['actual_GPU']+'/')in name)
   assert meta['SHA256']==proof['metadata_SHA256'];md=json.loads(meta['text']);assert md['stats']==s and md['item']==item
  critical=max(range(8),key=lambda k:rows[k]['stats']['prepare_s']+rows[k]['stats']['commit_s']);s=rows[critical]['stats'];pw=[x/1e9 for x in s['phase_wall_ns']];pc=[x/1e9 for x in s['phase_thread_CPU_ns']]
  c=dict(index=i,phase=item['phase'],timing=item['timing'],CPP_critical_owner=critical,max8_CPP_prepare_commit_s=s['prepare_s']+s['commit_s'],max8_CAPI_wall_s=max(r['CAPI_wall_s']for r in rows),same_trial_checked_batch_wall_s=max(r['measurement_end_s']for r in rows)-min(r['measurement_begin_s']for r in rows),critical_owner_prepare_s=s['prepare_s'],critical_owner_commit_s=s['commit_s'],critical_owner_phase_wall_s=pw,critical_owner_phase_thread_CPU_s=pc,critical_owner_phase_fraction_of_prepare=[x/s['prepare_s']for x in pw],critical_owner_unenclosed_prepare_wall_s=s['prepare_s']-sum(pw)if item['timing']else None,critical_owner_status_wall_s=[x/1e9 for x in s['status_wall_ns']],critical_owner_status_thread_CPU_s=[x/1e9 for x in s['status_thread_CPU_ns']],per_owner_prepare_s=[r['stats']['prepare_s']for r in rows],per_owner_phase_wall_s=[[x/1e9 for x in r['stats']['phase_wall_ns']]for r in rows],per_owner_phase_thread_CPU_s=[[x/1e9 for x in r['stats']['phase_thread_CPU_ns']]for r in rows])
  for k in ['CPP_critical_owner','max8_CPP_prepare_commit_s','max8_CAPI_wall_s','same_trial_checked_batch_wall_s']:assert math.isclose(c[k],q['matched_queries'][i][k],rel_tol=1e-12)
  matched.append(c)
 assert len(q['raw_archives'])==len(a['NPZ_archives'])==40 and clock_reads==512
 journal=next(v['text']for n,v in files.items()if n.endswith('/scalar.jsonl'));jr=[json.loads(x)for x in journal.splitlines()];assert len(jr)==40
 for rank,r in allrows:assert sum(x==dict(rank=rank,**r)for x in jr)==1
 for phase in ['fresh_before','post']:
  x=watch[phase];assert x['KFD_available']and not x['kfd_pids']and not x['matching_component_processes']and not x['port43882_listening']
 off0,on,off1=matched[2:5];keys=['max8_CPP_prepare_commit_s','max8_CAPI_wall_s','same_trial_checked_batch_wall_s'];drift={k:off1[k]/off0[k]-1 for k in keys};observer={k:on[k]/statistics.mean([off0[k],off1[k]])for k in keys}
 individual=[]
 for owner in q['owners']:
  x,y,z=[owner['calls'][i]for i in [2,3,4]];individual.append(dict(rank=owner['rank'],CAPI_OFF_drift=z['CAPI_wall_s']/x['CAPI_wall_s']-1,CAPI_ON_over_mean_OFF=y['CAPI_wall_s']/statistics.mean([x['CAPI_wall_s'],z['CAPI_wall_s']]),prepare_ON_over_mean_OFF=y['stats']['prepare_s']/statistics.mean([x['stats']['prepare_s'],z['stats']['prepare_s']])))
 return dict(passed=True,original_archive_SHA256=hashlib.sha256(b).hexdigest(),actual_GPU=watch['actual_GPU'],actual_CPU_keeper=watch['CPU_keeper'],publication_commit=watch['publication_commit'],report_SHA256=watch['report_SHA256'],manifest_SHA256=watch['manifest_SHA256'],phase_labels=LABELS,matched_queries=matched,formal_OFF_endpoint_drift=drift,formal_observer_ON_over_mean_OFF=observer,per_owner_formal=individual,owner_query_records=40,formal_owner_records=24,warm_owner_records=16,timed_phase_sites=80,timed_status_sites=48,total_clock_reads=clock_reads,O1_journal_lines=40,numeric_record_max_CPU_FP8_NRMSE=nummax,raw_NPZ_total_bytes=sum(v['bytes']for v in a['NPZ_archives']),raw_NPZ_not_in_Git=True,driver_wall_s=q['driver_wall_s'],pre_owner_setup_s=q['weight_read_input_prepare_s'],sum_same_trial_max8_CPP_all5_s=sum(x['max8_CPP_prepare_commit_s']for x in matched),sum_same_trial_max8_CPP_formal3_s=sum(x['max8_CPP_prepare_commit_s']for x in matched[2:]),all8_destroy_Runtime_expired_and_post_clean=True,accounting_GPU=a['910374_accounting'],accounting_CPU_keeper=a['910407_accounting'],decision='DIAGNOSTIC_ONLY: TPsum+TopK through original finalsync is largest instrumented phase; observer effect must be retained. No graph/provider/Model speedup or exclusive collective/TopK/CPU attribution.',limits=['Fixed one-layer synthetic hidden/q_lora with realcheckpoint smallweights; not current Model activation/context cost','Original completion enclosures include upstream queued work; threadCPU can include spin, and wall minus threadCPU is not a specific GPU/CCL duration','Five same-owner source windows disjoint; status is nested and must not be added; unenclosed remainder is not CPUdispatch','One formalON and twoOFF endpoints; observed ON phases cannot be treated as unperturbed shares if observer overhead is material','No multiplication by21/43/owners, no endtoend Model/SLO or GPUbusy claim','Original in-run CPU FP8 oracle/checks verified; this stdlib reader does not independently recompute raw NPZ math or independent temporal commit oracle'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');x=p.parse_args();print(json.dumps(analyze(x.archive),indent=2))
