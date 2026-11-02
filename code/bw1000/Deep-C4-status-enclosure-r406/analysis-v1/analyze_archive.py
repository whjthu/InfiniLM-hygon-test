"""stdlib-only reader of original report/watch/40 sidecars; no GPU or Model."""
from pathlib import Path
import json,hashlib,argparse,statistics,math
def analyze(path):
 b=Path(path).read_bytes();a=json.loads(b);files=a['raw_files']
 for name,v in files.items():assert hashlib.sha256(v['text'].encode()).hexdigest()==v['SHA256']and len(v['text'].encode())==v['bytes'],name
 watch=json.loads(next(v['text']for n,v in files.items()if n.endswith('-watch.json')))
 report_name=watch['report_path'];q=json.loads(files[report_name]['text']);assert files[report_name]['SHA256']==watch['report_SHA256']
 assert watch['passed']and watch['srun_exit_code']==0 and q['passed']and q['completed']and q['normal']
 assert [x['timing']for x in q['plan']]==[0,1,0,1,0]and len(q['owners'])==8
 allrows=[];matched=[];clock_reads=0;timed_sites=0;nummax=0
 for i,item in enumerate(q['plan']):
  rows=[]
  for owner in q['owners']:
   assert owner['normal_close']and owner['destroy_returned']and owner['Runtime_expired']and owner['actual_affinity']==owner['expected_affinity']
   assert len(owner['calls'])==5;r=owner['calls'][i];st=r['stats'];rows.append(r);allrows.append(r)
   assert r['index']==i and st['counts']==[1,1,1,0,0]and st['errors']==[0]*5 and st['clocks_valid']
   assert st['clock_reads']==(12 if item['timing']else 0);clock_reads+=st['clock_reads']
   assert r['numeric']['finite']and r['numeric']['TopK512_unique_valid']and r['numeric']['state_finite']and r['numeric']['state_prepare_readonly']and r['numeric']['valid_cache_prepare_commit_unchanged']
   assert r['numeric']['CPU_FP8_score_NRMSE']<2e-4;nummax=max(nummax,r['numeric']['CPU_FP8_score_NRMSE'])
   if item['timing']:timed_sites+=3
   else:assert st['status_wall_ns']==st['status_thread_CPU_ns']==[0]*5
   proof=next(x for x in q['raw_archives']if x['query']==i and x['rank']==owner['rank'])
   actual_meta=next(v for name,v in files.items()if name.endswith(Path(proof['metadata']).name)and '/Deep406-status-910374.1336/'in name)
   assert actual_meta['SHA256']==proof['metadata_SHA256'];meta=json.loads(actual_meta['text']);assert meta['stats']==st and meta['item']==item
  critical=max(range(8),key=lambda k:rows[k]['stats']['prepare_s']+rows[k]['stats']['commit_s']);s=rows[critical]['stats'];status=sum(s['status_wall_ns'])/1e9
  calc=dict(index=i,phase=item['phase'],timing=item['timing'],CPP_critical_owner=critical,max8_CPP_prepare_commit_s=s['prepare_s']+s['commit_s'],max8_CAPI_wall_s=max(r['CAPI_wall_s']for r in rows),same_trial_checked_batch_wall_s=max(r['measurement_end_s']for r in rows)-min(r['measurement_begin_s']for r in rows),critical_owner_prepare_s=s['prepare_s'],critical_owner_commit_s=s['commit_s'],critical_owner_status_wall_s=status,critical_owner_status_thread_CPU_s=sum(s['status_thread_CPU_ns'])/1e9,critical_owner_status_fraction_of_prepare=status/s['prepare_s'],per_owner_status_wall_s=[sum(r['stats']['status_wall_ns'])/1e9 for r in rows],per_owner_status_thread_CPU_s=[sum(r['stats']['status_thread_CPU_ns'])/1e9 for r in rows])
  stored=q['matched_queries'][i]
  for k in ['CPP_critical_owner','max8_CPP_prepare_commit_s','max8_CAPI_wall_s','same_trial_checked_batch_wall_s']:assert math.isclose(calc[k],stored[k],rel_tol=1e-12)
  matched.append(calc)
 assert len(q['raw_archives'])==40 and len(a['NPZ_archives'])==40 and timed_sites==48 and clock_reads==192
 for phase in ['fresh_before','post']:assert watch[phase]['KFD_available']and not watch[phase]['kfd_pids']and not watch[phase]['matching_component_processes']and not watch[phase]['port43882_listening']
 off0,on,off1=matched[2:5];drift={};observer={}
 for key in ['max8_CPP_prepare_commit_s','max8_CAPI_wall_s','same_trial_checked_batch_wall_s']:
  drift[key]=off1[key]/off0[key]-1;observer[key]=on[key]/statistics.mean([off0[key],off1[key]])
 return dict(passed=True,original_archive_SHA256=hashlib.sha256(b).hexdigest(),actual_GPU=watch['actual_GPU'],actual_CPU_keeper=watch['CPU_keeper'],publication_commit=watch['publication_commit'],report_SHA256=watch['report_SHA256'],manifest_SHA256=watch['manifest_SHA256'],matched_queries=matched,formal_OFF_endpoint_drift=drift,formal_observer_ON_over_mean_OFF=observer,owner_query_records=40,measured_query_C4_status_checks=120,timed_status_sites=timed_sites,status_clock_reads=clock_reads,prefix_status_counts_not_reported=True,numeric_record_max_CPU_FP8_NRMSE=nummax,raw_NPZ_total_bytes=sum(v['bytes']for v in a['NPZ_archives']),raw_NPZ_not_in_Git=True,driver_wall_s=q['driver_wall_s'],pre_owner_setup_s=q['weight_read_input_prepare_s'],pre_owner_setup_scope='Includes selected checkpoint mmap/input creation/Native dlopen/Torch device initialization/provider hashing, not isolated weight loading',sum_same_trial_max8_CPP_all5_s=sum(v['max8_CPP_prepare_commit_s']for v in matched),sum_same_trial_max8_CPP_formal3_s=sum(v['max8_CPP_prepare_commit_s']for v in matched[2:]),all8_destroy_Runtime_expired_and_post_clean=True,accounting_GPU=a['910374_accounting'],accounting_CPU_keeper=a['910407_accounting'],decision='STOP deferred-status/containment change on this evidence: completed-check enclosures are not dominant in this one-layer component. Model78ms/current43layer shared-process contribution remains UNKNOWN.',limits=['Observed one-layer synthetic hidden/q_lora and realcheckpoint smallweights, not Model activations','Status enclosure includes upstream GPU completion wait/CPU/spin; ThreadCPU is occupancy, not pure dispatch or recoverable time','One formal ON and two OFF endpoints; observer ratios are diagnostics, not speedup','Do not multiply21/43 or interpret Slurm51s as GPUbusy','Independent temporal commit oracle not asserted; state real dtype finite/prepare readonly/cache unchanged/normal commit checked','stdlib reader verifies raw-byte provenance, all recorded numerical gates and sidecar correspondence; optional NPZ reader recomputes original CPU FP8 oracle'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');args=p.parse_args();print(json.dumps(analyze(args.archive),indent=2))
