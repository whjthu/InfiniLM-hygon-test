"""Stdlib original scalar/sidecar reader; no device access or raw NPZ oracle replay."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math,statistics

def analyze(path):
 b=Path(path).read_bytes();a=json.loads(b);files=a['raw_files']
 for n,v in files.items():
  raw=v['text'].encode();assert hashlib.sha256(raw).hexdigest()==v['SHA256'] and len(raw)==v['bytes'],n
 w=json.loads(next(v['text']for n,v in files.items()if n.endswith('-watch.json')));q=json.loads(files[w['report_path']]['text'])
 assert files[w['report_path']]['SHA256']==w['report_SHA256'] and w['passed'] and w['srun_exit_code']==0
 assert q['completed'] and q['passed'] and q['normal'] and q['Model_loaded'] is False and q['phase_observer'] is False
 assert [p['mode']for p in q['plan']]==[0,1,0,1,1,0,1,0,0,1] and len(q['owners'])==8 and len({x['PCI']for x in q['hardware']})==8
 assert q['initial_mapped_providers']==q['final_mapped_providers']
 assert q['additional_value_proof_D2H_bytes_per_query']==2048 and q['additional_value_proof_sync_calls']==0 and not q['additional_value_proof_wait_free']
 assert len(q['raw_archives'])==len(a['NPZ_archives'])==80
 matched=[];allrows=[];nr=0.;margin=[];modes=[0,1,0,1,1,0,1,0,0,1]
 for i,item in enumerate(q['plan']):
  rows=[]
  for owner in q['owners']:
   assert owner['normal_close'] and owner['destroy_returned'] and owner['Runtime_expired'] and owner['actual_affinity']==owner['expected_affinity']
   assert len(owner['calls'])==10
   r=owner['calls'][i];s=r['stats'];n=r['numeric'];rows.append(r);allrows.append((owner['rank'],r))
   assert r['index']==i and r['mode']==item['mode'] and s['topk_calls']==1 and s['topk_hits']==modes[i] and s['Torch_fallbacks']==1-modes[i] and s['phase_clock_reads']==0
   assert s['owner_thread_hash']==owner['calls'][0]['stats']['owner_thread_hash']
   for key in ['finite','TopK512_unique_valid','TopK_values_equal_selected_original_score_bits','TopK_exact_ranked_value_set_on_actual_F32_scores','TopK_sorted_largest','state_finite','state_prepare_readonly','state_commit_read','valid_cache_prepare_commit_unchanged']:assert n[key],key
   assert n['CPU_FP8_score_NRMSE']<2e-4;nr=max(nr,n['CPU_FP8_score_NRMSE']);margin.append(n['cutoff_margin_512_513'])
   assert r['additional_proof_bytes']==2048 and r['proof_async_copy_added']==1 and r['proof_sync_calls_added']==0 and not r['proof_extra_wait_free']
   assert math.isclose(r['forward_and_proof_CAPI_s'],r['forward_CAPI_only_s']+r['proof_and_related_checks_s'],rel_tol=1e-12)
   assert r['CAPI_wall_s']+1e-12>=r['forward_and_proof_CAPI_s']+r['restore_group_CAPI_s']
   z=r['freeze_restore'];assert z['captures']==1 and z['restores']==i+1 and z['prefix_generation']==z['restored_generation']==i+1 and z['valid_raw_bytes']==280368 and z['state_bytes']==8192 and z['restore_D2D_bytes']==288560 and z['D2D_copies']==4 and z['same_frozen_effective_payload'] and z['metadata_from_real_prefix']
   base=owner['calls'][0]['freeze_restore'];assert all(z[k]==base[k]for k in ['valid_raw_SHA256','state_SHA256','state_dtype','state_shape','input_identity'])
   assert n['valid_cache_SHA256']==z['valid_raw_SHA256']
   if i:assert z['realprefix_revision']>owner['calls'][i-1]['freeze_restore']['realprefix_revision']
   proof=next(x for x in q['raw_archives']if x['query']==i and x['rank']==owner['rank'])
   meta=next(v for name,v in files.items()if name.endswith('/'+PurePosixPath(proof['metadata']).name)and('/Deep416-TopK-'+w['actual_GPU']+'/')in name)
   assert meta['SHA256']==proof['metadata_SHA256'];md=json.loads(meta['text']);assert md['stats']==s and md['item']==item and md['fields']['11']['shape']==[1,512] and md['fields']['9']['SHA256']==r['freeze_restore']['state_SHA256']
  crit=max(range(8),key=lambda k:rows[k]['stats']['prepare_s']+rows[k]['stats']['commit_s'])
  v=dict(**item,CPP_critical_owner=crit,max8_CPP_prepare_commit_s=rows[crit]['stats']['prepare_s']+rows[crit]['stats']['commit_s'],max8_CAPI_wall_s=max(r['CAPI_wall_s']for r in rows),max8_forward_CAPI_only_s=max(r['forward_CAPI_only_s']for r in rows),max8_forward_and_proof_CAPI_s=max(r['forward_and_proof_CAPI_s']for r in rows),max8_restore_group_CAPI_s=max(r['restore_group_CAPI_s']for r in rows),max8_proof_and_related_checks_s=max(r['proof_and_related_checks_s']for r in rows),same_trial_checked_batch_wall_s=max(r['measurement_end_s']for r in rows)-min(r['measurement_begin_s']for r in rows),max8_prefix_rebuild_s=max(r['prefix_rebuild_s']for r in rows))
  for key in ['CPP_critical_owner','max8_CPP_prepare_commit_s','max8_CAPI_wall_s','same_trial_checked_batch_wall_s']:assert math.isclose(v[key],q['matched_queries'][i][key],rel_tol=1e-12)
  matched.append(v)
 journal=next(v['text']for n,v in files.items()if n.endswith('/scalar.jsonl'));jr=[json.loads(x)for x in journal.splitlines()];assert len(jr)==80
 for rank,r in allrows:assert sum(x==dict(rank=rank,**r)for x in jr)==1
 for phase in ['fresh_before','post']:
  x=w[phase];assert x['KFD_available'] and not x['kfd_pids'] and not x['matching_component_processes'] and not x['port43882_listening']
 keys=['max8_CPP_prepare_commit_s','max8_CAPI_wall_s','max8_forward_CAPI_only_s','max8_forward_and_proof_CAPI_s','max8_restore_group_CAPI_s','same_trial_checked_batch_wall_s']
 formal={}
 for key in keys:
  off=statistics.mean(matched[i][key]for i in [2,5,7,8]);on=statistics.mean(matched[i][key]for i in [3,4,6,9])
  formal[key]=dict(mean_OFF_s=off,mean_ON_s=on,ON_over_OFF=on/off,change_fraction=on/off-1,OFF_endpoint_drift=matched[8][key]/matched[2][key]-1,block_ABBA_OFF_drift=matched[5][key]/matched[2][key]-1,block_BAAB_OFF_drift=matched[8][key]/matched[7][key]-1,adjacent_ON_over_OFF=[matched[x][key]/matched[y][key]for x,y in [(3,2),(4,5),(6,7),(9,8)]])
 cross=[r['numeric']for o in q['owners']for r in o['calls'][1:]]
 individual=[]
 for o in q['owners']:
  rows=o['calls'];d=dict(rank=o['rank'])
  for key in ['CAPI_wall_s','forward_CAPI_only_s']:
   d[key]=dict(ON_over_OFF=statistics.mean(rows[i][key]for i in [3,4,6,9])/statistics.mean(rows[i][key]for i in [2,5,7,8]),OFF_endpoint_drift=rows[8][key]/rows[2][key]-1,block_ABBA_OFF_drift=rows[5][key]/rows[2][key]-1,block_BAAB_OFF_drift=rows[8][key]/rows[7][key]-1)
  d['CPP_prepare_commit']=dict(ON_over_OFF=statistics.mean(rows[i]['stats']['prepare_s']+rows[i]['stats']['commit_s']for i in [3,4,6,9])/statistics.mean(rows[i]['stats']['prepare_s']+rows[i]['stats']['commit_s']for i in [2,5,7,8]),OFF_endpoint_drift=(rows[8]['stats']['prepare_s']+rows[8]['stats']['commit_s'])/(rows[2]['stats']['prepare_s']+rows[2]['stats']['commit_s'])-1)
  individual.append(d)
 assert all(x['cross_trial_valid_cache_same']and x['cross_trial_state_same']and x['cross_trial_score_maxabs']==0 for x in cross)
 return dict(passed=True,all80_frozen_input_signatures_and_prestate_sidecar_hash_pass=True,same_frozen_input_stage_signature_source='Actual sameowner CAPI reads effectivevalidraw and BF16state afterrestore; this stdlibreader checks scalar signatures/field9SHA, not independentNPZ byte extraction',original_archive_SHA256=hashlib.sha256(b).hexdigest(),actual_GPU=w['actual_GPU'],actual_CPU_keeper=w['CPU_keeper'],GPU_epoch_unix=w['GPU_epoch_unix'],publication_commit=w['publication_commit'],manifest_SHA256=w['manifest_SHA256'],report_SHA256=w['report_SHA256'],matched_queries=matched,formal_ABBA_BAAB=formal,per_owner_formal=individual,owner_query_records=80,formal_owner_records=64,warm_owner_records=16,ON_owner_hits=40,OFF_owner_Torch_fallbacks=40,numeric_max_CPU_FP8_NRMSE=nr,cross_trial_comparisons=len(cross),cross_trial_same_valid_cache=sum(x['cross_trial_valid_cache_same']for x in cross),cross_trial_same_state=sum(x['cross_trial_state_same']for x in cross),cross_trial_max_score_difference=max(x['cross_trial_score_maxabs']for x in cross),min_cutoff_margin_512_513=min(margin),all80_strict_ranked_F32_values_and_original_bits_pass=True,all8_destroy_Runtime_expired_and_post_clean=True,additional_value_proof_per_owner_query_bytes=2048,additional_proof_copy_calls=1,additional_proof_sync_calls=0,additional_proof_wait_free=False,raw_NPZ_bytes=sum(x['bytes']for x in a['NPZ_archives']),driver_wall_s=q['driver_wall_s'],weight_read_input_prepare_s=q['weight_read_input_prepare_s'],sum_same_trial_max8_CPP_all10_s=sum(x['max8_CPP_prepare_commit_s']for x in matched),sum_same_trial_max8_CPP_formal8_s=sum(x['max8_CPP_prepare_commit_s']for x in matched[2:]),sum_same_trial_checked_all10_s=sum(x['same_trial_checked_batch_wall_s']for x in matched),accounting_GPU=a['910374_accounting'],accounting_CPU_keeper=a['910407_accounting'],decision='STOP: staging-inclusive completeCAPI point improvement is within formal OFF block variation; checkedwall has no gain, CPP/forward variance is large. Preserve numeric/frozeninput qualification; no grid, measurement repair replay or Model load.',limits=['Each trueprefix regenerates metadata then restores first frozen validraw/BF16state; perquery signature and sidecar beforestate checked. Raw NPZ floating oracle not independently recomputed by this stdlib reader','New restore/signature staging dominates primaryCAPI; its wall is an enclosure, not proof of CPUdispatch, GIL or GPUtime; completeCAPI max and stage max need not have samecriticalowner','One fixed actual layer16 smallweight fixture with synthetic hidden/q_lora; no Model activation, gain, quality, fit, SLO or peak-memory certificate','Warm first-use costs retained but excluded from formal ABBA; GPU Slurm elapsed is not GPU busy time','CAPI complete includes restore/signature plus forward/integrity/stat/proof; restore,forward-only,forward+proof and checked H2D/CPUoracle/archive/barriers are separate','Both arms add one2048B values proof copy to an existing drain: same sync count, additional work and waiting are not free','CPU14 Torch/shared comparator covers nonfinite corners; actual GPU qualification uses finite real C4 fixture scores, not a separate nonfinite GPU suite','This stdlib reader validates original scalar/sidecar guards and hashes; it does not independently recompute NPZ floating oracle or independent temporal commit math','No summing per-stage maxima, no multiplication by21/43, no cross-node absolute clock subtraction'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');args=p.parse_args();print(json.dumps(analyze(args.archive),indent=2))
