"""Original two-rank380 records audit; no torch/GPU/model imports."""
import pathlib,json,hashlib,statistics,math,sys
MF='d603d60fddaa495bec9493ec5f8bd13507ad265961d3a1f25c1d8fe5c268e0a9'
REPORTS=['45e9a254835f22cf99be762e884248d893dc7742b5dea5c5adabeacd35dd5f80','f0816ee02560c0dd707ed4dc5fcdde18de6c47daf8cd84bb55525fd2a9ff2b9e']
def audit(path):
 path=pathlib.Path(path);j=json.loads(path.read_bytes());watch=j['watch'];w=watch['data'];assert hashlib.sha256((json.dumps(w,indent=2)+'\n').encode()).hexdigest()==watch['SHA256']
 assert w['publication_commit']=='448237a2d8bf57e2c982ceafd1672f2225d4d8c4'and w['manifest_SHA256']==MF and w['actual_GPU']=='910371.707'and w['CPU_keeper_actual']=='910407.763'and w['passed']and w['srun_client_exit']==0 and w['completed']and not w['Model_loaded']
 for key,step in [('fresh_before','910371.706'),('post','910371.708')]:
  v=w[key];assert v['actual_step']==step and v['node']=='a02r3n07'and v['KFD_available']and not v['kfd_pids']and not v['matching_component_processes']
 assert len(j['sacct'])==4 and all(v.split('|')[2:4]==['COMPLETED','0:0']for v in j['sacct']);assert len(j['reports'])==2
 summary=[];ranks=[];slower=exceed=0;measured_sum=0.;private_bytes=0;maxnr=base_nr=half_nr=0.;fallbacks=0
 expected={'baseline':26,'hit':26,'range_fallback':1,'weight_range_fallback':0,'output_fallback':0,'rejected_nonfinite':0}
 for item in j['reports']:
  r=item['data'];rank=r['rank'];assert rank in (0,1)and item['SHA256']==REPORTS[rank];assert hashlib.sha256((json.dumps(r,indent=2)+'\n').encode()).hexdigest()==item['SHA256']
  assert r['passed']and r['completed']and r['normal_close']and not r['cleanup_errors']and r['all_owned_module_parameter_weakrefs_expired']and not r['Model_loaded']and not r['full_draft_loaded']and not r['target_head_loaded']
  assert r['device']['physical_mask']==str(rank)and r['device']['visible_device']==0 and r['layer']==(0,4)[rank]and r['source_manifest_SHA256']==MF
  assert all(v is False for v in r['matmul_policy']['effective_FP16_flags'].values())and r['matmul_policy']['BF16_reduced_precision_unchanged']is True
  assert len(r['cases'])==4 and all(v==expected for v in r['final_counts'].values())and set(r['final_counts'])=={'KVA','KVB'}
  assert len(r['weight_loads'])==2
  for load in r['weight_loads']:
   assert load['CPU_to_GPU_FP16_storage_exact_diagnostic']and load['GPU_storage_different_from_CPU_half_count']==0 and load['FP16_conversion']['FP16_nonfinite']==0;private_bytes+=load['private_FP16_bytes']
  for index,c in enumerate(r['cases']):
   name,M=[('KVA',32646),('KVB',32646),('KVA',8),('KVB',8)][index];assert c['projection']==name and c['M']==M and (c['K'],c['N'])==((7168,576)if name=='KVA'else(512,16384))and c['seed']==380000+rank*100+index
   assert c['input_and_weight_state_unchanged']and c['candidate_all_hits']and c['measurement_call_count']==20 and len(c['groups'])==5 and len(c['warm'])==4
   assert [a['mode']for a in c['warm']]==[0,1,0,1]
   for label in ['baseline_qualification','candidate_qualification']:
    q=c[label]['independent_CPU'];assert q['finite']and q['NRMSE']<.02 and math.isfinite(q['NRMSE']);assert c[label]['path']==('baseline'if label.startswith('baseline')else'hit')
   base_nr=max(base_nr,c['baseline_qualification']['independent_CPU']['NRMSE']);half_nr=max(half_nr,c['candidate_qualification']['independent_CPU']['NRMSE']);maxnr=max(maxnr,c['whole_output_GPU_F32_NR']);assert math.isfinite(c['whole_output_GPU_F32_NR'])and c['whole_output_GPU_F32_NR']<.02
   groups=[];wall=0.
   for group,g in enumerate(c['groups']):
    assert g['group']==group and [a['mode']for a in g['arms']]==[0,1,1,0];times=[a['wall_s']for a in g['arms']];assert all(math.isfinite(t)and t>0 for t in times)
    for a in g['arms']+c['warm']:
     assert a['path']==('hit'if a['mode']else'baseline');delta={k:0 for k in expected};delta['hit'if a['mode']else'baseline']=1;assert a['count_delta']==delta
    off=(times[0]+times[3])/2;on=(times[1]+times[2])/2;ratio=on/off;drift=abs(times[3]-times[0])/off
    assert math.isclose(off,g['OFF_s'],rel_tol=1e-14)and math.isclose(on,g['ON_s'],rel_tol=1e-14)and math.isclose(ratio,g['ratio'],rel_tol=1e-14)and math.isclose(drift,g['OFF_endpoint_drift'],rel_tol=1e-14)
    slower+=on>off;exceed+=ratio-1>drift;wall+=sum(times);groups.append(dict(group=group,OFF_us=off*1e6,ON_us=on*1e6,ratio=ratio,OFF_endpoint_drift=drift,arms=g['arms']))
   assert math.isclose(wall,c['measured_wall_sum_s'],rel_tol=1e-14);measured_sum+=wall
   summary.append(dict(rank=rank,layer=r['layer'],projection=name,M=M,K=c['K'],N=c['N'],seed=c['seed'],median_OFF_us=statistics.median(g['OFF_us']for g in groups),median_ON_us=statistics.median(g['ON_us']for g in groups),median_same_group_ratio=statistics.median(g['ratio']for g in groups),max_OFF_endpoint_drift=max(g['OFF_endpoint_drift']for g in groups),all_groups_candidate_slower=all(g['ratio']>1 for g in groups),whole_output_NR=c['whole_output_GPU_F32_NR'],baseline_independent_CPU=c['baseline_qualification'],candidate_independent_CPU=c['candidate_qualification'],all_groups=groups,whole_case_including_fixture_qual_state_and_checks_wall_s=c['whole_case_including_fixture_qual_state_and_checks_wall_s'],Torch_peak_allocated_bytes=c['Torch_peak_allocated_bytes'],Torch_peak_scope=c['peak_scope']))
  assert len(r['range_fallback_probes'])==2
  for p in r['range_fallback_probes']:
   assert p['path']=='range_fallback';delta={k:0 for k in expected};delta['range_fallback']=1;assert p['count_delta']==delta;fallbacks+=1
  ranks.append(dict(rank=rank,layer=r['layer'],physical_mask=r['device']['physical_mask'],device=r['device'],driver_wall_s=r['whole_driver_wall_s'],prepare_wall_s=r['prepare_wall_s'],measured_arms_wall_s=sum(c['measured_wall_sum_s']for c in r['cases']),whole_cases_wall_s=sum(c['whole_case_including_fixture_qual_state_and_checks_wall_s']for c in r['cases']),final_counts=r['final_counts'],weight_loads=r['weight_loads'],matmul_policy=r['matmul_policy'],cleanup_errors=r['cleanup_errors'],normal_close=r['normal_close'],raw_report_path=item['path'],raw_report_SHA256=item['SHA256']))
 assert len({r['rank']for r in ranks})==2 and slower==40 and private_bytes==50069504 and fallbacks==4
 return dict(passed=True,decision='STOP_GUARDED_FP16_DTYPE_ROUTE',state='NUMERIC_AND_LIFETIME_PASS_PERFORMANCE_NEGATIVE_CLEAN',scope='Four actual checkpoint KV projections with synthetic inputs; complete percall guarded FP16 compute vs original raw BF16Linear, no fullquery/head/Model',actual_GPU='910371.707',actual_keeper='910407.763',source_publication_commit=w['publication_commit'],GPU_manifest_SHA256=MF,cases=summary,ranks=ranks,paired_groups=40,groups_candidate_slower=slower,groups_slowdown_exceeds_own_OFF_drift=exceed,measured_calls=160,total_wrapper_or_Linear_calls=212,actual_range_fallback_probes=4,measured_candidate_fallbacks=0,max_whole_output_GPU_F32_NR=maxnr,max_baseline_sample_CPU_NR=base_nr,max_candidate_sample_CPU_half_NR=half_nr,additional_private_FP16_weight_bytes=private_bytes,GPU_storage_nonzero_to_zero=sum(t['GPU_storage_nonzero_to_zero']for z in j['reports']for t in z['data']['weight_loads']),summed_two_rank_measured_complete_call_wall_s=measured_sum,GPU_busy='NOT_MEASURED',Slurm=j['sacct'],Slurm_GPUstep_s=58,Slurm_GPUcards=2,Slurm_keeper_s=61,fresh=w['fresh_before'],post=w['post'],raw_watch_path=watch['path'],raw_watch_SHA256=watch['SHA256'],local_archive=str(path),local_archive_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),M8_max_OFF_drift_retained=max(c['max_OFF_endpoint_drift']for c in summary if c['M']==8),bare_FP16_GEMM_performance='NOT_ISOLATED',BF16_physical_peak='NOT_CERTIFIED',GPU_output_overflow_fallback_qualification='NOT_TESTED_CPU_ONLY',Model_gain='UNMEASURED',quality_or_acceptance='NOT_QUALIFIED',no_Model_load=True,no_new_dtype_or_size_grid=True,no_more_GPU=True)
if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2)+'\n')
