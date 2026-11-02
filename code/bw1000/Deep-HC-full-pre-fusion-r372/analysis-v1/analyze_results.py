"""Offline original-report consumer. Maxima always use one identical trial index."""
import pathlib,json,argparse,statistics,hashlib

def metric(rows,key):
    off=sum(r[key]for r in rows if r['mode']==0);on=sum(r[key]for r in rows if r['mode']==1);pairs=[]
    for i in range(0,20,4):
        a,b,c,d=rows[i:i+4];ratio=(b[key]+c[key])/(a[key]+d[key]);drift=d[key]/a[key]-1
        pairs.append(dict(ABBA=i//4,ON_over_OFF=ratio,OFF_endpoint_drift=drift,gain_exceeds_absolute_drift=1-ratio>abs(drift)))
    return dict(OFF_us_per_call=off/320*1e6,ON_us_per_call=on/320*1e6,ON_over_OFF=on/off,pairs=pairs,all5_same_direction=all(p['ON_over_OFF']<1 for p in pairs),all5_gain_exceeds_drift=all(p['gain_exceeds_absolute_drift']for p in pairs),median_all5_OFF_us_per_call=statistics.median((rows[i][key]+rows[i+3][key])/64*1e6 for i in range(0,20,4)),median_all5_ON_us_per_call=statistics.median((rows[i+1][key]+rows[i+2][key])/64*1e6 for i in range(0,20,4)))

def analyze(raw,old=None):
    a=raw['archive'];main=a['report'];watch=a['watch'];closes=main['protocol']['closes'];archive={'watch':{'report':watch},'main':{'report':main,'path':watch['report_path'],'SHA256':watch['report_SHA256']},'owners':[{'report':q,'path':c['path'],'SHA256':c['SHA256']}for q,c in zip(a['owners'],closes)]}
    w=archive['watch']['report'];main=archive['main']['report'];p=main['protocol'];assert main['passed']and p['passed']and not any(p['errors'])and not p['incomplete_measurement_trials'];assert all(p['threads_joined'])
    assert len(archive['owners'])==len(p['closes'])==8
    assert [r['rank']for r in p['owners']]==list(range(8))and len({r['TID']for r in p['owners']})==8
    assert all(r['initial_allowed']==main['parent_initial_allowed']for r in p['owners'])and p['parent_final_allowed']==main['parent_initial_allowed']
    masks=[set(r['final_allowed'])for r in p['owners']];assert all(len(m)==4 for m in masks)and all(not masks[i]&masks[j]for i in range(8)for j in range(i))
    measurements=[r for r in p['records']if r['phase']['kind']=='measurement'];assert len(measurements)==20
    assert [r['phase']['mode']for r in measurements]==[0,1,1,0]*5
    aligned=[];owners=[]
    for trial,record in enumerate(measurements):
        assert record['phase']['trial']==trial and len(record['owners'])==8
        a=dict(trial=trial,mode=record['phase']['mode'],whole_phase_wall_s=record['whole_phase_wall_s'],owners=[])
        for rank,r in enumerate(record['owners']):
            q=archive['owners'][rank]['report'];assert q['rank']==rank and q['calls'][trial]==r and r['trial']==trial and r['repeats']==32
            assert len(r['repeat_CAPI_wall_s'])==len(r['repeat_native_component_s'])==32
            assert abs(sum(r['repeat_CAPI_wall_s'])-r['CAPI_wall_s'])<1e-12 and abs(sum(r['repeat_native_component_s'])-r['native_component_s'])<1e-12
            assert r['GPU_hits']==r['mode']*32
            a['owners'].append({k:r[k]for k in ['trial','mode','repeats','CAPI_wall_s','native_component_s','local_loop_wall_s','loop_begin_s','loop_end_s','GPU_hits','max_NRMSE']}|dict(rank=rank))
        for target,source in [('max8_CAPI_wall_s','CAPI_wall_s'),('max8_native_component_s','native_component_s'),('max8_local_loop_wall_s','local_loop_wall_s')]:a[target]=max(r[source]for r in record['owners'])
        a['owner_loop_envelope_s']=max(r['loop_end_s']for r in record['owners'])-min(r['loop_begin_s']for r in record['owners'])
        a['aligned_repeat_sum_max8_CAPI_wall_s']=sum(max(r['repeat_CAPI_wall_s'][i]for r in record['owners'])for i in range(32))
        original=p['aligned_measurements'][trial]
        assert all(abs(a[k]-original[k])<1e-12 for k in ['max8_CAPI_wall_s','max8_native_component_s','max8_local_loop_wall_s','whole_phase_wall_s','owner_loop_envelope_s','aligned_repeat_sum_max8_CAPI_wall_s'])
        aligned.append(a)
    for rank,e in enumerate(archive['owners']):
        q=e['report'];close=p['closes'][rank];assert e['SHA256']==close['SHA256']and q['passed']and q['normal_close']and q['Runtime_expired']==1 and close['normal_close']
        assert q['weights_state_initial']==q['weights_state_final'];assert q['CPU_affinity_after_first_baseline']==p['owners'][rank]['final_allowed']
        assert q['baseline_numeric_first']['actual976f_epilogueON_baseline_hits']==1 and q['candidate_numeric']['GPU_hit']==1
        r=dict(rank=rank,source_report_path=e['path'],source_report_SHA256=e['SHA256'],fixture=q['fixture'],CAPI=metric(q['calls'],'CAPI_wall_s'),native_component=metric(q['calls'],'native_component_s'),local_checked_loop=metric(q['calls'],'local_loop_wall_s'),actual_CAPI_measurement_s=sum(v['CAPI_wall_s']for v in q['calls']),actual_native_component_measurement_s=sum(v['native_component_s']for v in q['calls']),sum_local_checked_loop_s=sum(v['local_loop_wall_s']for v in q['calls']),prepare_and_qualification_s=q['prepare_and_qualification_s'],owner_lifetime_wall_s=q['overall_wall_s'],baseline_numeric=q['baseline_numeric_first']['numerics'],candidate_numeric=q['candidate_numeric']['numerics'],baseline_cold_event_ms=q['baseline_numeric_first']['producer_stream_event_ms'],candidate_cold_event_ms=q['candidate_numeric']['producer_stream_event_ms'],max_measurement_NRMSE=max(v['max_NRMSE']for v in q['calls']),weights_state_unchanged=True,normal_close=True,Runtime_expired=1,CPU_initial=q['CPU_affinity_initial'],CPU_final=q['CPU_affinity_after_first_baseline'],memory_before_context=q['memory_before_context'],memory_after_measurement=q['memory_after_measurement'])
        if old:
            previous=old['ranks'][rank];r['old1277_isolated_context_observation']=dict(OFF_us_per_call=previous['CAPI_OFF_us_per_call'],ON_us_per_call=previous['CAPI_ON_us_per_call'],CPU_mask=previous['CPU_final'],scope='Different earlier8processcontext/CPUallocationlayout. Informational only; no causal global-lock/CPU attribution or oldalignedmax8 reconstruction.')
        owners.append(r)
    metrics={key:metric(aligned,key)for key in ['max8_CAPI_wall_s','max8_native_component_s','max8_local_loop_wall_s','whole_phase_wall_s']}
    candidate=all(metrics[k]['all5_gain_exceeds_drift']for k in ['max8_CAPI_wall_s','max8_native_component_s'])
    return dict(scope='StrictstagedHCpreversuscurrent976fepilogueON;originalMean/Add/Rsqrt/normalizedF32Mul/Gemmunchanged;CastSquareandepilogue-r89fusiononly;singleprocess8actualRuntimeowners,4realcheckpointHCgroupsx2syntheticBF16seeds. NoModel.',publication_commit=w['publication_commit'],manifest_SHA256=w['manifest_SHA256'],actual_GPU=w['actual_GPU'],actual_CPU_keeper=w['CPU_keeper_actual'],terminal=w['terminal'],post=w['post'],main_report_path=archive['main']['path'],main_report_SHA256=archive['main']['SHA256'],all8_numeric_state_hit_and_normalclose=True,all8_threads_joined=True,parent_allowed_unchanged=True,same_trial_max8=metrics,aligned_trials=aligned,owners=owners,complete_HC_component_candidate=candidate,whole_checked_harness_wall_stable_gain=metrics['whole_phase_wall_s']['all5_gain_exceeds_drift'],measurement_calls_per_owner=640,measurement_calls_all_owners=5120,qualification_and_warm_calls_per_owner=4,valid_component_causal_rounds=1,valid_owner_ABBA_groups=40,valid_max8_ABBA_groups=5,parent_prepare_before_threads_s=main['parent_prepare_before_threads_s'],whole_thread_batch_wall_s=p['whole_batch_wall_s'],whole_driver_wall_s=main['whole_driver_wall_s'],sum_measurement_phase_wall_s=sum(r['whole_phase_wall_s']for r in aligned),all_measurement_owner_loop_envelope_s=max(v['loop_end_s']for r in aligned for v in r['owners'])-min(v['loop_begin_s']for r in aligned for v in r['owners']),Model_gain='UNMEASURED',scope_limits='CAPI includesfullproducerwait/copies/drain/FFIreturn; C++timer covers staging+originalHC/candidate+outputcopies/finaldrain. Local-loop andwholephase includeeachcallGPU-to-CPU finite/canary checks; wholephase addsbarriers butexcludesbetween-phase JSONwrites. Threadbatchincludes setup/qualification/warm/measurement/persistence/close; driveralsoincludes parentpreparation. Coldqualificationevents excludedfromABBA; noGPUbusy claim; memorycheckpoints notpeak. Repeat-index maxima onlylogicalalignment, notper-repeat synchronization. No ratio-times-Model HCspan/layercount projection orlock/CPU attribution.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('output');ap.add_argument('--old');a=ap.parse_args();src=pathlib.Path(a.archive);dst=pathlib.Path(a.output);assert not dst.exists()
    q=analyze(json.loads(src.read_bytes()),json.loads(pathlib.Path(a.old).read_bytes())if a.old else None);q['local_raw_archive_SHA256']=hashlib.sha256(src.read_bytes()).hexdigest();raw=json.loads(src.read_bytes());q['remote_raw_archive_path']=raw['remote_archive_path'];q['remote_raw_archive_SHA256']=raw['remote_archive_SHA256'];q['raw_evidence_SHA256']=raw['archive']['evidence_SHA256'];dst.write_text(json.dumps(q,indent=2)+'\n',newline='\n');print(json.dumps({k:q[k]for k in ['actual_GPU','complete_HC_component_candidate','whole_checked_harness_wall_stable_gain','valid_owner_ABBA_groups','valid_max8_ABBA_groups']}))
