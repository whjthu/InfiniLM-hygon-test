"""Offline consumer of original compact component reports; no GPU or service calls."""
import pathlib,json,statistics,argparse,hashlib

def analyze(archive):
    watch=archive['watch']['report'];rows=[]
    for entry in archive['reports']:
        q=entry['report'];r=dict(rank=q['rank'],passed=q.get('passed'),normal_close=q.get('normal_close'),source_report_path=entry['path'],source_report_SHA256=entry['SHA256'],error=q.get('error'))
        if not q.get('passed'):
            r.update(valid_ABBA=0,baseline_numeric_passed='baseline_numeric_first'in q,candidate_numeric_passed='candidate_numeric'in q)
            rows.append(r);continue
        assert q['normal_close']and q['Runtime_expired']==1
        calls=q['calls'];assert len(calls)==20 and [v['mode']for v in calls]==[0,1,1,0]*5
        assert all(v['repeats']==32 and v['GPU_hits']==v['mode']*32 for v in calls)
        assert q['weights_state_initial']==q['weights_state_final']
        pairs=[]
        for i in range(0,20,4):
            a,b,c,d=calls[i:i+4]
            ratio=(b['CAPI_wall_s']+c['CAPI_wall_s'])/(a['CAPI_wall_s']+d['CAPI_wall_s'])
            native=(b['native_component_s']+c['native_component_s'])/(a['native_component_s']+d['native_component_s'])
            drift=d['CAPI_wall_s']/a['CAPI_wall_s']-1
            assert abs(ratio-q['pairs'][i//4]['ON_over_OFF'])<1e-12
            pairs.append(dict(ON_over_OFF=ratio,native_ON_over_OFF=native,OFF_endpoint_drift=drift,gain_exceeds_absolute_drift=1-ratio>abs(drift)))
        arm=lambda mode,key:sum(v[key]for v in calls if v['mode']==mode)
        off=arm(0,'CAPI_wall_s');on=arm(1,'CAPI_wall_s')
        r.update(valid_ABBA=5,fixture=q['fixture'],CAPI_OFF_us_per_call=off/320*1e6,CAPI_ON_us_per_call=on/320*1e6,pooled_ON_over_OFF=on/off,pooled_speedup=off/on,component_OFF_us_per_call=arm(0,'native_component_s')/320*1e6,component_ON_us_per_call=arm(1,'native_component_s')/320*1e6,pairs=pairs,all5_same_direction=all(v['ON_over_OFF']<1 for v in pairs),all5_gain_exceeds_drift=all(v['gain_exceeds_absolute_drift']for v in pairs),baseline_numeric=q['baseline_numeric_first']['numerics'],candidate_numeric=q['candidate_numeric']['numerics'],max_NRMSE_all_measurement_calls=max(v['max_NRMSE']for v in calls),candidate_measurement_hits=arm(1,'GPU_hits'),baseline_qualification_hits=q['baseline_numeric_first']['actual_afc6_baseline_hits'],baseline_qualification_producer_stream_event_ms=q['baseline_numeric_first']['producer_stream_event_ms'],candidate_qualification_producer_stream_event_ms=q['candidate_numeric']['producer_stream_event_ms'],actual_CAPI_measurement_s=off+on,measurement_loop_wall_s=q['measurement_loop_wall_s'],prepare_qualification_s=q['prepare_and_qualification_s'],driver_wall_s=q['overall_wall_s'],CPU_initial=q['CPU_affinity_initial'],CPU_final=q['CPU_affinity_after_first_baseline'],physical_Runtime_device=q['hardware']['Runtime_device'],memory_before_context=q['memory_before_context'],memory_after_measurement=q['memory_after_measurement'])
        r['measurement_calls_per_arm']=320
        r['median_all5_ABBA_OFF_us_per_call']=statistics.median((calls[i]['CAPI_wall_s']+calls[i+3]['CAPI_wall_s'])/64*1e6 for i in range(0,20,4))
        r['median_all5_ABBA_ON_us_per_call']=statistics.median((calls[i+1]['CAPI_wall_s']+calls[i+2]['CAPI_wall_s'])/64*1e6 for i in range(0,20,4))
        r['qualification_cold_CAPI_OFF_s']=q['baseline_numeric_first']['CAPI_wall_s']
        r['qualification_cold_CAPI_ON_s']=q['candidate_numeric']['CAPI_wall_s']
        rows.append(r)
    good=[v for v in rows if v['passed']]
    if len(good)==8:
        masks=[set(v['CPU_final'])for v in good];assert all(len(v)==4 for v in masks)and all(not masks[i]&masks[j]for i in range(8)for j in range(i));assert {v['physical_Runtime_device']for v in good}==set(range(8))
    return dict(scope='Full T1 HyperConnection::pre component including staging/output copies/final drain. Actual afc6 baseline and actual HC small parameter tiles; synthetic BF16 residuals, not service activations. No Model speed or quality promotion.',publication_commit=watch['publication_commit'],manifest_SHA256=watch['manifest_SHA256'],GPU=watch.get('actual_GPU'),CPU_keeper=watch['CPU_keeper_actual'],terminal=watch.get('terminal'),before=watch.get('fresh_before'),post=watch.get('post'),valid_ranks=len(good),valid_ABBA_total=sum(v['valid_ABBA']for v in rows),all8_passed=len(good)==8,ranks=rows,all8_all5_same_direction=all(v['all5_same_direction']for v in good)if len(good)==8 else False,all8_all5_gain_exceeds_drift=all(v['all5_gain_exceeds_drift']for v in good)if len(good)==8 else False,Model_gain='UNMEASURED',quality='Component finite, independent CPU full-chain numerical, immutable state/input/tail checks only.',timing_limits='Qualification events bracket complete synchronous producer-stream CAPI including host launch gaps; no events in ABBA. Summed per-rank CAPI intervals are not GPUbusy or cross-rank wall. Startup/qualification, loop, CAPI, allocation elapsed separated; memory checkpoints are not peak.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();src=pathlib.Path(a.archive);dst=pathlib.Path(a.output);assert not dst.exists()
    q=analyze(json.loads(src.read_bytes()));q['archive_SHA256']=hashlib.sha256(src.read_bytes()).hexdigest();dst.write_text(json.dumps(q,indent=2)+'\n',newline='\n');print(json.dumps({k:q[k]for k in ['GPU','valid_ranks','valid_ABBA_total','all8_all5_same_direction','all8_all5_gain_exceeds_drift']}))
