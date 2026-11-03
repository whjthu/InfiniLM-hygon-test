"""Offline original CPU874 receipt, semantic guards and formal ABBA arithmetic."""
import argparse,hashlib,json,math,statistics
from pathlib import Path


def audit(path):
    raw=Path(path).read_bytes();q=json.loads(raw)
    matches=[v for p,v in q['files'].items()if p.endswith('/result.json')]
    assert len(matches)==1;v=matches[0];b=v['text'].encode()
    assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['SHA256']
    r=json.loads(b)
    assert r['actual_CPU']=='910407.874'and r['completed']and r['passed']and r['new_GPU']==0
    assert r['actual_GPU_visibility']==dict(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
    assert not r['project_Native_maps']and len(r['actual_cpu_affinity'])==1
    assert r['source_manifest_SHA256']=='6b945936e8fdf281665403081727273a9e8fceb9628ef69655115ac5390dfc35'
    assert r['CPU_v2_source_manifest_SHA256']=='d0f11c1f9bf61b645154b82b6d1c403dd90f98be6ecbe1f7abd7f6866d38dd3d'
    assert r['semantic']['passed']and r['semantic']['cases']==243
    rows=q['accounting'];assert len(rows)==1 and rows[0][:4]==['910407.874','Kimi-r436-loader-row-copy-CPU-v2','COMPLETED','0:0']
    cases=r['paired']['cases'];assert r['paired']['passed']and len(cases)==3
    output=[]
    for c in cases:
        a=c['schedule'];assert [v['mode']for v in a]==[0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0]
        assert c['block_bytes']in [32,64,128]and c['rows']==32768 and c['logical_copy_bytes_per_call']==32768*c['block_bytes']
        assert all(v['calls']==32 and math.isfinite(v['seconds'])and v['seconds']>0 for v in a)
        blocks=[]
        for i in range(2,18,4):
            off=[a[i]['seconds'],a[i+3]['seconds']];on=[a[i+1]['seconds'],a[i+2]['seconds']]
            om,nm=statistics.mean(off),statistics.mean(on)
            blocks.append(dict(OFF_mean_s=om,ON_mean_s=nm,ON_over_OFF=nm/om,speedup=om/nm,
                endpoint_signed_drift=off[1]/off[0]-1,
                adjacent_ON_over_OFF=[on[0]/off[0],on[1]/off[1]],
                mean_gain_exceeds_endpoint_abs_drift=1-nm/om>abs(off[1]/off[0]-1)))
        off=[v['seconds']for v in a[2:]if v['mode']==0];on=[v['seconds']for v in a[2:]if v['mode']==1]
        om,nm=statistics.mean(off),statistics.mean(on)
        output.append(dict(block_bytes=c['block_bytes'],rows=c['rows'],actual_CPU_thread_count=1,
            samples_per_arm=8,logical_copy_calls_per_sample=32,
            OFF_mean_per_logical_copy_s=om/32,ON_mean_per_logical_copy_s=nm/32,
            ratio_of_means_ON_over_OFF=nm/om,speedup=om/nm,
            all_four_ABBA_mean_ON_faster=all(v['ON_over_OFF']<1 for v in blocks),
            all_four_mean_gain_exceeds_endpoint_abs_drift=all(v['mean_gain_exceeds_endpoint_abs_drift']for v in blocks),
            four_ABBA_blocks=blocks))
    return dict(passed=True,actual_CPU=r['actual_CPU'],actual_accounting=rows[0],
        original_receipt_SHA256=matches[0]['SHA256'],
        original_observation_SHA256=hashlib.sha256(raw).hexdigest(),
        actual_compiler=r['compiler'],actual_cpu_affinity=r['actual_cpu_affinity'],
        original_CPU_elapsed_s=r['seconds'],geometry_guards=243,cases=output,
        scope='Three synthetic positive row-stride CPU copy patterns; unchanged original helper vs private fixed-row candidate. Not Native loader replay, no real geometry histogram or exclusive Model copy time.',
        full_loading_speedup='UNKNOWN',Model_decode_speedup='NOT_MEASURED',GPU_GO=False,
        Runtime_installed_or_replaced=False,original873_precompile_failure_retained=True,
        decision='Only exact actual loader geometry/provider provenance investigation may follow; no HBM/Model speed or automatic reload prediction.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.original)
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'CPU':result['actual_CPU'],'speedups':[v['speedup']for v in result['cases']]}))
