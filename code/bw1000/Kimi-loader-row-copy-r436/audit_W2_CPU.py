"""Original CPU878 exact W2-geometry receipt and four-ABBA arithmetic."""
import argparse,hashlib,json,math,statistics
from pathlib import Path


def audit(path):
    raw=Path(path).read_bytes();q=json.loads(raw)
    v=q['files']['Kimi-loader-row-copy-r436-W2-CPU-v3'];data=v['text'].encode()
    assert len(data)==v['bytes']and hashlib.sha256(data).hexdigest()==v['SHA256']
    r=json.loads(data)
    assert r['completed']and r['passed']and r['actual_CPU']=='910407.878'and not r['new_GPU']
    assert r['source_manifest_SHA256']=='8b550dc925cbd7ee0e713c2511ad0afff01bd5c22eec4a55c038617d00fa975a'
    assert len(r['actual_cpu_affinity'])==1 and not r['project_Native_maps']
    assert all(x=='-1'for x in r['actual_GPU_visibility'].values())
    assert r['semantic']['passed']and r['semantic']['cases']==32
    account=[a for a in q['accounting']if a[0]=='910407.878']
    assert len(account)==1 and account[0][1:4]==['Kimi-r436-W2-row-copy-CPU-v3','COMPLETED','0:0']
    cases=r['paired']['cases'];assert r['paired']['passed']and len(cases)==2
    results=[]
    for c in cases:
        width=c['block_bytes'];assert width in [96,6]and c['rows']==3584
        assert c['source_row_stride_bytes']==(1536 if width==96 else 96)
        assert c['logical_copy_bytes_per_call']==3584*width
        a=c['schedule'];assert [v['mode']for v in a]==[0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0]
        assert all(v['calls']==32 and math.isfinite(v['seconds'])and v['seconds']>0 for v in a)
        blocks=[]
        for i in range(2,18,4):
            off=[a[i]['seconds'],a[i+3]['seconds']];on=[a[i+1]['seconds'],a[i+2]['seconds']]
            om,nm=statistics.mean(off),statistics.mean(on);drift=off[-1]/off[0]-1
            blocks.append(dict(OFF_mean_s=om,ON_mean_s=nm,speedup=om/nm,
                OFF_endpoint_signed_drift=drift,mean_gain_exceeds_endpoint_abs_drift=1-nm/om>abs(drift)))
        om=statistics.mean(v['seconds']for v in a[2:]if v['mode']==0)
        nm=statistics.mean(v['seconds']for v in a[2:]if v['mode']==1)
        results.append(dict(row_bytes=width,rows=3584,source_row_stride_bytes=c['source_row_stride_bytes'],
            OFF_mean_full_copy_s=om/32,ON_mean_full_copy_s=nm/32,speedup=om/nm,
            four_ABBA_blocks=blocks,all_four_blocks_mean_gain_exceeds_drift=all(v['mean_gain_exceeds_endpoint_abs_drift']for v in blocks)))
    return dict(passed=True,actual_CPU='910407.878',accounting=account[0],
        original_observation_SHA256=hashlib.sha256(raw).hexdigest(),original_receipt_SHA256=v['SHA256'],
        compiler=r['compiler'],actual_CPU_thread_affinity=r['actual_cpu_affinity'],
        original_elapsed_s=r['seconds'],new_geometry_guard_cases=32,old243_suite_replayed=False,
        actual_CPU_visibility=r['actual_GPU_visibility'],cases=results,
        scope='Exact archived W2 uint8 geometry/strides, synthetic byte payloads/warm RAM, standalone target CPU; not an actual Native loader or cold checkpoint/pagefault/H2D replay.',
        actual_model_loading_benefit='UNKNOWN',actual_Model_decode_benefit='NOT_MEASURED',
        Runtime_installed_or_replaced=False,Native_dlopen=False,GPU_GO=False,Model_GO=False,
        decision='Retain private positive exact-geometry CPU candidate. Next at most minimal actual Runtime caller/source binding feasibility; no shared-library replacement or Model reload from this result.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=audit(a.original)
    with a.output.open('x',encoding='utf8')as f:json.dump(q,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'CPU':q['actual_CPU'],'speedup':[v['speedup']for v in q['cases']]}))
