"""Independent root consumption of the first parallel KDA search, CPU only."""
from pathlib import Path
import json
import hashlib
import statistics as stat
import math
import datetime


def audit(root):
    base=root/'results/bw1000/2026-11-02'
    paths=[base/n for n in ['Kimi339-GPU606-607-raw.json','Kimi339-GPU606-607-post.json','Kimi339-process-post.json']]
    raw,post,proc=[json.loads(p.read_bytes())for p in paths]
    values=list(raw['results'].values())
    assert len(values)==16
    assert {(v['host'],v['dev'])for v in values}=={(h,i)for h in ['a02r3n03','a02r3n07']for i in range(8)}
    records=[]
    for v in values:
        assert v['arm']==v['dev']+1 and v['seed']==(339 if v['host']=='a02r3n03' else 340)
        assert v['shape']==dict(C=1,T=1,H=6,D=128,pool=3,read=1,write=2)
        assert v['source_commit']=='92cbaf228f433ec099c086bcd7414ae975449992'
        assert v['source_manifest_SHA256']=='8bf14e17c14a12e99bf1c2a5facd86bf1ec80e7c73bb08a739215c167a84b5e6'
        assert v['binary_SHA256']=='8feb14507414b3f2b3afbecd413a700a7b258d795405bbef556faf65a561f502'
        assert v['finite'] and v['CPU_F32_reference'] and v['canary_read_unchanged'] and len(v['rounds'])==32
        ratios=[];aa_all=[];bb_all=[]
        for i in range(8):
            rows=[x for x in v['rounds']if x['round']==i]
            expected=[0,v['arm'],v['arm'],0] if i%2==0 else [v['arm'],0,0,v['arm']]
            assert [x['arm']for x in rows]==expected
            aa=[x['wall_ms']for x in rows if x['arm']==0]
            bb=[x['wall_ms']for x in rows if x['arm']==v['arm']]
            assert all(math.isfinite(x)and x>0 for x in aa+bb)
            ratios.append(stat.mean(bb)/stat.mean(aa));aa_all.extend(aa);bb_all.extend(bb)
        records.append(dict(host=v['host'],GPU=v['dev'],seed=v['seed'],arm=v['arm'],
            baseline_wall_us=stat.mean(aa_all)*1000,candidate_wall_us=stat.mean(bb_all)*1000,
            paired_median_candidate_over_baseline=stat.median(ratios),all8_rounds_faster=all(x<1 for x in ratios),
            baseline_early_late_drift=aa_all[-1]/aa_all[0]-1,promotable_launch_bound=v['arm']not in [4,8],
            raw_measurement_s=v['elapsed_measure_s']))
    for actual,node in [('910371.606','a02r3n03'),('910371.607','a02r3n07')]:
        rows=[x.split('|')for x in raw['accounting'].splitlines()if x.startswith(actual+'|')]
        assert len(rows)==1 and rows[0][1:4]==['Kimi339-search','COMPLETED','0:0'] and rows[0][-1]==node
    assert {x['node']for x in post['post_KFD_empty']}=={'a02r3n03','a02r3n07'}
    assert all(not x['kfd_pids']for x in post['post_KFD_empty'])
    assert {x['node']for x in proc['process_post']}=={'a02r3n03','a02r3n07'}
    assert all(not x['own_bench_pids']for x in proc['process_post'])
    best=[x for x in records if x['arm']==2]
    assert len(best)==2 and all(x['all8_rounds_faster']for x in best)
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='ROOT_AUDITED_STANDALONE_KDA_POSITIVE_COMPLETE_CLEAN_NOT_DELTA_MODEL',
        GPU_steps=['910371.606 COMPLETED0/41s','910371.607 COMPLETED0/40s'],
        post_steps=['910371.608','910371.609'],CPU_reference_finite_state_canary_all16=True,
        max_output_abs_error=max(v['numerics']['output']['max_abs']for v in values),
        max_state_abs_error=max(v['numerics']['state']['max_abs']for v in values),rows=records,
        selected_arm=2,best_two_seeds=best,invalid_for_promotion_arms=[4,8],
        invalid_reason='DTK default launch bound256 versus W8block512 warning; numerical agreement does not prove a supported launch',
        scope='Synthetic C1H6D128 indexed KDA component. Original baseline includes fullpool D2D snapshot; candidate legally omits it only for disjoint read1/write2. Cannot attribute all gain solely to arithmetic rewrite.',
        limits=['About0.09s warmed direct measurement perGPU; startup separate',
                'No real checkpoint activation, TP communication, fullDelta, Model or client SLO proof',
                'No-snapshot original baseline ablation still needed to isolate arithmetic contribution'],
        evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},
        next_task='coordination/tasks/Kimi-KDA-fused-Delta-r343.json',new_Model_speedup=False)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    p=root/'results/bw1000/2026-11-02/Kimi339-root-independent-audit.json'
    with p.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(all16_validated=True,post_clean=True,best=result['best_two_seeds'])))
