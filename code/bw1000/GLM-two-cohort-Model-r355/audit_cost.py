"""Offline actual r355 cost audit. All times remain on their original clock."""
import argparse,hashlib,json,statistics
from pathlib import Path


def percentile(values,p):
    if not values:return None
    v=sorted(values);x=(len(v)-1)*p;lo=int(x);hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(x-lo)


def emitter_metrics(pair):
    result=[];all_ticks=[];admissions=[];useful=0
    for q in pair['requests']:
        ids=q['generated_ids'];ticks=q['emitter_token_monotonic'];admission=q['emitter_admission_monotonic']
        assert len(ids)==len(ticks)and len(ids)>0 and ticks==sorted(ticks)
        visible=[t for token,t in zip(ids,ticks)if token not in [154820,154827,154829]]
        assert visible and visible[0]>=admission
        itl=[b-a for a,b in zip(visible,visible[1:])]
        result.append(dict(row=q['row'],fixture_id=q['fixture_id'],useful_tokens=len(visible),
            TTFT_s=visible[0]-admission,user_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,
            ITL_P50_s=percentile(itl,.5),ITL_P95_s=percentile(itl,.95),ITL_P99_s=percentile(itl,.99),ITL_max_s=max(itl)if itl else None,
            natural_EOS=q['natural_EOS'],includes_serial_other_row_prefill_wait=True))
        useful+=len(visible);all_ticks.extend(visible);admissions.append(admission)
    elapsed=max(all_ticks)-min(admissions)
    return dict(requests=result,useful_output_tokens=useful,emitter_admission_to_last_visible_s=elapsed,
                emitter_total_tok_s=useful/elapsed,emitter_total_tok_s_per_32cards=useful/elapsed/32,
                clock='Same emitter process monotonic, actual admission and token emissions; not HTTP incremental or onlineSLO')


def main():
    p=argparse.ArgumentParser();p.add_argument('--owner-jsons',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    paths=[Path(x)for x in a.owner_jsons];owners=[json.loads(p.read_bytes())for p in paths]
    assert len(owners)==32 and {(q['stage'],q['rank'])for q in owners}=={(s,r)for s in range(4)for r in range(8)}
    assert all(q['completed']and q['worker_shutdown_completed']and q['normal_reset_close']for q in owners)
    assert all(len(q['pairs'])==6 for q in owners)
    aligned=[]
    for i in range(6):
        ranked=[(q['pairs'][i]['complete_schedule_wall_s'],q['stage'],q['rank'])for q in owners]
        worst=max(ranked);dec=max(q['pairs'][i]['decode_schedule_wall_s']for q in owners)
        aligned.append(dict(index=i,enabled=i in [1,3,4],warmup=i<2,max32_complete_schedule_wall_s=worst[0],
                            argmax_stage=worst[1],argmax_rank=worst[2],max32_decode_schedule_wall_s=dec,
                            clock='Max local elapsed durations at same trial index; no cross-host absolute subtraction'))
    off=statistics.mean(aligned[i]['max32_complete_schedule_wall_s']for i in [2,5])
    on=statistics.mean(aligned[i]['max32_complete_schedule_wall_s']for i in [3,4])
    drift=abs(aligned[5]['max32_complete_schedule_wall_s']/aligned[2]['max32_complete_schedule_wall_s']-1)
    tail=next(q for q in owners if(q['stage'],q['rank'])==(3,0))
    q=dict(completed=True,scope='One static two-request six-pair Model pilot; no formal benchmark/onlineSLO',
           all_service_cards=32,aligned_trials=aligned,measured_OFF_s=off,measured_ON_s=on,
           complete_schedule_rate_gain_fraction=off/on-1,measured_OFF_endpoint_drift_fraction=drift,
           emitter=[dict(index=i,**emitter_metrics(x))for i,x in enumerate(tail['pairs'])],
           old719_gain_inherited=False,local_native_CAPI_enclosures_not_pure_GPU_work=True,
           evidence=[dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())for p in paths],
           promotion='Not automatic; root must inspect ordinary paired gain versus drift, natural EOS/independent gold and normal32 lifecycle')
    Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
