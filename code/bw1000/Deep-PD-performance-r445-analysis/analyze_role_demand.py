"""Conditional service-demand comparison; never a physical hardware bound."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from analyze_schedule import audit,SEAL

def read(path):
    schedule=audit(path)  # Verify original bytes and all formal client intervals.
    seal=json.loads(Path(path).read_bytes())
    member=next(f for f in seal['files']if Path(f['path']).name=='Deep382-pd445-20261104-a-client.json')
    client=json.loads(gzip.decompress(base64.b64decode(member['gzip_base64'],validate=True)))
    cases=[]
    for pdgroup,cogroup in [(5,4),(6,7)]:
        rows=[r for r in client['requests']if r['request']['group']==pdgroup]
        p=next(g for g in schedule['groups']if g['id']==pdgroup)
        co=next(g for g in schedule['groups']if g['id']==cogroup)
        assert p['fixture_order']==co['fixture_order']
        demands=[]
        for row in rows:
            r=row['D_report'];n=r['Native_forward_summary']
            assert r['operation']=='import_decode'and n['prefill_calls']==n['prefill_input_tokens']==0
            assert r['normal_reset_before']and r['normal_reset_after']and r['completed']
            assert r['natural_EOS']and r['actual_Native']and r['all8_import_publish_contract_completed']
            assert n['decode_calls']==len(r['generated_token_ids'])-1
            demands.append(dict(index=row['request']['index'],decode_calls=n['decode_calls'],Native_decode_host_enclosure_s=n['decode_wall_s']))
        demand=sum(r['Native_decode_host_enclosure_s']for r in demands)
        cases.append(dict(PD_group=pdgroup,co_group=cogroup,order=p['fixture_order'],D_sequential_Native_decode_host_enclosure_sum_s=demand,co_complete_group_wall_s=co['group_wall_s'],D_observed_decode_demand_over_co_complete_wall=demand/co['group_wall_s'],P_ready_queue_hold_s=p['P_ready_hold_for_D_s'],details=demands))
    return dict(scope='Two formal originalC4 groups, measured D sequential service demand; not hardware maximum or predicted performance',original_seal_SHA256=SEAL,cases=cases,conclusion='With the observed D per-request decode costs held fixed, even zero P/transfer/queue time cannot beat the measured co group completion. Removing P backpressure alone is insufficient for this frozen workload; a future candidate must also reduce D service demand or change its serving capacity.',limits=['The counterfactual explicitly holds observed decode costs fixed. Scheduling changes can change overlap, clocks, contention and decode costs, so this is not a universal latency lower bound.','Native host enclosures include copies, CPU and communication; no GPU-busy or hardware-peak claim.','Co one gold error remains a quality failure; no quality superiority or general noninferiority inference.','No new model execution or new performance round.'],new_GPU=0,new_performance_round=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--output',required=True);a=p.parse_args();q=read(a.seal)
    Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':[{k:v for k,v in c.items()if k!='details'}for c in q['cases']],'conclusion':q['conclusion']}))
