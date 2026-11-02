"""Stdlib-only original six-file reader. No jobs, imports of runtime or Model.

Input is the exact gzip/base64 scalar archive in L; every original byte hash is
verified before any assertion or reduction. Histograms are HOST branch evidence.
"""
from pathlib import Path
import argparse,base64,collections,gzip,hashlib,json,math,re,statistics

NATIVE='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
SOURCE='e9d2e7ab60c231c5ded9ff0003c146b01babd007129520ec8f4cc954014df7d1'
WORKER='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31'
GPU_MF='6312654c564fdc46477d6be31e16da36540322c1003d44145ea1130aa5fea537'
SCHEDULE=[0,1,0,1,1,0]
OLD_IDS='804aa5c53ba4a6d1d75242689186aa0ed67fc95f016a3cbda21a5d3d6cf97279'
OLD_DECODE='6c00be310acdf3d97da2fd109bf3d078016364185836e26892f894ef35084725'
sha=lambda b:hashlib.sha256(b).hexdigest()

def percentile(xs,p):
    xs=sorted(xs);n=(len(xs)-1)*p;i=int(n)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(n-i)

def mean(xs):return statistics.fmean(xs)

def emitter(row):
    eos={163585,163586};ids=row['generated_ids'];flat=[i for e in row['emitter_events'] for i in e['tokens']]
    assert flat==ids
    ticks=[e['at_s'] for e in row['emitter_events'] for i in e['tokens'] if i not in eos]
    assert len(ticks)==58 and ticks==sorted(ticks) and all(math.isfinite(t) and t>=0 for t in ticks)
    gaps=[b-a for a,b in zip(ticks,ticks[1:])];span=ticks[-1]-ticks[0];rate=57/span
    assert abs(row['TTFT_s']-ticks[0])<1e-9 and abs(row['backend_decode_tok_s']-rate)<1e-9
    return dict(visible_tokens=58,TTFT_s=ticks[0],first_last_s=span,backend_decode_tok_s=rate,
                ITL_P50_s=percentile(gaps,.5),ITL_P95_s=percentile(gaps,.95),ITL_P99_s=percentile(gaps,.99),ITL_max_s=max(gaps),
                actual_visible_ticks_s=ticks,ITL_s=gaps)

def counter_audit(row,node):
    count=row['I192'];calls=count['calls'];assert count['node']==node and count['GROUPED']==row['GROUPED'] and count['MOE_EXECUTION']==1
    assert count['fallback_counter_available'] is False and count['no_fixed_layer_count_assumption'] is True
    forward_calls=[c for c in calls if c['operation']!='replay']
    assert len(forward_calls)==len(row['Native_calls'])
    for c,n in zip(forward_calls,row['Native_calls']):
        assert c['T']==sum(n['width'])
        cp=c['past'] if isinstance(c['past'],list) else [c['past']]
        assert cp==n['past']
    totals=[[0]*65 for _ in range(8)];widths=collections.Counter();selected=[]
    for call in calls:
        T=call['T'];widths[T]+=1
        assert call['completed'] and call['node']==node and call['PP_stage']==node//2 and call['TP_node_rank']==node%2
        assert call['GROUPED']==row['GROUPED'] and call['ordinal']==row['ordinal']
        hist=call['T_hist_delta_by_local_owner'];assert len(hist)==8 and all(len(x)==65 for x in hist)
        assert all(type(v)is int and v>=0 for x in hist for v in x)
        assert call['selected_HOST_branches_by_local_owner']==[sum(x) for x in hist]
        if row['GROUPED'] and 8<=T<=64:
            assert all(x[T]>0 and all(v==0 for t,v in enumerate(x) if t!=T) for x in hist)
            selected.append(dict(T=T,past=call['past'],by_local_owner=[x[T] for x in hist]))
        else:assert all(not any(x) for x in hist)
        for d in range(8):totals[d]=[a+b for a,b in zip(totals[d],hist[d])]
    assert totals==count['T_hist_delta_by_local_owner']
    assert [x[8] for x in totals]==count['T8_HOST_selected_by_local_owner']
    assert all(x[1]==0 for x in totals)
    assert widths[512]==16 and widths[443]==1 and widths[8]==8
    assert widths[1]==0, 'Current fixture has no actual W1 call; do not pretend it was measured'
    return dict(node=node,PP_stage=node//2,TP_node_rank=node%2,local_owners=8,global_owner_ids=list(range(node*8,node*8+8)),
                actual_call_T_counts=dict(widths),T8_selected_by_local_owner=[x[8] for x in totals],
                selected_call_details=selected,ineligible_actual_512_443_selected_zero=True,
                T1_bucket_zero=True,actual_T1_calls=0,T1_ineligibility_scope='Source predicate and zero bucket, not an observed W1 call',
                all_65_buckets_checked=True,fallback_counter_exists=False,
                meaning='Native HOST branch selected counts, no GPU hardware events; per-PP differences retained')

def elapsed(text):
    parts=list(map(int,text.split(':')));assert len(parts)==3
    return parts[0]*3600+parts[1]*60+parts[2]

def read(path):
    archive=json.loads(Path(path).read_bytes());assert archive['source_manifest_SHA256']==SOURCE and archive['GPU_manifest_SHA256']==GPU_MF
    assert archive['source_32_members_unchanged'] is True
    records=[];refs=[]
    for item in archive['files']:
        raw=gzip.decompress(base64.b64decode(item['payload']));assert len(raw)==item['bytes'] and sha(raw)==item['SHA256']
        records.append(json.loads(raw));refs.append({k:item[k] for k in ['path','bytes','SHA256']})
    assert len(records)==6
    watch=records[0];leaders=records[1:5];draft=records[5]
    assert watch['completed'] and watch['passed'] and watch['post_empty']
    assert watch['actual_target']=='910380.556' and watch['actual_draft']=='910371.719'
    expected_refs={r['path']:r['SHA256'] for r in watch['final_evidence']['target_reports']}
    expected_refs[watch['final_evidence']['draft']['path']]=watch['final_evidence']['draft']['SHA256']
    assert expected_refs=={r['path']:r['SHA256'] for r in refs[1:]}
    assert not watch.get('owned_forced_cleanup') and not watch.get('cleanup_error_type')
    for key in ['target_observation','draft_observation']:
        o=watch[key];assert o['row'][2]=='COMPLETED' and o['row'][3]=='0:0' and o['srun_exit_code']==0
    assert len(watch['post'])==5 and {p['node'] for p in watch['post']}=={'a14r2n08','a14r2n09','a14r2n10','a14r2n12','a02r3n15'}
    for p in watch['post']:
        assert p['passed'] and p['KFD_empty'] and p['kfd_pids']==[] and p['owned_worker_PIDs']==[] and all(p['ports_bindable'].values())
        assert set(p['ports_bindable'])=={'32947','39947','45947','46947'}
        assert p['TCP_connections_made']==0
        assert all(e['state']=='06' for sample in p['passive_TIME_WAIT'] for e in sample['entries'])
    assert {q['node'] for q in leaders}==set(range(4))
    leaders.sort(key=lambda q:q['node'])
    fixture=archive['fixture'];assert fixture['input_tokens']==8635 and fixture['prompt_ID_SHA256']==fixture['recomputed_prompt_ID_SHA256']
    payload=json.loads(next(m['content'] for m in fixture['body']['messages'] if m['role']=='tool'))
    values={r['id']:r['value'] for r in payload['records']}
    gold=dict(zip(['a','b'],[values[payload['query_ids'][i]] for i in range(2)]))
    assert gold==fixture['expected_calculate_arguments']=={'a':16195,'b':39747}
    decodes={d['ordinal']:d for d in archive['official_rank_decodes']}
    signatures=[];histograms=[];all_rows=[]
    for q in leaders:
        assert q['completed'] and q['actual_step']=='910380.556' and q['Native_SHA256']==NATIVE and q['cleanup_errors']==[]
        assert all(q[k] for k in ['all8_owned_workers_shutdown_completed','capture_CPU_cleanup_completed','sampler_CPU_cleanup_completed','control_CPU_cleanup_completed'])
        assert q['actual_CAPI_providers']['modern_kimi_I192_T_counts_r9129']=='/refactor/candidates/Kimi-long-capture-link-CPU-r327/libKimi_long_capture_r327.so'
        assert q['actual_loaded_libraries']['libinfiniops.so']['SHA256']=='6d5a6204f7eef70e84deb9cc737a36c6ae2c1fbc75a0f70645557c8441ed85ca'
        first=q['first_capture_validation'];assert first['passed'] and first['finite'] and first['layerIDs']==[19,37,66,78,90] and first['rows']==512 and first['routed_leaders']==[0,2]
        pair=q['pair_report'];assert pair['completed'] and pair['normal_draft_protocol_close'] and len(pair['requests'])==6
        signature=[];rows=[]
        for i,r in enumerate(pair['requests']):
            assert r['ordinal']==i and r['GROUPED']==SCHEDULE[i] and r['warm']==(i<2)
            assert r['mode']=='spec' and r['name']=='retained8635' and r['prompt_SHA256']==fixture['prompt_ID_SHA256']
            ids=r['generated_ids'];assert len(ids)==r['tokens']==59 and ids[-1] in [163585,163586] and not set(ids[:-1])&{163585,163586}
            assert r['finish']=='eos' and r['completed'] and r['normal_reset'] and r['quality']['passed']
            assert r['quality']['details']['arguments']==gold
            assert sha(json.dumps(ids,separators=(',',':')).encode())==r['ids_SHA256']==OLD_IDS
            decode=decodes[i];assert decode['ids_SHA256']==OLD_IDS and sha(decode['decoded'].encode())==OLD_DECODE
            assert re.findall(r'call tool="([^"]+)"',decode['decoded'])==['calculate']
            parsed={key:int(value) for key,value in re.findall(r'argument key="([ab])" type="number"<\|sep\|>(-?\d+)<\|close\|>argument',decode['decoded'])}
            assert parsed==gold
            signature.append((r['prompt_SHA256'],r['ids_SHA256'],r['tokens'],r['finish']))
            h=counter_audit(r,q['node']);histograms.append(dict(ordinal=i,warm=i<2,GROUPED=r['GROUPED'],**h))
            m=emitter(r);m.update(node=q['node'],ordinal=i,GROUPED=r['GROUPED'],warm=i<2,
                                generation_s=r['generation_local_wall_s'],through_reset_s=r['request_through_reset_local_wall_s'])
            rows.append(m)
        signatures.append(signature);all_rows.append(rows)
    assert all(s==signatures[0] for s in signatures)
    assert draft['completed'] and draft['passed'] and draft['normal_protocol_close'] and draft['actual_step']=='910371.719'
    assert draft['authenticated'] and draft['ready'] and draft['source_manifest_SHA256']==WORKER and draft['GPU_count']==1
    assert draft['runtime_manifest_SHA256']=='358016b6b9e334817d2d4929ed613d67548dcd6764a20f2372b7748be3c99a2e'
    assert draft['nonce_SHA256']==watch['nonce_SHA256']
    expected=dict(begin=6,append_prompt=102,seal=6,query=48,append_committed=42,retire=6,close=1)
    assert draft['phase_counts']==expected and draft['commands']==sum(expected.values())==211
    assert draft['feature_bytes']==6*(8635+7*8)*35840*2
    assert [f['index'] for f in draft['memory_fences']]==list(range(len(draft['memory_fences'])))
    assert all(f['floor_passed'] and f['floor_bytes']==4<<30 and f['free_after_bytes']>=4<<30 for f in draft['memory_fences'])
    arms=[]
    for i,r in enumerate(leaders[0]['pair_report']['requests']):
        costs=collections.defaultdict(float)
        for c in r['publisher_costs']:costs[c['operation']]+=c['local_wall_s']
        cycles=r['cycles'];proposed=sum(sum(c['width'])-len(c['width']) for c in r['Native_calls'] if c['kind']=='verify')
        counts={k:sum(c[k] for c in cycles) for k in ['raw_matched_proposals','accepted_visible_proposals','useful_emitted_tokens','emitted_tokens','committed_inputs']}
        assert counts['useful_emitted_tokens']+1==58 and counts['emitted_tokens']+1==59
        assert counts['accepted_visible_proposals']<=proposed and counts['raw_matched_proposals']<=proposed
        native=collections.defaultdict(float)
        for c in r['Native_calls']:
            native[c['kind']+'_CAPI_maxhost_s']+=c['CAPI_maxhost_s']
            native[c['kind']+'_feature_harvest_local_s']+=c['feature_harvest_local_s']
        arm=dict(ordinal=i,GROUPED=SCHEDULE[i],warm=i<2,node0=all_rows[0][i],Native_emitter_node2=all_rows[2][i],
                 max4leader_generation_s=max(rows[i]['generation_s'] for rows in all_rows),
                 max4leader_through_reset_s=max(rows[i]['through_reset_s'] for rows in all_rows),
                 publisher_inclusive_phase_s=dict(costs),Native_phase_s=dict(native),
                 phase_scope='Nested/inclusive: query includes worker+RPC; commit_replay_publish contains append_committed/retire. Do not sum overlapping phase series into a second fullwall.',
                 actual_cycles=len(cycles),proposal_count=proposed,cycle_counts=counts,
                 accepted_useful_rate=counts['accepted_visible_proposals']/proposed,
                 output_IDS_SHA256=r['ids_SHA256'],visible_output=58,natural_EOS=True,gold=gold)
        arm['full_generation_visible_tok_per_card_s']=58/arm['max4leader_generation_s']/33
        arms.append(arm)
    off=[arms[i] for i in [2,5]];on=[arms[i] for i in [3,4]]
    def compare(field,extract):
        a=[extract(r) for r in off];b=[extract(r) for r in on]
        return dict(OFF=a,ON=b,ON_over_OFF_mean=mean(b)/mean(a),OFF_endpoint_ratio=a[1]/a[0],
                    adjacent_ON_over_OFF=[b[0]/a[0],b[1]/a[1]],scope=field)
    comparison=dict(node0_decode=compare('node0 selected-ID callback (n-1)/(last-first)',lambda a:a['node0']['backend_decode_tok_s']),
                    emitter2_decode=compare('Native producer node2 local selected-ID callback; same-node first-last',lambda a:a['Native_emitter_node2']['backend_decode_tok_s']),
                    max4_full_generation=compare('max of4local fullgeneration durations; never subtract crossnodeclocks',lambda a:a['max4leader_generation_s']),
                    max4_through_reset=compare('max of4local throughreset durations',lambda a:a['max4leader_through_reset_s']),
                    verify_CAPI=compare('sum actualverify percall fourleadermax CAPI; subset of fullgeneration',lambda a:a['Native_phase_s']['verify_CAPI_maxhost_s']),
                    full_generation_per_card=compare('58visible / max4fullgeneration /33servicecards',lambda a:a['full_generation_visible_tok_per_card_s']))
    accounting=[x.split('|') for x in archive['accounting']]
    byid={r[0]:r for r in accounting}
    for sid in ['910380.556','910371.719','910407.779','910380.557','910371.721']:
        assert byid[sid][2:4]==['COMPLETED','0:0']
    target_s=elapsed(byid['910380.556'][4]);draft_s=elapsed(byid['910371.719'][4])
    total_card_s=32*target_s+draft_s
    formal_ids=sum(arms[i]['visible_output'] for i in [2,3,4,5]);all_ids=sum(a['visible_output'] for a in arms)
    fullratio=comparison['max4_full_generation']['ON_over_OFF_mean']
    drift=abs(comparison['max4_full_generation']['OFF_endpoint_ratio']-1)
    assert abs(fullratio-1)<drift
    return dict(passed=True,scope='Six same8635 real Spec Model engineering/quality scope;2warm excludedformal;not broadbenchmark/HTTP/SSE/onlineSLO',
                source_manifest_SHA256=SOURCE,GPU_manifest_SHA256=GPU_MF,original_files=refs,
                full_lifecycle='PASS actual556/719/779 normal;32ownedworkers and sole draft close;no forcedcleanup;fivepostclean',
                counter_audit=histograms,all6_requests=arms,formal_indices=dict(OFF=[2,5],ON=[3,4],excluded_warm=[0,1]),formal_comparison=comparison,
                all4_leader_callback_timelines=all_rows,first_capture_note='Inherited helper arm_ordinal2 is legacy static label. Actual newfirstSpec ordinal0 first512/epoch2 validated; no fifthtap mathematical change.',
                finite_scope='Actual completed source-bound335/Native flags/BF16 finite/capture guards; raw logits/features not saved, so no independent raw numeric rerun claimed.',
                semantic_evidence=dict(IDS_SHA256=OLD_IDS,official_decoded_text_SHA256=OLD_DECODE,tokenizer=archive['tokenizer_identity'],
                    actual_official_rank_decode=True,independent_records_lookup_gold=gold,all6_equal_old_rund_IDS_SHA256=True,
                    reuse='Original run-d node0 db44b895... was official-BPE semantically root-audited; current complete59IDs hashrecomputed identical. Also current officialrankdecode and parser/gold checked. Token equality is diagnostic, not a quality prerequisite.'),
                draft=dict(actual_step=draft['actual_step'],phase_counts=expected,commands=draft['commands'],feature_bytes=draft['feature_bytes'],
                    memory_fences=len(draft['memory_fences']),trim_count=sum(f['trimmed'] for f in draft['memory_fences']),
                    min_free_bytes=min(f['free_after_bytes'] for f in draft['memory_fences']),peak_allocated_bytes=draft['peak_allocated_draft_bytes'],peak_reserved_bytes=draft['peak_reserved_draft_bytes'],
                    load_s=draft['load_wall_s'],one_time_node0_connect_load_before_request_s=leaders[0]['pair_report']['draft_connect_load_handshake_local_s']),
                residency_cost=dict(target_cards=32,target_elapsed_s=target_s,draft_cards=1,draft_elapsed_s=draft_s,total_GPU_card_s=total_card_s,
                    all6_visible_tokens=all_ids,formal_visible_tokens=formal_ids,all_including_warm_visible_tok_per_card_s=all_ids/total_card_s,
                    formal_visible_tok_per_entire_paid_residency_card_s=formal_ids/total_card_s,
                    note='Includes coldtargetload/draftinit/warm/idle until process exit;allocationseconds resolution. No amortized wide-service capacity claim.'),
                per_leader_load=[dict(node=q['node'],create_s=q.get('create_seconds'),loading_elapsed_s=q.get('loading_elapsed_seconds'),loading_s=q.get('loading_seconds')) for q in leaders],
                assessment=dict(full_generation_gain_beyond_endpoint_drift=False,full_generation_mean_change_pct=100*(fullratio-1),
                                OFF_full_generation_endpoint_change_pct=100*drift,
                                node0_decode_mean_gain_pct=100*(comparison['node0_decode']['ON_over_OFF_mean']-1),
                                verify_CAPI_mean_reduction_pct=100*(1-comparison['verify_CAPI']['ON_over_OFF_mean']),
                                limited_samples='OneABBA/2formal perarm, fixed single8635 tool prompt; no statistical generalization or >15qualification.'),
                post=watch['post'],accounting=archive['accounting'],
                decision='PARK_FULL_REQUEST_PROMOTION: retain measured slight decode/verify gains and normal ON Model path qualification; fullrequest/cost has no gain beyondOFFdrift. No grid/reload or latest99/ec/widebenchmark qualification.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--sealed',type=Path,required=True);p.add_argument('--output',type=Path);args=p.parse_args()
    result=read(args.sealed);result['sealed_archive']=dict(path=str(args.sealed.resolve()),SHA256=sha(args.sealed.read_bytes()))
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf8',newline='\n')
    else:print(text,end='')

if __name__=='__main__':main()
