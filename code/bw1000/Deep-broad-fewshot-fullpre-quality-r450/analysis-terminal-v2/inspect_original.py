"""Only existing original timing and seven paired outcomes; no jobs or capture."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,statistics
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
CORE='1a4554b081a12e46a73b2e61159e5d1d0c067ad85dded6b899c544f968240b25'
def core_module():
    p=D/'global_original.py';assert H(p.read_bytes())==CORE
    spec=importlib.util.spec_from_file_location('Deep450_terminal_core',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stats(values):
    a=sorted(float(v)for v in values if v is not None);assert all(math.isfinite(v)for v in a)
    def pct(p):
        if not a:return None
        x=(len(a)-1)*p;i=int(x);j=min(i+1,len(a)-1);return a[i]+(a[j]-a[i])*(x-i)
    return dict(n=len(a),minimum=min(a)if a else None,median=statistics.median(a)if a else None,mean=statistics.mean(a)if a else None,p95=pct(.95),p99=pct(.99),maximum=max(a)if a else None)
def analyze(manifest,private_dir=None):
    core=core_module();baseline=core.analyze(manifest,core.BASE_DEPLOYMENT);assert baseline['verified_requests']==512 and baseline['normal_lifecycle_qualified']and not baseline['all_historical_attempts_normal']
    b=core.Bundle(manifest);modules=core.consumers();cases=b.json('inputs/cases.json')['cases'];bycase={q['id']:q for q in cases};enc=b.json('inputs/encoding.json');encoding={q['case_id']:q for q in enc['cases']};assets=b.json('inputs/requests-private.json');byrequest={q['case_id']:q for q in assets['requests']};frozen={r['logical_index']:r for r in assets['logical_plan']}
    for case in cases:
        e=encoding[case['id']];p=b.json('inputs/private/'+case['id']+'.json');assert H(b.read('inputs/private/'+case['id']+'.json'))==e['private_SHA256']
        assert p['case_id']==case['id']and p['body']==byrequest[case['id']]['body']and H(C(p['body']))==e['request_body_SHA256']and H(p['prompt'].encode())==e['prompt_SHA256']
        ids=p['prompt_token_ids'];assert len(ids)==e['prompt_tokens']and H(C(ids))==e['prompt_IDs_SHA256']and all(type(t)is int and 0<=t<129280 for t in ids)
        assert e['shot_count']==case['shot_count']==(8 if case['suite']=='gsm8k'else 5)and e['actual_thinking']is False and e['text_delivery']=='P1'and e['output_budget']==4096 and len(ids)+4096<=32768
    indexed={};per_shard=[];native_provider='/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so';accounting={r[0]:r for r in b.json('accounting.json')}
    for j in range(8):
        key='j%d/'%j;owner=b.json(key+'owner.json');client=b.json(key+'client.json');watch=b.json(key+'watch.json');active=baseline['active_attempts'][j];rows=owner['requests'];assert len(rows)==64 and len({r['r450_logical_index']for r in rows})==64
        assert set(owner['actual_providers_before_weights'].values())=={native_provider}
        for handle,expected_node in[(active['actual_GPU'],active['node']),(active['actual_client'],active['node']),(active['CPU_keeper'],'a14r2n20')]:assert accounting[handle][2:4]==['COMPLETED','0:0']and accounting[handle][-1]==expected_node
        total=0
        for recorded,row in zip(client['rows'],rows):
            ix=recorded['logical_index'];entry=frozen[ix];assert row['r450_logical_index']==ix and entry['mode']==recorded['mode']==row['r450_mode'];proof=modules['coverage'].validate(row,entry)
            g=row['generated_token_ids'];ticks=row['backend_token_monotonic'];summary=row['Native_forward_host_summary'];assert summary['prefill_calls']==proof['prefill_forwards']and summary['decode_calls']==proof['decode_forwards']and summary['prefill_calls']+summary['decode_calls']==row['actual_forward_completed']
            assert all(type(summary[k])in(int,float)and math.isfinite(summary[k])and summary[k]>=0 for k in['prefill_s','decode_s'])
            visible=[t for t,token in zip(ticks,g)if token!=1];deltas=[v-u for u,v in zip(visible,visible[1:])];assert all(v>0 for v in deltas)
            assert row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['ended_monotonic']
            metric=dict(shard=j,logical_index=ix,case_id=entry['case_id'],suite=bycase[entry['case_id']]['suite'],mode=entry['mode'],generated_including_EOS=len(g),visible_generated=len(visible),decode_calls=summary['decode_calls'],prefill_calls=summary['prefill_calls'],Native_prefill_CAPI_s=summary['prefill_s'],Native_decode_CAPI_s=summary['decode_s'],Native_prefill_decode_CAPI_s=summary['prefill_s']+summary['decode_s'],Native_mean_decode_CAPI_s=summary['decode_s']/summary['decode_calls']if summary['decode_calls']else None,Native_owner_whole_s=row['ended_monotonic']-row['started_monotonic'],backend_owner_TTFT_s=ticks[0]-row['started_monotonic'],backend_visible_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,backend_max_visible_ITL_s=max(deltas)if deltas else None,client_full_wire_wall_s=recorded['client_wall_s'],no_eligible_T1=proof['no_eligible_T1'])
            assert all(v is None or math.isfinite(v)for k,v in metric.items()if k.endswith('_s')or k.endswith('_tps'))
            stem=key+'%04d'%ix;assert ix not in indexed;indexed[ix]=dict(recorded=recorded,owner=row,entry=entry,metrics=metric,proof=proof,wire_name=stem+'.wire',body_name=stem+'.request.json')
            total+=row['actual_forward_completed']
        assert owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==total
        post=watch['post'][0];assert post['CPU_only']and not post['GPU_initialized']and post['KFD_available']is True and post['TCP_connections_made']==0 and len(set(post['physical_PCI']))==8
        per_shard.append(dict(shard=j,Native_CAPI_calls=total,owner_count=8,observed8threads_exited=owner['observed_all8_threads_exited'],post_step=post['actual_step'],node=active['node'],loading_seconds=owner['loading_seconds'],original_GPU_start_unix=active['original_GPU_start_unix'],source_manifest_SHA256=active['deployment_manifest_SHA256']))
    assert set(indexed)==set(range(512));by_pair={};discordants=[];wrong=[];timing_pairs=[];private=Path(private_dir)if private_dir else None
    if private is not None:private.mkdir(exist_ok=False)
    for case in cases:
        pair={mode:next(v for v in indexed.values()if v['entry']['case_id']==case['id']and v['entry']['mode']==mode)for mode in(0,1)};a,z=pair[0],pair[1]
        assert a['entry']['request_body_SHA256']==z['entry']['request_body_SHA256']and a['entry']['prompt_IDs_SHA256']==z['entry']['prompt_IDs_SHA256']and b.read(a['body_name'])==b.read(z['body_name'])
        ids0,ids1=a['owner']['generated_token_ids'],z['owner']['generated_token_ids'];first=next((i for i,(x,y)in enumerate(zip(ids0,ids1))if x!=y),min(len(ids0),len(ids1))if ids0!=ids1 else None)
        timing_pairs.append(dict(case_id=case['id'],suite=case['suite'],same_generated_ID_array=ids0==ids1,output_tokens_OFF=len(ids0),output_tokens_ON=len(ids1),output_tokens_ON_minus_OFF=len(ids1)-len(ids0),OFF=a['metrics'],ON=z['metrics'],Native_owner_whole_ON_over_OFF=z['metrics']['Native_owner_whole_s']/a['metrics']['Native_owner_whole_s'],Native_CAPI_total_ON_over_OFF=z['metrics']['Native_prefill_decode_CAPI_s']/a['metrics']['Native_prefill_decode_CAPI_s'],Native_mean_decode_ON_over_OFF=z['metrics']['Native_mean_decode_CAPI_s']/a['metrics']['Native_mean_decode_CAPI_s']if a['metrics']['Native_mean_decode_CAPI_s']and z['metrics']['Native_mean_decode_CAPI_s']else None,client_wire_whole_ON_over_OFF=z['metrics']['client_full_wire_wall_s']/a['metrics']['client_full_wire_wall_s']))
        for mode,item in pair.items():
            row=item['recorded']
            if not row['correct']:wrong.append(dict(case_id=case['id'],suite=case['suite'],mode=mode,logical_index=item['entry']['logical_index'],parsed_answer=row['parsed_answer'],gold=str(case.get('gold_numeric',case.get('gold_choice'))),answer_matches_gold=row['answer_matches_gold'],parse_failure=row['parse_failure'],public_SSE_parse_error=row['public_parse_error'],cap_hit=row['cap_hit'],natural_EOS=row['natural_EOS']))
        if a['recorded']['correct']!=z['recorded']['correct']:
            arms={}
            for mode,item in pair.items():
                entry=item['entry'];row=item['recorded'];owner=item['owner'];score=core.response_row(entry,case,byrequest[case['id']]['body'],row,owner,b.read(item['body_name']),b.read(item['wire_name']),modules)
                arms[str(mode)]=dict(logical_index=entry['logical_index'],correct=score['correct'],parsed_answer=score['parsed_answer'],parse_failure=score['parse_failure'],answer_matches_gold=score['answer_matches_gold'],cap_hit=score['cap_hit'],natural_EOS=score['natural_EOS'],HTTP_status=row['HTTP_status'],HTTP_finish=row['HTTP_finish'],public_SSE_parse_error=row['public_parse_error'],generated_including_EOS=len(owner['generated_token_ids']),generated_IDs_SHA256=H(C(owner['generated_token_ids'])),actual_mode=owner['r450_mode'],actual_r366=owner['r450_r366_fixed'],actual_T1_coverage=item['proof'],request_body_SHA256=H(b.read(item['body_name'])),wire_reference=dict(logical_name=item['wire_name'],**b.used[item['wire_name']]),source_namespace=baseline['active_attempts'][item['metrics']['shard']]['source_namespace'])
                if private is not None:
                    folder=private/case['id'];folder.mkdir(exist_ok=True)
                    for suffix,data in [('body.json',b.read(item['body_name'])),('wire',b.read(item['wire_name'])),('generated-ids.json',C(owner['generated_token_ids']))]:
                        with(folder/('mode%d.'%mode+suffix)).open('xb')as f:f.write(data)
            discordants.append(dict(case_id=case['id'],suite=case['suite'],source_index=case['source_index'],gold=str(case.get('gold_numeric',case.get('gold_choice'))),complete_input_body_and_IDs_equal=True,prompt_IDs_SHA256=a['entry']['prompt_IDs_SHA256'],first_generated_ID_difference_index=first,arms=arms,logits_or_margin='UNKNOWN: no logits captured in this quality run; token divergence doesnot establish lowmargin or a cause.'))
    assert len(discordants)==7
    metrics=[v['metrics']for _,v in sorted(indexed.items())];names=['generated_including_EOS','visible_generated','Native_prefill_CAPI_s','Native_decode_CAPI_s','Native_prefill_decode_CAPI_s','Native_mean_decode_CAPI_s','Native_owner_whole_s','backend_owner_TTFT_s','backend_visible_tps','client_full_wire_wall_s']
    perarm={str(mode):dict(statistics={name:stats(m[name]for m in metrics if m['mode']==mode)for name in names},totals=dict(requests=256,generated_including_EOS=sum(m['generated_including_EOS']for m in metrics if m['mode']==mode),visible_generated=sum(m['visible_generated']for m in metrics if m['mode']==mode),completed_decode_calls=sum(m['decode_calls']for m in metrics if m['mode']==mode),completed_prefill_calls=sum(m['prefill_calls']for m in metrics if m['mode']==mode),no_eligible_T1_queries=sum(m['no_eligible_T1']for m in metrics if m['mode']==mode)))for mode in(0,1)}
    return dict(core_original_audit=baseline,complete_private_input_body_prompt_ID_proof=True,per_shard=per_shard,wrong_or_invalid_original_rows=wrong,discordant_case_count=len(discordants),discordant_cases=discordants,all512_original_metrics=metrics,primary_timing_pair_denominator=256,all_planned_pairs_including_first_cold_OFF_ON=True,per_arm_descriptive_timing=perarm,paired_descriptive_timing={name:stats(p[name]for p in timing_pairs)for name in['Native_owner_whole_ON_over_OFF','Native_CAPI_total_ON_over_OFF','Native_mean_decode_ON_over_OFF','client_wire_whole_ON_over_OFF']},same_generated_ID_array_pairs=sum(p['same_generated_ID_array']for p in timing_pairs),paired_timing_rows=timing_pairs,timing_limits='Primary all256 pairs, no gold/cap/parse/hit/sameoutput filtering. No extra warm; firstOFFON remainsincluded and coldload/cache effects arepossible. Eight independent loads and one counterbalanced pair percase; variableprompt/outputlength. NativeCAPI summary is whole TP8 hostforward inclusive enclosure, not perkernel or sum8 perowner clocks (individual ownerdurations notcaptured). Ownerwhole and clientfullwire include widercosts. Allclocks within eachprocess only. Ratios posthoc exploratory, not a frozen independent fixedtoken perf ABBA or general Modelgain. Client lacks perdelta arrivals: incrementalTTFT/clienttokenSLO UNKNOWN.',new_jobs=0,new_requests=0,rootcause='UNKNOWN',default_promotion=False,noninferiority_proven=False)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);p.add_argument('--private-discordants');a=p.parse_args(argv);q=analyze(a.bundle,a.private_discordants);out=Path(a.output);assert not out.exists();out.write_bytes((json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());print(json.dumps(dict(verified_requests=q['core_original_audit']['verified_requests'],discordants=q['discordant_case_count'],same_generated_ID_array_pairs=q['same_generated_ID_array_pairs'],historical_failed_attempts_retained=True,rootcause='UNKNOWN')));return 0
if __name__=='__main__':raise SystemExit(main())
