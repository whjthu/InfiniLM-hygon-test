"""Read-only independent terminal audit and compact real C1T1 profile analysis.

No jobs, model imports, HTTP requests, writes or source execution. Raw token,
prompt and event data are read locally on the experiment host, never exported.
"""
from pathlib import Path
import argparse
import datetime
from fractions import Fraction
import hashlib
import json
import math
import re
import subprocess

LABEL='Kimi-latest99-supplement-Model-r279'
RUN='kimi-latest99-supplement-r279-20261031'
NATIVE='ec239cf5d60fa4c00d6f2168beb2289b13151063f1f068a83e3cf64d8b9c99e4'
MODEL='f72bbc3fcb0971e395121ba297eae08080d67eab16c7eb9aefb101eaa1057671'
NODES={'a14r2n08','a14r2n09','a14r2n10','a14r2n12'}


def require(condition, message):
    if not condition: raise ValueError(message)


def bracket(values):
    require(len(values)==3 and all(type(x)in(int,float) and math.isfinite(x) and x>0 for x in values),'Invalid bracket')
    a,b,c=values; ref=(a+c)/2; ratio=b/ref; spread=abs(a-c)/ref
    return {'OFF_before':a,'ON':b,'OFF_after':c,'ON_over_OFF_bracket':ratio,
            'OFF_relative_spread':spread,'positive_overhead_no_more_than3pct':ratio<=1.03 and spread<=.03,
            'absolute_disturbance_within3pct':abs(ratio-1)<=.03 and spread<=.03}


def numeric_answer(text):
    values=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
    if not values: return None
    try: return Fraction(values[-1].replace(',',''))
    except(ValueError,ZeroDivisionError): return None


def audit(root, accounting):
    root=Path(root).resolve(); evidence={}
    def read(relative, expected=None):
        p=(root/relative).resolve(); require(p.is_relative_to(root),'Evidence escapes root')
        raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
        require(expected is None or digest==expected,'Evidence SHA mismatch: '+str(relative))
        evidence[str(relative)]=digest; return json.loads(raw)
    w=read('results/'+LABEL+'-watch.json')
    require(w['actual_handle']=='910380.518' and w['terminal'][0]=='910380.518' and
            w['terminal'][2:4]==['COMPLETED','0:0'],'Unexpected GPU terminal')
    require(all(w.get(k)is True for k in ['completed','post_empty','normal_service_lifecycle','client_passed',
            'Model_qualified','all4_actual_request_prompt_output_signatures_equal','Native_backend_serial_proven']),
            'Incomplete terminal/lifecycle proof')
    require(w['normal34_repeated']is False and w['same_run75_claim']is False,'Supplement scope differs')
    post=w['post_KFD']; require(post['terminal'][2:4]==['COMPLETED','0:0'] and len(post['rows'])==4 and
            {x['node']for x in post['rows']}==NODES and all(x['passed'] and not x['kfd_pids'] and
            x['service_port_free'] and x['listener_count']==0 and len(x['port_states'])==3 and
            {s['port']for s in x['port_states']}=={19743,26743,43844} and
            all(s['SO_REUSEADDR_bindable'] and s['LISTEN']==0 for s in x['port_states'])for x in post['rows']),
            'Four-node independent postcheck incomplete')
    a=w['automatic_client']; require(a['client_actual_handle']=='910380.519' and a['client_terminal'][0]=='910380.519'
            and a['client_terminal'][2:4]==['COMPLETED','0:0'] and a['retries']==0,'Client not clean terminal')
    records={x[0]:x for x in accounting}; require(all(h in records and records[h][2:4]==['COMPLETED','0:0']
            for h in ['910380.518','910380.519','910380.520','910407.626']),'Independent accounting incomplete')
    cp='results/Kimi-agent-P0-client-r275-'+RUN+'/'
    c=read(cp+'result.json',a['client_result_SHA256'])
    require(all(c.get(k)is True for k in ['completed','passed','no_retries','no_cropping','full_request_sequence_bound'])
            and c['actual_HTTP_requests']==c['actual_Native_requests']==41,'Client scope incomplete')
    require(c['multicall']['completed'] and c['multicall']['passed'] and c['multicall']['actual_GPU_handle']=='910380.518'
            and c['multicall']['same_run']==RUN and c['multicall_returncode']==0 and
            [x['mode']for x in c['multicall']['cases']]==['chat','thinking'] and
            all(x['passed'] and x['actual_HTTP_requests']==2 and x['same_response_two_actual_model_calls'] and
                x['unique_opaque_IDs'] and x['tool_results_reversed_without_ID_replacement'] and
                x['file_nonce_sum_associations_correct']for x in c['multicall']['cases']),'Multicall witness incomplete')
    require(len(w['nodes'])==4 and {x['node']for x in w['nodes']}==set(range(4)),'Node references incomplete')
    owners=[]; signatures=[]; profiles=[]; spans=[]; node_summaries=[]
    for i in range(4):
        ref=next(x for x in w['nodes']if x['node']==i)
        q=read('results/Kimi-latest-service-r275-'+RUN+'-node%d.json'%i,ref['SHA256'])
        require(q['manifest_SHA256']==MODEL and q['library_sha256']==NATIVE and all(q.get(k)is True for k in
                ['completed','all8_owned_workers_shutdown_completed','protocol_before_destroy_passed','agent_HTTP_normal_shutdown']),
                'Owner source/lifecycle differs')
        rows=q['agent_requests']; require(len(rows)==41 and [x['admitted_index']for x in rows]==list(range(41)),
                'Owner admissions incomplete')
        sig=[]
        for j,x in enumerate(rows):
            h=x['history_render_audit']; ticks=x['backend_token_monotonic']; ids=x['generated_ids']
            require(x['normal_reset'] and x['finish']=='eos' and ids[-1]in[163585,163586] and
                    x['input_tokens']+x['output_budget']<=32768 and x['prefill_mode']=='W512' and
                    h['qualified'] and h['prequeue_bound'] and not h['markers_sent_to_Native'] and
                    x['long_prefill_progress']['input_tokens_committed']==x['input_tokens']==h['prompt_tokens'] and
                    len(ticks)==len(ids) and all(a<=b for a,b in zip(ticks,ticks[1:])), 'Owner request/EOS/input incomplete')
            sig.append([x['request_id'],x['client_session'],x['input_tokens'],x['output_budget'],ids,x['finish'],h['Native_prompt_token_ids_SHA256']])
        signatures.append(hashlib.sha256(json.dumps(sig,separators=(',',':')).encode()).hexdigest())
        require(all(x['finished_monotonic']<=y['started_monotonic']for x,y in zip(rows,rows[1:])), 'Backend overlaps')
        recs={x['label']:x for x in q['sparse_model_profiles']}; on=recs['agent-'+rows[39]['request_id']]
        mapping=on['actual_forward_map']; raw=on['raw_decoder_owner_events']
        require(on['armed'] and raw['all8_published'] and len(raw['owners'])==8 and len(mapping)==1 and
                mapping[0]['C']==mapping[0]['T']==1 and mapping[0]['widths']==[1] and
                mapping[0]['actual_verify_index_zero_based']==2,'Real third-verify C1T1 coverage incomplete')
        selected=[]; visible_spans=[]
        for j in [38,39,40]:
            v=recs['agent-'+rows[j]['request_id']]
            samples=[x['CAPI_maxhost_s']for x in v['verify_host_samples']if x['verify_index_zero_based']==2]
            require(len(samples)==1,'Selected verify missing');selected.append(samples[0])
            r=rows[j]; ts=[t for t,k in zip(r['backend_token_monotonic'],r['generated_ids'])if k not in[163585,163586]]
            require(len(ts)>1,'Visible decode span missing');visible_spans.append(ts[-1]-ts[0])
        local=on['decoder_summary']['local_owner_calls']; require(len(local)==8,'Local owner profile incomplete')
        for entry in local:
            require(math.isfinite(entry['whole_task_stream_ms']) and entry['whole_task_stream_ms']>0,'Bad stream span')
            sums={}
            for interval in entry['adjacent_intervals']:
                duration=interval['stream_span_ms']; require(math.isfinite(duration) and duration>=0,'Bad event interval')
                scope=interval['scope']; sums[scope]=sums.get(scope,0)+duration
            require(sums.keys()==entry['adjacent_disjoint_stream_ms_by_scope'].keys() and
                    all(math.isclose(v,entry['adjacent_disjoint_stream_ms_by_scope'][k],rel_tol=1e-6,abs_tol=1e-5)for k,v in sums.items()),
                    'Adjacent interval aggregation differs')
            owners.append(dict(entry,node=i))
        node_summaries.append({'node':i,'owner_report_SHA256':ref['SHA256'],'actual_third_verify_CAPI':bracket(selected),
                'visible_decode_span':bracket(visible_spans),'host_adapter_s':on['host_adapter_s'],
                'profile_record_SHA256':hashlib.sha256(json.dumps(on,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
        profiles.append([rows[j]['generated_ids']for j in range(36,41)])
        if i==0: head=rows
    require(len(set(signatures))==1 and signatures[0]==w['prompt_output_signature_SHA256'],'Cross-node signature differs')
    require({(x['PP_stage'],x['TP_rank'])for x in owners}=={(s,r)for s in range(2)for r in range(16)} and len(owners)==32,
            'Global32 owner coordinates differ')
    require(all(x==profiles[0][0]for rows in profiles for x in rows),'Profile output paths differ')
    gs=c['GSM8K']; suite=read('incoming/Kimi-latest-service-r275/benchmark/gsm8k-fixed-random32.json',gs['suite_SHA256'])
    require(len(suite['cases'])==len(gs['rows'])==32,'Fixed screening suite incomplete')
    correct=0; speeds=[]
    for j,(case,logged)in enumerate(zip(suite['cases'],gs['rows'])):
        p=cp+'gsm32/%02d/'%j; body=read(p+'request-001.request.json',logged['request_SHA256'])
        wire=read(p+'request-001.wire',logged['wire_SHA256']); choice=wire['choices'][0]
        require(body['messages'][-1]=={'role':'user','content':case['question']} and body['max_tokens']==4096 and
                choice['finish_reason']=='stop' and not choice['message'].get('tool_calls') and
                wire['id']==head[j+4]['request_id'],'GSM request/reply binding differs')
        ok=numeric_answer(choice['message'].get('content')or'')==Fraction(case['gold_numeric'].replace(',',''))
        require(ok is logged['correct'],'GSM scoring differs');correct+=ok
        r=head[j+4]; ts=[t for t,k in zip(r['backend_token_monotonic'],r['generated_ids'])if k not in[163585,163586]]
        speed=(len(ts)-1)/(ts[-1]-ts[0]);require(math.isclose(speed,logged['backend_node0_decode_tok_s'],rel_tol=1e-10),'Emitter speed differs');speeds.append(speed)
    critical=[]
    for stage in [0,1]:
        entry=max((x for x in owners if x['PP_stage']==stage),key=lambda x:x['whole_task_stream_ms'])
        critical.append({k:v for k,v in entry.items()if k!='adjacent_intervals'})
    return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'SUPPLEMENT41_QUALITY_AND_REAL_C1_PROFILE_SCOPED_PASS',
            'actual_GPU':'910380.518','actual_client':'910380.519','actual_CPU_keeper':'910407.626',
            'independent_accounting':{h:records[h]for h in ['910380.518','910380.519','910380.520','910407.626']},
            'all32_lifecycle_and_41_EOS':True,'four_node_KFD_ports_clean':True,'signature_SHA256':signatures[0],
            'model_manifest_SHA256':MODEL,'Native_SHA256':NATIVE,'normal34_inherited_from':'910380.513','same_run75':False,
            'multicall4_scoped_PASS':True,'GSM32':{k:v for k,v in gs.items()if k!='rows'},
            'independent_GSM_correct':correct,'C1_profile':{'owners':32,'C':1,'T':1,'verify_index_zero_based':2,
                'matched_actual_output_path':True,'per_node_brackets':node_summaries,'critical_owner_per_stage':critical,
                'absolute_bracket_disturbance_within3pct':all(v[k]['absolute_disturbance_within3pct']for v in node_summaries for k in ['actual_third_verify_CAPI','visible_decode_span']),
                'scope':'Same-stream spans include dependencies and communication. Do not add PP stages; stage1 includes upstream waiting. No pure GPU busy, CCL, PP or router/GEMM/dequant split.'},
            'new_speedup_claim':False,'client_streaming_SLO_qualified':False,'wide_benchmark_no_regression_proven':False,
            'raw_tokens_prompts_or_event_intervals_exported':False,'evidence_SHA256':evidence}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args()
    raw=subprocess.check_output(['sacct','-S','2026-10-31','-j','910380,910407','-n','-P',
            '-o','JobID,JobName%120,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30)
    print(json.dumps(audit(args.root,[x.split('|')for x in raw.splitlines()]),indent=2,allow_nan=False))
