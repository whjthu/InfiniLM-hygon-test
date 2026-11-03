"""Independent stdlib audit of the sealed original128-request quality run.

Inputs are existing L-only artifacts. No model, job, HTTP request or tokenizer
is executed. The output contains scores/identities, never original responses.
"""
from pathlib import Path
from fractions import Fraction
import argparse, base64, gzip, hashlib, json, math, re, statistics

NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
OWNER='439cdee25fa8a0a3070b1aec8ef06601f176476a97028c56a46e3b8a8a47eddd'
POLICY='695b1a12bfd2929851b3ffa89a9267564f9fd9dc0045c4a9adfbcf7142b21584'
EOS={154820,154827,154829}
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()

def numeric(text):
    # Same stated #### final-answer format, independently read from raw wire.
    if '####'not in text:return None
    tail=text.rsplit('####',1)[1].lstrip(' *$')
    hit=re.match(r'[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?',tail)
    if not hit:return None
    try:return Fraction(hit[0].replace(',',''))
    except (ValueError,ZeroDivisionError):return None

def percentile(values,p):
    values=sorted(values)
    k=(len(values)-1)*p;i=int(k)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(k-i)

def analyze(archive,encoding_archive,observation,fixture_dir):
    outer_raw=Path(archive).read_bytes();outer=json.loads(outer_raw)
    zipped=base64.b64decode(outer['base64']);assert sha(zipped)==outer['compressed_SHA256']
    raw=gzip.decompress(zipped);assert sha(raw)==outer['raw_SHA256']and len(raw)==outer['raw_bytes']
    sealed=json.loads(raw);assert sealed['schema']=='GLM418-original-terminal-files-v1'
    assert sealed['new_jobs']==sealed['new_requests']==0
    files=sealed['files'];assert len(files)==485
    for path,row in files.items():
        b=row['raw'].encode();assert len(b)==row['bytes']and sha(b)==row['SHA256'],path
    def get(suffix):
        names=[p for p in files if p.endswith('/'+suffix)]
        assert len(names)==1,(suffix,len(names))
        return json.loads(files[names[0]]['raw'])
    def original(path,expected):
        assert files[path]['SHA256']==expected
        return json.loads(files[path]['raw'])
    watch=get('GLM-r418-two-NoV16-quality-quality-20261103-a-watch.json')
    assert watch['completed']and watch['passed']and not watch['forced_cleanup']
    assert watch['warms']==0 and not watch['retry']and not watch['reload']
    assert watch['original_GPU_start_unix']==1793682572.873715
    assert watch['GPU_manifest_SHA256']=='6fa5706e9d5ccb0852a1ae270e0566c1f880bc2a31268a9ba8a5b0e4cade0275'
    obs=json.loads(Path(observation).read_bytes())
    accounting=obs['accounting'];assert accounting['returncode']==0 and len(accounting['rows'])==7
    expected={'910376.855','910376.856','910376.857','910376.858','910376.859','910376.860','910407.849'}
    actual=set()
    for row in accounting['rows']:
        cols=row.split('|');actual.add(cols[0]);assert cols[2:4]==['COMPLETED','0:0']
    assert actual==expected
    assert len(watch['post'])==4 and {p['node']for p in watch['post']}=={'a02r4n12','a02r4n13','a02r4n14','a02r4n15'}
    for p in watch['post']:
        assert p['actual_step']=='910376.860'and p['CPU_only']and not p['GPU_initialized']and p['KFD_available']
        assert not p['kfd_pids']and not p['matching_component_processes']and not p['expected_PIDs_still_exist']
        assert all(p['ports_bindable'].values())and p['TCP_connections_made']==0
    F=Path(fixture_dir)
    data_raw=(F/'gsm8k-new64.json').read_bytes();plan_raw=(F/'requests-and-pairs.json').read_bytes()
    assert sha(data_raw)=='ec1d98eb5dc6a45c7ab06c54df707715854afaecc046b3b657d62a1b028de523'
    assert sha(plan_raw)=='b536c18715a60aebd593d181436503b65ab10f70258f7abc791f1c9426797183'
    data=json.loads(data_raw);plan=json.loads(plan_raw)
    cases={c['id']:c for c in data['cases']};assert len(cases)==64
    bodies={x['case_id']:x for x in plan['requests']};assert len(bodies)==64
    enc_outer=json.loads(Path(encoding_archive).read_bytes());enc=enc_outer['archive']
    receipt=enc['files'][enc['receipt_path']];enc_raw=receipt['rawtext'].encode()
    assert sha(enc_raw)==enc['receipt_SHA256']=='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12'
    encoding=json.loads(enc_raw);encoded={x['case_id']:x for x in encoding['cases']}
    owners=[];emitters={};signatures={};masks={};native_calls=0
    for item in watch['reports']:
        rep,stage,rank=item['replica'],item['stage'],item['rank']
        assert rep in (0,1)and stage in (0,1)and 0<=rank<8
        o=original(item['path'],item['SHA256']);life=original(item['lifecycle_path'],item['lifecycle_SHA256'])
        constructor=original(item['constructor_path'],item['constructor_SHA256'])
        step=['910376.856','910376.857'][rep];run='quality-20261103-a-r'+str(rep)
        for x in [o,life,constructor]:
            assert x['actual_step']==step and x['source_manifest_SHA256']==OWNER and x['run']==run
            assert (x['stage'],x['rank'])==(stage,rank)
        assert o['library_sha256']==o['Native_SHA256']==constructor['Native_SHA256']==NATIVE
        assert o['service_profile_manifest_SHA256']==POLICY and o['constructor_calls_expected']==1
        assert o['completed']and o['normal_shutdown']and o['normal_reset']and o['normal_reset_close']
        assert o['observer_native_destroy_returned0']and o['worker_shutdown_completed']and o['post_close_Torch_liveness']
        assert o['cache_rows']==1 and o['cache_capacity']==32768 and o['context_policy_env']=='1'
        assert not constructor['loaded']and constructor['context_policy']=='v403-no-resident-V'
        assert life['completed']and life['normal']and life['primary_error']is None and not life['cleanup_errors']
        assert life['destroy_attempts']==life['shutdown_attempts']==life['constructor_calls_expected']==1
        assert life['destroy_RC']==life['shutdown_RC']==0
        assert o['cache_after_destroy']['active']==o['cache_after_destroy']['active_bytes']==0
        assert not any(o['M15_after_destroy'])and not any(o['host_bank_after_destroy_r4504'])
        assert o['owner_affinity']['verified']and o['owner_affinity']['width']==4
        affinity=o['owner_affinity']['actual_owner_report']['owners'][rank]
        assert affinity['device']==rank and affinity['changes']==1
        mask=set(int(v)for v in affinity['verified_mask'].split(','));assert len(mask)==4
        for previous in masks.setdefault((rep,stage),[]):assert not mask&previous
        masks[(rep,stage)].append(mask)
        rows=o['service_requests'];assert len(rows)==o['completed_Native']==o['full_requests_completed']==64
        sig=[]
        for i,row in enumerate(rows):
            assert row['request_index']==row['admitted_index']==i and row['original_reset_all16_completed']
            assert row['reason']=='eos_token'and row['generated_natural_EOS']and row['requested_output_budget']==4096
            assert row['prefill_control_window']in (2,4)and row['native_profile']=='glm-nov16-w'+str(row['prefill_control_window'])
            calls=row['native_calls'];prefill=[c for c in calls if c['prefill']];decode=[c for c in calls if not c['prefill']]
            assert len(prefill)==1 and sum(c['T']for c in prefill)==row['prompt_tokens']and prefill[0]['past']==0
            assert len(decode)==row['generated_tokens']-1
            for j,c in enumerate(decode):assert c['T']==1 and c['past']==row['prompt_tokens']+j
            assert all(math.isfinite(c['Native_CAPI_s'])and c['Native_CAPI_s']>=0 for c in calls)
            native_calls+=len(calls)
            sig.append([row[k]for k in ['sequence','prompt_tokens','generated_tokens','reason','generated_token_ids_SHA256','actual_prompt_token_ids_SHA256','prefill_control_window']])
        signature=sha(compact(sig))
        if rep in signatures:assert signatures[rep]==signature
        else:signatures[rep]=signature
        owners.append(dict(replica=rep,stage=stage,rank=rank,actual_step=step,report_SHA256=item['SHA256'],
            lifecycle_SHA256=item['lifecycle_SHA256'],requests=64,signature_SHA256=signature,normal=True,
            preweight_free_bytes=item['preweight_free_bytes'],minimum_sampled_forward_free_bytes=item['minimum_sampled_forward_free_bytes']))
        if stage==1 and rank==0:
            cleanup=o['cleanup'];assert o['normal_HTTP_shutdown']and cleanup['normal_server_shutdown']and not cleanup['forced_cleanup']and not cleanup['errors']
            assert not cleanup['HTTP_thread_alive_after_cleanup']
            emitters[rep]=o
    assert len(owners)==32 and all(len(v)==8 for v in masks.values())
    checked=[];unique=set();paired={};wrong=[]
    for rep in watch['replicas']:
        n=rep['index'];client=original(rep['client_result'],rep['client_result_SHA256'])
        assert client['completed']and client['execution_passed']and not client['errors']and client['actual_HTTP']==client['planned_HTTP']==64
        folder=rep['client_result'].rsplit('/',1)[0]
        expected_cases={v['case_id']for i,v in enumerate(plan['requests'])if(i//2)%2==n}
        assert len(expected_cases)==32
        for i,c in enumerate(client['rows']):
            case_id=c['case_id'];W=c['window'];key=(case_id,W)
            assert key not in unique and case_id in expected_cases and W in (2,4);unique.add(key)
            case=cases[case_id];enc_case=encoded[case_id]
            where=folder+('/%03d-%s-W%d'%(i,case_id,W))
            wire=files[where+'/response.wire']['raw'].encode();body=files[where+'/request.body.json']['raw'].encode()
            assert sha(wire)==c['response_wire_SHA256']and sha(body)==c['request_body_SHA256']==bodies[case_id]['compact_body_SHA256']
            assert body==compact(bodies[case_id]['body'])and json.loads(files[where+'/headers.json']['raw'])['X-InfiniLM-Native-Profile']=='glm-nov16-w'+str(W)
            reply=json.loads(wire);assert reply['id']==c['response_id']and len(reply['choices'])==1
            choice=reply['choices'][0];message=choice['message'];assert message['role']=='assistant'
            value=numeric(message.get('content')or'');gold=Fraction(case['gold_numeric'])
            assert str(value)==c['parsed_numeric']and not message.get('tool_calls')
            o=emitters[n]['service_requests'][i]
            assert o['request_id']==reply['id']and sha(compact(o))==c['owner_row_compact_SHA256']
            assert o['prefill_control_window']==W and o['normal_reset_before']and o['normal_reset_after']
            assert o['prompt_tokens']==reply['usage']['prompt_tokens']==enc_case['prompt_tokens']
            assert o['actual_prompt_token_ids_SHA256']==enc_case['prompt_IDs_SHA256']
            ids=o['generated_token_ids'];assert len(ids)==o['actual_generated_tokens']==reply['usage']['completion_tokens']
            assert sha(compact(ids))==o['generated_token_ids_SHA256']==c['generated_IDs_SHA256']
            assert ids[-1]in EOS and not any(v in EOS for v in ids[:-1])and all(type(v)is int and 0<=v<154880 for v in ids)
            assert choice['finish_reason']==c['HTTP_finish']=='stop'and c['natural_EOS']and not c['cap_hit']and not c['parse_failure']
            correct=value==gold;assert c['correct']==correct
            ticks=o['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(v)for v in ticks)
            assert all(a<=b for a,b in zip(ticks,ticks[1:]))
            visible=ticks[:-1];assert len(visible)==c['visible_tokens']and len(visible)>1
            speed=(len(visible)-1)/(visible[-1]-visible[0]);assert abs(speed-c['backend_visible_decode_tok_s'])<1e-10
            ttft=visible[0]-o['admission']['accepted_monotonic'];assert abs(ttft-c['backend_TTFT_from_admission_s'])<1e-10
            assert o['admission']['accepted_monotonic']<=o['started_monotonic']<=ticks[0]<=ticks[-1]<=o['ended_monotonic']
            paired.setdefault(case_id,{})[W]=correct
            row=dict(case_id=case_id,window=W,correct=correct,natural_EOS=True,visible_tokens=len(visible),
                backend_decode_tok_s=speed,backend_TTFT_s=ttft,HTTP_wall_s=c['HTTP_wall_s'],
                request_body_SHA256=sha(body),response_wire_SHA256=sha(wire),generated_IDs_SHA256=c['generated_IDs_SHA256'])
            checked.append(row)
            if not correct:wrong.append(dict(case_id=case_id,window=W,gold=str(gold),parsed=str(value),response_wire_SHA256=sha(wire)))
    assert len(checked)==128 and len(paired)==64 and all(set(v)=={2,4}for v in paired.values())
    arms={}
    for W in (2,4):
        rows=[v for v in checked if v['window']==W];speeds=[v['backend_decode_tok_s']for v in rows]
        arms[str(W)]=dict(requests=len(rows),correct=sum(v['correct']for v in rows),natural_EOS=64,caps=0,parse_failures=0,
            backend_visible_decode_median=statistics.median(speeds),backend_visible_decode_min=min(speeds),backend_visible_decode_max=max(speeds),
            strict_over15=sum(x>15 for x in speeds),HTTP_wall_mean_s=statistics.mean(v['HTTP_wall_s']for v in rows))
    discordant=dict(W4_only_correct=sum(v[4]and not v[2]for v in paired.values()),W2_only_correct=sum(v[2]and not v[4]for v in paired.values()))
    return dict(id='GLM418-root-original128-quality-and-normal32owner-audit',state='SCOPED_64PAIR_SCREEN_AND_NORMAL_LIFECYCLE_PASS_NO_NONINFERIORITY_PROOF',
        archive_SHA256=sha(outer_raw),archive_raw_SHA256=outer['raw_SHA256'],archive_raw_bytes=len(raw),original_files_checked=485,
        actual_GPU=['910376.856 COMPLETED0/1h48m51s','910376.857 COMPLETED0/1h31m55s'],
        actual_clients=['910376.858 COMPLETED0','910376.859 COMPLETED0'],keeper='910407.849 COMPLETED0/1h49m04s',post='910376.860 COMPLETED0/all4nodesclean',
        Native_SHA256=NATIVE,owner_manifest_SHA256=OWNER,policy_manifest_SHA256=POLICY,
        root_reparsed_raw_HTTP_and_gold=True,root_verified_all32_owner_and_lifecycle=True,
        arms=arms,complete_pairs=64,paired_discordance=discordant,wrong_answers=wrong,
        paired_accuracy_W4_minus_W2=(arms['4']['correct']-arms['2']['correct'])/64,
        one_sided95_zero_observed_harm_upper_if_zero_discordance=1-.05**(1/64),
        noninferiority_margin_prespecified=False,quality_noninferiority_proven=False,full_benchmark=False,
        requests=checked,owners=owners,owner_request_executions=2048,Native_CAPI_calls_all32=native_calls,
        all32_context_create_destroy_shutdown_once=True,all16_per_replica_signature_equal=True,all4nodes_post_clean=True,
        original_retry=False,original_reload=False,new_jobs=0,new_requests=0,
        limits=['Custom0shot randomly selected64GSM8K disjoint from333historical questions; one wrong answer per arm is retained.',
            'Both arms63/64 and zero discordant pairs do not establish full benchmark noninferiority or correctness on other tasks.',
            'All prompts fit one512chunk: this screens normal W4 profile safety; it does not test multichunk W4 speedup.',
            'Backend emitter clocks are not public incremental HTTP/SSE clocks; buffered response remains so.',
            'Actual original418 lifecycle passes; new425 public provider/source remains independently device-unqualified.'],
        decision='Retain this independent64-pair quality screen and normal original418 lifecycle; free original n12..15 only after fresh resource check for any new job. No128 replay or broad/default/performance promotion.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('encoding_archive');p.add_argument('observation');p.add_argument('fixture_dir');p.add_argument('--out',required=True);a=p.parse_args()
    output=Path(a.out);assert not output.exists()
    result=analyze(a.archive,a.encoding_archive,a.observation,a.fixture_dir)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k]for k in ['state','original_files_checked','arms','paired_discordance','wrong_answers','root_verified_all32_owner_and_lifecycle']},ensure_ascii=False))
