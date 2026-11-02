"""Read local-only original evidence; publish only compact Model measurements."""
import argparse,hashlib,json,math,pathlib,statistics

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)
def answer(text):
    text=text.strip()
    if text.startswith('```json\n')and text.endswith('\n```'):text=text[8:-4]
    return json.loads(text)
def analyze(archive):
    r,c,f,w,watch=[archive[k]for k in ['report','client','fixture','wires','watch']]
    assert r['Native_SHA256']=='7e2b2f852499497377b8042c4430b55d5fe69052d7dc6509d0f03c5f3ffd3f45'
    calls={v['id']:json.loads(v['function']['arguments'])['path']for m in f['body']['messages']for v in m.get('tool_calls',[])}
    gold={}
    for m in f['body']['messages']:
        if m['role']!='tool':continue
        data=json.loads(m['content']);values={v['id']:v['value']for v in data['records']}
        gold[calls[m['tool_call_id']]]=dict(nonce=data['nonce'],sum=sum(values[i]for i in data['query_ids']))
    gold['total']=sum(v['sum']for v in gold.values());assert gold==f['expected_answer']
    assert r['completed']and r['passed']and r['actual_HTTP']==r['actual_Native']==6
    assert r['normal_server_shutdown']and r['normal_Native_shutdown']
    providers=r['provider_preweight']['actual_new_Native_providers']
    assert all(providers['deep_position_proof_'+n+'_r387']['SHA256']==r['Native_SHA256']for n in ['set','get','clear','stats'])
    assert not r['cleanup']['forced_cleanup']and not r['cleanup']['errors']and not r['cleanup']['HTTP_thread_alive_after_cleanup']
    owners=r['owner_exit_observation']['owners'];assert len(owners)==len({v['tid']for v in owners})==8
    assert {v['device']for v in owners}==set(range(8))and not any(v['task_exists_after_destroy']for v in owners)
    assert c['completed']and c['passed']and c['actual_HTTP_attempts']==c['actual_HTTP_completed']==c['actual_Native_completed']==6
    assert len(r['requests'])==len(c['requests'])==len(w)==6
    assert all(v[:2]==[86,86]for v in r['HC_actual_loaded_modules'])
    rows=[];last=None;last_positions=None
    for i,(native,client,wire,mode)in enumerate(zip(r['requests'],c['requests'],w,[0,1,0,1,1,0])):
        assert native['admitted_index']==client['index']==i and native['HC_mode']==client['hc_epilogue_mode']==1
        assert native['positions_mode']==client['positions_mode']==native['prefill_binding']['positions_mode']==mode
        assert native['prefill_binding']['prefill_chunk']==client['chunk']==1024
        assert native['prefill_binding']['hc_epilogue_mode']==1
        assert native['normal_reset_before']and native['normal_reset_after']
        assert native['prompt_tokens']==f['actual_prompt_tokens']==8489
        assert native['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==f['actual_prompt_token_ids_SHA256']
        ids=native['generated_token_ids'];assert ids[-1]==1 and len(ids)==native['actual_generated_tokens']<=128
        assert native['native_finish_reason']=='eos_token'
        response=wire['response'];request=wire['request'];choice=response['choices'][0]
        assert request['messages']==f['body']['messages']and request['tools']==f['body']['tools']and request['max_tokens']==128
        assert response['id']==native['request_id']and choice['finish_reason']=='stop'
        assert answer(choice['message']['content'])==gold
        assert wire['wire_SHA256']==client['wire_SHA256']
        trace=[v for v in native['Native_continuation']if v['kind']=='prefill'];past=0
        for t in trace:
            assert t['past']==past and t['tokens']==min(1024,8489-past);past+=t['tokens'];assert t['total']==past
        assert past==8489 and len(trace)==9
        forward=native['Native_forward_host_summary'];assert forward['prefill_calls']==9 and forward['decode_calls']==len(ids)-1
        coverage=native['HC_forward_coverage'];tokens=[v['tokens']for v in trace]+[1]*(len(ids)-1)
        assert len(coverage)==len(tokens)
        expected_sum=[0]*7
        for v,t in zip(coverage,tokens):
            delta=[0,0,86,86 if t==1 else 0,86 if t==1 else 0,86 if t!=1 else 0,0]
            assert v['input_tokens']==t and v['mode']==1 and v['not_hardware_GPU_hit_counter']
            assert v['per_owner_delta']==[delta]*8
            assert v['candidate_completed_branch']==(t==1)
            expected_sum=[a+b for a,b in zip(expected_sum,delta)]
        before=native['HC_counters_before_request'];after=native['HC_counters_after_request']
        if last is not None:assert before==last
        assert all([b-a for a,b in zip(x,y)]==expected_sum for x,y in zip(before,after));last=after
        pc=native['positions_forward_coverage'];assert len(pc)==len(tokens)
        kinds=['prefill']*len(trace)+['decode']*(len(ids)-1);position_sum=[0]*5
        groups={kind:[[0]*5 for _ in range(8)]for kind in ['prefill','decode']}
        for v,t,kind in zip(pc,tokens,kinds):
            delta=[43,62,496*t,0,0]if mode else[0,0,0,0,62]
            assert v['mode']==mode and v['input_tokens']==t and v['kind']==kind
            assert v['per_owner_delta']==[delta]*8 and v['actual_Runtime_owner_HOST_branch_counts']
            position_sum=[a+b for a,b in zip(position_sum,delta)]
            groups[kind]=[[a+b for a,b in zip(row,delta)]for row in groups[kind]]
        pb=native['positions_counters_before_request'];pa=native['positions_counters_after_request']
        assert len(pb)==len(pa)==8
        if last_positions is not None:assert pb==last_positions
        assert all([b-a for a,b in zip(x,y)]==position_sum for x,y in zip(pb,pa));last_positions=pa
        assert native['positions_counts_by_kind']=={kind:{'forwards':kinds.count(kind),'per_owner_sum':groups[kind]}for kind in ['prefill','decode']}
        assert client['positions_counts_by_kind']==native['positions_counts_by_kind']
        ticks=[t for t,token in zip(native['backend_token_monotonic'],ids)if token!=1]
        assert len(ticks)==len(ids)-1 and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))
        rate=(len(ticks)-1)/(ticks[-1]-ticks[0]);assert math.isclose(rate,client['backend_emitter_decode_tok_s'],rel_tol=1e-12)
        gaps=[b-a for a,b in zip(ticks,ticks[1:])]
        rows.append(dict(index=i,warmup=i<2,HC_mode=1,positions_mode=mode,input_tokens=8489,prefill_chunk=1024,generated_including_EOS=len(ids),output_token_ids_SHA256=digest(ids),visible_tokens=len(ticks),task_correct=True,natural_EOS=True,
                         Native_prefill_s=forward['prefill_s'],Native_decode_s=forward['decode_s'],Native_decode_s_per_call=forward['decode_s']/forward['decode_calls'],
                         backend_TTFT_from_Native_start_s=ticks[0]-native['started_monotonic'],backend_TTFT_from_admission_s=ticks[0]-native['admission']['accepted_monotonic'],
                         HTTP_wall_s=client['client_HTTP_wall_s'],visible_decode_tok_s=rate,backend_ITL_p95_s=percentile(gaps,.95),backend_ITL_p99_s=percentile(gaps,.99),
                         HC_qualified_forward_calls=len(coverage),HC_completed_candidate_per_owner=expected_sum[4],HC_counter_observation_s=native['HC_counter_observation_s'],positions_counter_observation_s=native['positions_counter_observation_s'],positions_per_owner_totals=position_sum,positions_by_kind=native['positions_counts_by_kind'],
                         memory_free_before_all8_bytes=[v[0]for v in native['GPU_free_before_request_all8']],memory_free_after_all8_bytes=[v[0]for v in native['GPU_free_all8']]))
    pairs={}
    for key in ['Native_prefill_s','Native_decode_s','Native_decode_s_per_call','backend_TTFT_from_Native_start_s','HTTP_wall_s','visible_decode_tok_s','backend_ITL_p95_s','backend_ITL_p99_s']:
        off=statistics.mean(rows[i][key]for i in [2,5]);on=statistics.mean(rows[i][key]for i in [3,4])
        pairs[key]=dict(OFF=off,ON=on,ON_over_OFF=on/off,OFF_endpoint_ratio=rows[5][key]/rows[2][key],ON_endpoint_ratio=rows[4][key]/rows[3][key])
    assert watch['Model_qualified']and watch['post_empty']and not watch['post']['kfd_pids']and watch['post']['HTTP43882_bindable']
    post=archive['independent_post'];assert post['owned_Model_PIDs']==post['kfd_pids']==[]and post['HTTP43882_bindable']and not post['GPU_imported']
    cost={'loading_wall_s':archive['legacy'].get('loading_seconds'),'all6_HTTP_complete_sum_s':sum(v['HTTP_wall_s']for v in rows),'formal4_HTTP_complete_sum_s':sum(rows[i]['HTTP_wall_s']for i in [2,3,4,5]),'all6_Native_prefill_sum_s':sum(v['Native_prefill_s']for v in rows),'all6_Native_decode_sum_s':sum(v['Native_decode_s']for v in rows),'all6_HC_counter_observation_sum_s':sum(v['HC_counter_observation_s']for v in rows),'all6_positions_counter_observation_sum_s':sum(v['positions_counter_observation_s']for v in rows),'request_batch_span_s':r['requests'][-1]['ended_monotonic']-r['requests'][0]['started_monotonic'],'legacy_phase_history':archive['legacy'].get('phase_history'),'nested_scopes_not_additive':True,'all8_allocated_cardtime_not_GPUbusy':True}
    for line in archive.get('accounting',[]):
        parts=line.split('|')
        if parts[0]==r['actual_step']:
            elapsed=parts[4];days,clock=(elapsed.split('-')if'-'in elapsed else('0',elapsed));h,m,s=map(int,clock.split(':'));seconds=int(days)*86400+h*3600+m*60+s;cost.update(GPU_allocated_wall_s=seconds,all8_GPU_allocated_card_seconds=8*seconds,Slurm_elapsed_raw=elapsed,Slurm_start_raw=parts[5],Slurm_end_raw=parts[6])
    return dict(passed=True,actual_GPU=r['actual_step'],CPU_keeper=watch['CPU_keeper_actual'],publication_commit=watch['publication_commit'],Native_SHA256=r['Native_SHA256'],
                rows=rows,paired=pairs,complete_cost=cost,quality='All6answers independently recomputed viaactualopaqueIDtofile/toolrecords;naturalEOS/full8489input',
                all86_real_HC_instances_per_owner=True,per_forward_all8_actual86HCcoverage=True,per_forward_all8_actual43scopes62borrowed_ON=True,HC_ON_fixed1=True,counter_scope='CPUbranchcompletedafterdrain;notGPUhardwarecounter',
                normal8_owner_threads_exited=True,post=watch['post'],independent_post=post,source_and_raw_evidence_SHA256=archive['evidence_SHA256'],
                Model_scope='Onefrozen8KC1toolworkload;fixed1024/128/TP8PP1;2excludedwarm+oneABBA;sameNative7e2bHCfixed1positionsOFFvsON;not976fvs7e2bcrosslibrarytiming',
                warmups_excluded=True,raw_payloads_published=False,component_ratios_not_extrapolated=True,
                limitations=['No broadbenchmark noninferiority','Memorycheckpoints notpeak','BufferedHTTP notclientincrementalTTFT','NoGPUbusy/kernelattribution','OneABBA notglobaldefaultqualification','PositionsHOSTcountersnotphysicalGPUevents'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    raw=json.loads(a.archive.read_bytes());result=analyze(raw.get('archive',raw));result['local_original_archive_SHA256']=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    if 'remote_archive_SHA256'in raw:result['remote_original_archive_SHA256']=raw['remote_archive_SHA256'];result['remote_original_archive_path']=raw['remote_archive_path']
    with a.output.open('x',encoding='utf8',newline='\n')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['paired']))
