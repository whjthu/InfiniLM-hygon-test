"""Independent stdlib audit of the one fixed256 original Kimi baseline.

No producer modules, tokenizer, HTTP, model or job are executed. Raw questions,
outputs and clocks remain in the private input archive and private full output.
"""
from pathlib import Path
from fractions import Fraction
import argparse,base64,copy,gzip,hashlib,json,math,re,statistics
RUN='public-kimi-wide-20261104-a';LABEL='Kimi-r448-wide-quality-'+RUN
NATIVE='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];EOS={163585,163586}
MANIFESTS={'source448':'dad1151e194c6cdc4f9d921b197a1ca83745e09548bbcc770ec8f18d447487e4','deployment':'a77f841fee2fa3aa8b7ae63b0ccd4505591801e45b3e6b3d109fc47522f2e495','final':'6ddeb0ead919d90fad138d4e048f05dfb810ce9ee56eeb8aa1f45e833ae57365','provider434':'f0821642a154b0ca51172093b8bd4119b117c9977b5a2d22fbbddb53136be7b4','runtime428':'fc7e8bc3e4b846929c7616477283071ab136c8595a5374584641a394b5ce6be6','public425':'122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'}
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
def finite(v):return type(v)in(int,float)and math.isfinite(v)
def percentile(values,f):
    if not values:return None
    a=sorted(values);x=(len(a)-1)*f;i=int(x);j=min(i+1,len(a)-1);return a[i]+(a[j]-a[i])*(x-i)
def stats(values):
    values=[float(v)for v in values if v is not None];assert all(math.isfinite(v)for v in values)
    return dict(n=len(values),minimum=min(values)if values else None,median=statistics.median(values)if values else None,mean=statistics.mean(values)if values else None,p95=percentile(values,.95),p99=percentile(values,.99),maximum=max(values)if values else None)
def parse_answer(case,text):
    if case['suite']=='gsm8k':
        matches=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
        try:value=Fraction(matches[-1].replace(',',''))if matches else None
        except(ValueError,ZeroDivisionError):value=None
        return value,value is not None and value==Fraction(case['gold_numeric'])
    assert case['suite']=='mmlu'
    matches=re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer)\s*:\s*[\[(]?([ABCD])[\])]?(?:\*\*)?(?:[.\s]|$)',text,re.IGNORECASE)
    if matches:value=matches[-1].upper()
    else:
        match=re.fullmatch(r'\s*[\[(]?([ABCD])[\])]?\s*[.]?\s*',text,re.IGNORECASE);value=match[1].upper()if match else None
    return value,value==case['gold_choice']
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;v=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return[max(0.,m-v),min(1.,m+v)]
class Archive:
    def __init__(self,path):
        self.path=Path(path);self.packed=self.path.read_bytes();self.raw=gzip.decompress(self.packed);self.q=json.loads(self.raw);self.files={}
        assert len(self.packed)==4606328 and H(self.packed)=='b1101d7f5fdd55edcf1a4725be9e692d5aad6f05404482b643a0c1901f78cdd3'
        assert self.q['schema']=='Kimi448-585-original-terminal-seal-v1'and self.q['run']==RUN and self.q['new_jobs']==self.q['new_HTTP']==0
        for name,pin in self.q['files'].items():
            b=base64.b64decode(pin['base64'],validate=True);assert len(b)==pin['bytes']and H(b)==pin['SHA256'],name;self.files[name]=b
    def read(self,name):return self.files[name]
    def json(self,name):return json.loads(self.read(name))
    def pin(self,name):return{k:self.q['files'][name][k]for k in['path','bytes','SHA256']}
def analyze(path):
    a=Archive(path)
    for group,digest in MANIFESTS.items():
        m=a.json(group+'/SOURCE_MANIFEST.json');assert H(a.read(group+'/SOURCE_MANIFEST.json'))==digest,group
        for name,h in m['files_SHA256'].items():assert H(a.read(group+'/'+name))==h,(group,name)
    watch=a.json('watch.json');client=a.json('client/result.json');config=a.json('deployment/config.json');go=a.json('final/GPU_MANIFEST.json')
    assert watch['completed']and watch['passed']and not watch['forced_cleanup']and not watch.get('error')
    assert (watch['actual_GPU'],watch['actual_client'],watch['CPU_keeper'])==('910380.585','910380.586','910407.944')
    assert watch['provider_source_SHA256']==MANIFESTS['provider434']and watch['deployment_source_SHA256']==MANIFESTS['deployment']and watch['final_source_manifest_SHA256']==MANIFESTS['final']
    assert watch['publication_commit']=='b68e2e51c9a81c12ea1d18ea972c1367562e3864'
    assert H(a.read('final/GPU_MANIFEST.json'))==watch['execution_manifest_SHA256']=='53d7f91d02cedc489c896bc810c471b591751fcd64ff88f7a59b701271447907'
    assert go['HTTP']==256 and go['arms']==1 and go['additional_HTTP']==0 and not go['retry']and not go['reload']
    assert config['Native_SHA256']==NATIVE and config['nodes']==NODES and(config['TP'],config['PP'],config['context'],config['capacity'])==(16,2,32768,1)
    config_sha=H(a.read('deployment/config.json'));assert client['config_SHA256']==config_sha and client['provider_source_SHA256']==MANIFESTS['provider434']and client['run']==RUN and client['actual_step']=='910380.585'
    assert client['completed']and client['execution_passed']and not client['errors']and client['pending_attempt']is None and not client['retry']and not client['reload']
    assert client['planned_HTTP']==client['attempted_HTTP']==client['completed_HTTP']==len(client['rows'])==256 and client['Native']==NATIVE and client['HTTP']and not client['SSE']
    enc=a.json('inputs/encoding.json');requests=a.json('inputs/requests-private.json')['requests'];cases=a.json('inputs/cases.json')['cases'];old=a.json('inputs/original-requests-and-pairs.json')['requests']
    assert H(a.read('inputs/encoding.json'))=='01fea327aa81d984f815812228ee2d6801fab76952cf14e578c603248d538a84'
    assert H(a.read('inputs/requests-private.json'))=='066e8985f6e9322eb9ff1c523eb37aa1347285e092e963258271aab666ea2ad6'
    assert H(a.read('inputs/cases.json'))=='c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38'
    assert enc['completed']and enc['passed']and enc['actual_CPU']=='910407.937'and enc['source_manifest_SHA256']==MANIFESTS['source448']and enc['actual_encodings']==256
    assert enc['actual_Kimi_codec']and enc['actual_public_protocol']and enc['actual_prepared_FIFO']and not enc['Native_dlopen']and not enc['GPU']and not enc['weights_loaded']
    assert not enc['final_modules']['project_or_SDK_maps']and not enc['final_modules']['Torch_CUDA_initialized']and set(enc['EOS_ids'])==EOS
    assert enc['codec_identity']['tokenizer_config_SHA256']=='5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e'and enc['codec_identity']['tiktoken_model_SHA256']=='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103'
    for name,pin in enc['actual_loaded_source_modules'].items():
        prefixes={'/refactor/incoming/Kimi-public-incremental-text-r434/':'provider434/','/refactor/incoming/public-Kimi-native-provider-r428/runtime-v2/':'runtime428/'}
        matches=[(start,target)for start,target in prefixes.items()if pin['path'].startswith(start)];assert len(matches)==1,(name,pin['path']);start,target=matches[0]
        assert H(a.read(target+pin['path'][len(start):]))==pin['SHA256'],name
    cpu942=a.json('CPU942.json');cpu943=a.json('CPU943.json')
    assert not cpu942['passed']and cpu942['completed']and cpu942['error']=="KeyError('node')"
    assert H(a.read('CPU943.json'))=='fc747defcfc9fc3d020520366b7bbb4c1505592d9909ad20b965b60dddf2cde4'and cpu943['completed']and cpu943['passed']and cpu943['production_manifest_SHA256']==MANIFESTS['deployment']
    assert cpu943['inherited942_source_and_actualBash4role_proof']['receipt_SHA256']==H(a.read('CPU942.json'))
    assert len(cases)==len(requests)==len(enc['cases'])==256 and len({c['id']for c in cases})==256
    assert len({r['request_id']for r in client['rows']})==256
    assert sum(c['suite']=='gsm8k'for c in cases)==sum(c['suite']=='mmlu'for c in cases)==128
    assert all(c['shot_count']==(8 if c['suite']=='gsm8k'else 5)for c in cases)
    leaders=[];claims=[];signatures=[];leader_metrics=[];request_metrics=[];score_rows=[];bad=[];native_calls=0;prefill_calls=0;generated=0
    model_sha=H(a.read('runtime428/qualified-model.template.json'));assert model_sha==config['model_manifest_SHA256']=='1cc92b0abf2b348e9e58214cb681c188bfa44da83e7314eec0490d2faaadb4ba'
    model=a.json('runtime428/qualified-model.template.json');expected_libraries=model['loaded_library_SHA256'];native_path='/refactor/candidates/Kimi-append512-r247/libKimi_append512_r247.so'
    for node in range(4):
        q=a.json('runtime/Kimi-public-r428-'+RUN+'-node%d.json'%node);claim=a.json('runtime/claim-'+RUN+'-node%d.json'%node);leaders.append(q);claims.append(claim)
        assert claim['run']==q['run']==RUN and claim['node']==q['node']==node and claim['actual_step']=='910380.585'
        assert claim['source_manifest_SHA256']==q['public_source_manifest_SHA256']==MANIFESTS['provider434']and claim['config_SHA256']==config_sha
        assert q['completed']and q['phase']=='completed'and q['protocol_before_destroy_passed']and q['all8_owned_workers_shutdown_completed']and not q['cleanup_errors']and not q.get('error')and not q['service_cleanup_errors']
        assert q['serving_protocol_complete']and q['library_sha256']==NATIVE and q['manifest_SHA256']==model_sha and(q['TP'],q['PP'],q['PP_stage'])==(16,2,node//2)
        assert q['layout']==dict(node=node,local_workers=8,TP=16,TP_node_rank=node%2,PP=2,PP_stage=node//2,sampling_node=2,public_graph_enabled=False,owned_metadata_enabled=False)
        assert q['PP_ranges']==[[0,46],[46,93]]and q['cache_geometry']['max_total_tokens']==32768 and not q['capacity_only_mode']and q['weight_payload_opened']
        for checkpoint in ['head_after_finish','head_after_protocol']:
            heads=q[checkpoint]['owners'];assert len(heads)==8 and{v['device']for v in heads}==set(range(8))
            for owner in heads:assert owner['model_constructed']and owner['PP_stage']==node//2 and owner['TP']==16 and owner['PP']==2 and owner['tp_rank']==(node%2)*8+owner['device']and owner['head_expected']==(node//2==1)
        weight=q['actual_engine_weight_keys'];assert weight['actual_layer_ids']==list(range(*q['PP_ranges'][node//2]))and weight['key_count']==q['loaded_keys']and weight['checkpoint_membership_validated']
        assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3 and q['owner_affinity']['verified']and q['owner_affinity']['width']==4
        affinity=q['owner_affinity']['actual_owner_report']['owners'];assert len(affinity)==8 and{v['device']for v in affinity}==set(range(8));masks=[]
        for owner in affinity:assert owner['changes']==1;masks.extend(map(int,owner['verified_mask'].split(',')))
        assert len(masks)==len(set(masks))==32
        assert set(q['actual_CAPI_providers'].values())=={native_path}
        for snapshot in ['actual_loaded_libraries_before_create','actual_loaded_libraries_after_server_import','actual_loaded_libraries']:
            got=q[snapshot];assert {n:v['SHA256']for n,v in got.items()}==expected_libraries
        assert 'modern_kimi_engine_destroy_r2475'in q['actual_CAPI_providers']
        normal_log=a.read('logs/'+LABEL+'-910380.585-%d.log'%node).decode();assert normal_log.count('KIMI_PUBLIC_R428_NORMAL '+str(node))==1
        assert len(q['agent_requests'])==256
        sig=[];local_times=[];expected_calls=0
        for i,(case,request,e,row,c)in enumerate(zip(cases,requests,enc['cases'],q['agent_requests'],client['rows'])):
            assert case['id']==request['case_id']==e['case_id']==c['case_id']and e['selected_ordinal']==i==c['index']and c['suite']==case['suite']and c['source_index']==case['source_index']
            assert e['source_case_compact_SHA256']==H(C(case))
            previous=next(v for v in old if v['case_id']==case['id']);body=copy.deepcopy(previous['body']);body.update(model='Kimi-K3',chat_template_kwargs=dict(enable_thinking=False));assert body==request['body']
            directory='client/%03d-%s/'%(i,case['id']);actual_body=a.read(directory+'request.body.json');wire=a.read(directory+'response.wire')
            assert actual_body==C(body)and H(actual_body)==request['request_body_SHA256']==e['request_body_SHA256']==c['request_body_SHA256']and H(wire)==c['response_wire_SHA256']
            assert body['messages']==previous['body']['messages']and body['max_tokens']==4096 and body['stream']is False and body['temperature']==0 and body['tool_choice']=='none'
            private=a.json('inputs/private/'+case['id']+'.json');assert H(a.read('inputs/private/'+case['id']+'.json'))==e['private_SHA256']and private['body']==body
            prompt=private['prompt'];ids0=private['prompt_token_ids'];assert isinstance(prompt,str)and H(prompt.encode())==e['prompt_SHA256']and H(C(ids0))==e['prompt_IDs_SHA256']and len(ids0)==e['prompt_tokens']
            assert all(type(t)is int and 0<=t<163840 for t in ids0)and len(ids0)+4096<=32768 and prompt.endswith('<|open|>response<|sep|>')and body['messages'][1]['content']in prompt
            assert e['prefill_chunk_tokens']==[min(512,len(ids0)-s)for s in range(0,len(ids0),512)]and e['actual_thinking']is False
            assert row['request_id']==c['request_id']and row['input_tokens']==len(ids0)and row['output_budget']==4096 and row['normal_reset']and row['prefill_mode']=='W512'and not row.get('error')and not row.get('cleanup_errors')
            h=row['history_render_audit'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and h['generation_thinking']is False and h['Native_prompt_token_ids_SHA256']==e['prompt_IDs_SHA256']and h['actual_prompt_SHA256']==e['prompt_SHA256']
            assert h['RC_order_preserved']and h['RC_one_contribution_per_source_field']and h['marked_render_reconstruction_equals_actual_prompt']and h['official_reencode_equals_actual_prompt_and_IDs']
            obs=row['long_prefill_observations'];assert len(obs)==1 and obs[0]['full_prompt_consumed']and obs[0]['input_tokens_committed']==len(ids0)and obs[0]['prompt_tokens']==len(ids0)and obs[0]['actual_forward_records']==len(e['prefill_chunk_tokens'])and obs[0]['true_input_width_max']==512 and obs[0]['width8_prefill_verify_commit_count']==0
            ids=row['generated_ids'];ticks=row['backend_token_monotonic'];assert ids and all(type(t)is int and 0<=t<163840 for t in ids)and len(ids)<=4096
            natural=row['finish']=='eos';assert row['finish']in('eos','length')and (ids[-1]in EOS and not any(t in EOS for t in ids[:-1])if natural else len(ids)==4096 and not any(t in EOS for t in ids))
            assert len(ticks)==len(ids)and all(finite(t)for t in ticks)and all(x<=y for x,y in zip(ticks,ticks[1:]))and row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['finished_monotonic']
            reply=json.loads(wire);assert 'error'not in reply and len(reply['choices'])==1 and reply['id']==row['request_id'];choice=reply['choices'][0];message=choice['message'];assert message['role']=='assistant'
            usage=reply['usage'];assert usage==c['usage']and all(type(usage[k])is int for k in['prompt_tokens','completion_tokens','total_tokens'])and usage['prompt_tokens']==len(ids0)and usage['completion_tokens']==len(ids)and usage['total_tokens']==len(ids0)+len(ids)
            assert choice['finish_reason']==c['HTTP_finish']and(choice['finish_reason']!='length'if natural else choice['finish_reason']=='length')
            assert c['natural_EOS']==natural and c['cap_hit']==(not natural)and c['generated_including_EOS']==len(ids)and c['generated_IDs_SHA256']==H(C(ids))
            visible=[t for t,token in zip(ticks,ids)if token not in EOS];deltas=[y-x for x,y in zip(visible,visible[1:])];strict=all(d>0 for d in deltas)
            tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and strict else None
            metrics=dict(index=i,case_id=case['id'],node=node,prompt_tokens=len(ids0),generated_including_EOS=len(ids),visible_tokens=len(visible),backend_TTFT_from_owner_row_start_s=ticks[0]-row['started_monotonic'],backend_owner_whole_s=row['finished_monotonic']-row['started_monotonic'],backend_visible_decode_tps=tps,backend_ITL_P95_s=percentile(deltas,.95),backend_ITL_P99_s=percentile(deltas,.99),backend_max_gap_s=max(deltas)if deltas else None,strict_positive_backend_adjacent_intervals=strict,original_prefill_enclosure_s=obs[0]['prefill_wall_s'])
            local_times.append(metrics);expected_calls+=len(e['prefill_chunk_tokens'])+len(ids)-1
            sig.append([case['id'],row['request_id'],e['prompt_IDs_SHA256'],ids,row['finish']])
            if node==0:
                content=message.get('content')or'';assert type(content)is str;answer,answer_correct=parse_answer(case,content);tool=bool(message.get('tool_calls'));correct=bool(answer_correct and natural and not tool);parse_failure=answer is None
                assert c['correct']==correct and c['answer_matches_gold']==answer_correct and c['parse_failure']==parse_failure and c['unexpected_tool_intent']==tool
                assert finite(c['client_wall_s'])and c['client_wall_s']>0
                score_rows.append(dict(case_id=case['id'],suite=case['suite'],source_index=case['source_index'],correct=correct,answer_matches_gold=bool(answer_correct),natural_EOS=natural,cap_hit=not natural,parse_failure=parse_failure,unexpected_tool=tool))
                if not correct:bad.append(dict(case_id=case['id'],suite=case['suite'],source_index=case['source_index'],parsed_answer=str(answer),gold=str(case.get('gold_numeric',case.get('gold_choice'))),natural_EOS=natural,cap_hit=not natural,parse_failure=parse_failure))
                request_metrics.append(dict(metrics,client_nonstream_whole_s=c['client_wall_s']));generated+=len(ids);prefill_calls+=len(e['prefill_chunk_tokens'])
                assert (tps is None and c['backend_visible_decode_tps']is None)or math.isclose(tps,c['backend_visible_decode_tps'],rel_tol=1e-12)
        assert q['agent_native_forward_calls']==expected_calls;native_calls=expected_calls;signatures.append(sig)
        leader_metrics.append(dict(node=node,PID=claim['PID'],loaded_keys=q['loaded_keys'],Native_calls=expected_calls,generated=sum(len(r['generated_ids'])for r in q['agent_requests']),owner_affinity_width=4,actual_internal_owners=len(affinity),create_seconds=q['create_seconds'],loading_seconds=q['loading_seconds'],clocks={k:stats(v[k]for v in local_times)for k in['backend_TTFT_from_owner_row_start_s','backend_owner_whole_s','backend_visible_decode_tps','backend_ITL_P95_s','backend_ITL_P99_s']}))
    assert all(sig==signatures[0]for sig in signatures)
    suites={}
    for suite in['gsm8k','mmlu']:
        rows=[r for r in score_rows if r['suite']==suite];assert len(rows)==128;k=sum(r['correct']for r in rows)
        suites[suite]=dict(planned=128,observed=128,correct=k,wrong_or_invalid=128-k,completion_adjusted_accuracy=k/128,Wilson95=wilson(k,128),natural_EOS=sum(r['natural_EOS']for r in rows),caps=sum(r['cap_hit']for r in rows),parse_failures=sum(r['parse_failure']for r in rows),missing=0)
        assert client['summary']['per_suite'][suite]['correct']==k
    epoch=watch['original_GPU_start_unix'];assert watch['service_deadline_unix']==epoch+28200 and watch['Native_deadline_unix']==epoch+28800 and watch['GPU_deadline_unix']==epoch+29400 and watch['keeper_deadline_unix']==epoch+32400
    stop=a.json('runtime/STOP-'+RUN+'.json');assert stop==dict(identity=dict(run=RUN,source_manifest_SHA256=MANIFESTS['provider434'],config_SHA256=config_sha,gpu_start_epoch=epoch),action='stop')
    accounting=a.q['accounting'];assert{r[0]for r in accounting}=={'910380.584','910380.585','910380.586','910380.587','910407.944'}and all(r[2:4]==['COMPLETED','0:0']for r in accounting)
    assert watch['actual_GPU_Slurm'][2:4]==watch['actual_client_Slurm'][2:4]==['COMPLETED','0:0']
    post=watch['post'];assert len(post)==4 and{p['node']for p in post}==set(NODES)
    for p in post:
        assert p['actual_step']=='910380.587'and p['CPU_only']and not p['GPU_initialized']and p['KFD_available']and not p['kfd_pids']and not p['matching_component_processes']and not p['expected_PIDs_still_exist']and p['TCP_connections_made']==0
        assert p['ports_bindable']=={'19743':True,'26743':True,'43844':True}and len(set(p['physical_PCI']))==8
    assert len({(p['node'],pci)for p in post for pci in p['physical_PCI']})==32
    identity=a.json('client-identity.json');assert identity['actual_CPU']=='910380.586'and identity['node']==NODES[0]and identity['client_SHA256']==H(a.read('deployment/client.py'))
    assert watch['actual_client_process_identity']==identity
    for i,claim in enumerate(claims):assert claim['PID']in watch['expected_node_PIDs'][NODES[i]]
    assert identity['PID']in watch['expected_node_PIDs'][NODES[0]]
    # Source-to-run cleanup evidence, not a fabricated new per-thread counter.
    cleanup=a.read('runtime428/public_kimi_native/cleanup.py').decode();provider=a.read('runtime428/public_kimi_native/provider.py').decode();controller=a.read('final/controller.py').decode()
    assert 'check(lib.modern_kimi_engine_destroy_r2475(ctx))'in cleanup and 'assert not sampler.thread.is_alive()'in cleanup and "control.collect('after_destroy', {})"in cleanup
    assert 'self.thread.join('in provider and 'if self.thread.is_alive():'in provider and 'os._exit(2)'in provider and 'assert self.closed'in provider
    assert 'gpu.poll()==0'in controller and 'client.poll()==0'in controller
    return dict(passed=True,audit_scope='One original8df TP16PP2/C1 custom chat fewshot256 baseline; not paired noninferiority or fullbenchmark',archive=dict(path=a.path.name,bytes=len(a.packed),SHA256=H(a.packed),uncompressed_SHA256=H(a.raw),files=len(a.files)),actual_GPU='910380.585',actual_client='910380.586',actual_keeper='910407.944',actual_post='910380.587',publication_commit=watch['publication_commit'],source_MANIFEST_SHA256=MANIFESTS,actual_CPU_evidence=dict(encoder='910407.937',encoder_SHA256=H(a.read('inputs/encoding.json')),changed_entry='910407.943',changed_entry_SHA256=H(a.read('CPU943.json')),prior942_FAIL_preserved=True),fixed_questions=256,actual_HTTP=256,all_original_full_messages_and_fields_match=True,all4_prompt_ID_output_EOS_reset_signatures_equal=True,per_suite=suites,pooled_correct=sum(r['correct']for r in score_rows),pooled_planned=256,natural_EOS=sum(r['natural_EOS']for r in score_rows),caps=sum(r['cap_hit']for r in score_rows),parse_failures=sum(r['parse_failure']for r in score_rows),wrong_cases=bad,Native_calls_per_leader=native_calls,Native_calls_total_4leaders=native_calls*4,prefill_calls_per_leader=prefill_calls,generated_including_EOS=generated,visible_generated_tokens=generated-256,leaders=leader_metrics,request_metrics=request_metrics,backend_metrics_scope='Within each owner row local monotonic only. TTFT starts after command broadcast/owner row creation and includes reset+prefill; no crossprocess monotonic subtraction. Decode rate uses nonEOS visible tokens (n-1)/(last-first), requires alladjacentpositive. Includes actualcontrol/emit work, not GPU-only.',client_metrics_scope='Ordinary nonstream HTTP; whole response wall only. No incremental/firstsemantic/client tokenITL or15SLO claim.',client_nonstream_whole_s=stats(r['client_nonstream_whole_s']for r in request_metrics),lifecycle=dict(all4_normal_leader_stamps=True,all4_successful_original_destroy_CAPI_paths=True,normal32_owner_shutdown_by_source_contract_and_zero_process_exit_plus_post=True,internal32_perthread_join_counter='UNKNOWN: no independent new Native perworker exit counter recorded',frontend_owned_thread_join='Pinned stop joins owner thread, rejects alive/failed, allleader Slurm0 and postPIDs gone corroborate',memory_sampler_thread_exit='Pinned cleanup asserts not alive before normal stamp',normal_STOP=True,forced_cleanup=False,controller_srun_zero_exit_assertions_passed=True,independent_post_4nodes_KFD_PID_ports_clean=True,accounting=accounting),noninferiority_proven=False,performance_comparison=False,new_speedup_claim=False,full_benchmark=False,online_or_15tokSLO=False,new_jobs=0,new_requests=0)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output');a=p.parse_args(argv);q=analyze(a.archive);raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode()
    if a.output:
        out=Path(a.output);assert not out.exists();out.write_bytes(raw)
    else:print(raw.decode(),end='')
    return 0
if __name__=='__main__':raise SystemExit(main())
