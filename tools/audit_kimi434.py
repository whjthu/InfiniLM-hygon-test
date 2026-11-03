"""Independent stdlib reparse of original six public SSE requests and owners."""
import argparse,base64,copy,gzip,hashlib,json,math,statistics
from pathlib import Path
from audit_kimi428 import strict_json

def sha(b):return hashlib.sha256(b).hexdigest()

def tool_gold(call,expected,history):
    assert call['id']and call['id']not in history
    assert call['function']['name']=='write_file'
    args=strict_json(call['function']['arguments'])
    assert set(args)=={'path','content'}and args['path']=='final.json'
    content=strict_json(args['content'])
    assert content==expected and type(content['total'])is int
    return True

def reparse_sse(raw,clock,expected,history):
    lines=raw.splitlines(keepends=True);arrivals=clock['line_arrivals']
    assert len(lines)==len(arrivals)
    sent,ended=clock['sent'],clock['completed'];assert sent<=clock['headers']<=ended
    first={k:None for k in ['reasoning','content','tool','semantic','complete_tool']}
    identifiers=set();call={'id':'','function':{'name':'','arguments':''}}
    finish=usage=None;done=False;events=[];previous=sent
    for line,tick in zip(lines,arrivals):
        t=tick['arrival_monotonic'];assert math.isfinite(t)and previous<=t<=ended
        assert tick['bytes']==len(line)and tick['is_data']==line.startswith(b'data: ')
        previous=t
        if not line.startswith(b'data: '):assert not line.strip()or line.startswith(b':');continue
        value=line[6:].strip()
        row=dict(arrival_monotonic=t,bytes=len(line),nonempty_delta=False)
        if value==b'[DONE]':done=True;events.append(row);continue
        assert not done
        q=strict_json(value);assert 'error'not in q
        if q.get('id'):identifiers.add(q['id'])
        if q.get('usage')is not None:usage=q['usage']
        for channel,key in [('reasoning','reasoning_content'),('content','content'),('tool','tool_calls')]:
            nonempty=any(bool(c.get('delta',{}).get(key))for c in q.get('choices',[]))
            row[channel+'_nonempty']=nonempty
            if nonempty and first[channel]is None:first[channel]=t-sent
            if nonempty and first['semantic']is None:first['semantic']=t-sent
        row['nonempty_delta']=any(row.get(k+'_nonempty')for k in ['reasoning','content','tool']);events.append(row)
        for choice in q.get('choices',[]):
            assert choice.get('index',0)==0
            finish=choice.get('finish_reason')or finish
            for delta in choice.get('delta',{}).get('tool_calls',[]):
                assert delta.get('index',0)==0
                if delta.get('type')is not None:assert delta['type']=='function'
                call['id']+=delta.get('id','')
                for key in ['name','arguments']:call['function'][key]+=delta.get('function',{}).get(key,'')
        if first['complete_tool']is None:
            try:tool_gold(call,expected,history)
            except (AssertionError,ValueError,TypeError,KeyError):pass
            else:first['complete_tool']=t-sent
    assert done and len(identifiers)==1 and finish=='tool_calls'
    assert events==clock['events'],'Recorded SSE events differ from raw line reparse'
    assert tool_gold(call,expected,history)
    for key in ['reasoning','content','tool']:assert first[key]==clock['first_'+key+'_payload_s']
    assert first['semantic']==clock['first_payload_delta_s']
    assert ended-sent==clock['wall_s']
    return identifiers.pop(),usage,call,dict(first_reasoning_s=first['reasoning'],first_content_s=first['content'],first_tool_fragment_s=first['tool'],first_semantic_s=first['semantic'],complete_typed_tool_s=first['complete_tool'],client_completion_s=ended-sent)

def audit(path):
    blob=path.read_bytes();assert sha(blob)=='55e5ec0f06037fe5c90a7b4408567d4a281191c3aa3cb9a0527cdfc0a1a108e7'
    seal=strict_json(gzip.decompress(blob));files={}
    for name,r in seal['files'].items():
        b=base64.b64decode(r['base64'],validate=True);assert len(b)==r['bytes']and sha(b)==r['SHA256'];files[name]=b
    read=lambda name:strict_json(files[name])
    pins={'deployment':'912ba09e32a215a6c1650a9a5a6115c91d52115eefbda24dd2a9e77575999216','provider':'f0821642a154b0ca51172093b8bd4119b117c9977b5a2d22fbbddb53136be7b4','runtime-parent':'fc7e8bc3e4b846929c7616477283071ab136c8595a5374584641a394b5ce6be6','public-parent':'122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'}
    for label,h in pins.items():
        assert sha(files[label+'/SOURCE_MANIFEST.json'])==h
        for name,digest in read(label+'/SOURCE_MANIFEST.json')['files_SHA256'].items():assert sha(files[label+'/'+name])==digest
    watch,client,card,config=map(read,['watch.json','client/result.json','deployment/FIXTURE_CARD.json','deployment/config.json'])
    assert watch['completed']and watch['passed']and not watch['forced_cleanup']and not watch['retry']and not watch['reload']
    assert watch['actual_GPU']=='910380.576'and watch['actual_client']=='910380.577'and watch['CPU_keeper']=='910407.880'
    assert watch['publication_commit']=='d40bf47cea8d28c731f48dcf816910d027cc3a9e'
    assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==len(client['rows'])==6
    native='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
    model=read('runtime-parent/qualified-model.template.json');assert sha(files['runtime-parent/qualified-model.template.json'])==config['model_manifest_SHA256']
    assert config['Native_SHA256']==model['native_DSO_SHA256']==native
    encoded,original=read('fixture-encoded.json'),read('fixture-original.json')
    template=encoded['body'];assert template['messages']==original['messages']and template['tools']==original['tools']
    prompt=encoded['prompt_token_ids'];assert len(prompt)==9613
    prompt_sha=sha(json.dumps(prompt,separators=(',',':')).encode())
    records=[strict_json(m['content'])for m in template['messages']if m['role']=='tool'and 'records'in m['content']]
    assert len(records)==1;data=records[0];values={r['id']:r['value']for r in data['records']};assert len(values)==len(data['records'])
    selected=[values[k]for k in data['query_ids']];assert all(type(v)is int for v in selected)
    expected=dict(nonce=data['nonce'],total=sum(selected))
    history={c['id']for m in template['messages']for c in m.get('tool_calls',[])}
    assert len(history)>0 and all(m['tool_call_id']in history for m in template['messages']if m['role']=='tool')
    signatures=[];owners=[];timings=[];IDs=set()
    for node in range(4):
        r=read('run/Kimi-public-r428-public-kimi-text-20261103-a-node%d.json'%node)
        claim=read('run/claim-public-kimi-text-20261103-a-node%d.json'%node)
        assert claim['actual_step']=='910380.576'and claim['node']==r['node']==node
        assert claim['source_manifest_SHA256']==r['public_source_manifest_SHA256']==pins['provider']
        assert r['completed']and r['protocol_before_destroy_passed']and r['all8_owned_workers_shutdown_completed']and r['serving_protocol_complete']
        assert not r['cleanup_errors']and not r.get('service_cleanup_errors')
        assert r['library_sha256']==native and r['TP']==16 and r['PP']==2 and r['PP_ranges']==[[0,46],[46,93]]
        assert r['loaded_keys']==r['actual_engine_weight_keys']['key_count']and r['actual_allstage_weight_keys']['all93_layer_ids_exactly_once_across_PPstages']
        for phase in ['actual_loaded_libraries_before_create','actual_loaded_libraries_after_server_import','actual_loaded_libraries']:
            for library,digest in model['loaded_library_SHA256'].items():assert r[phase][library]['SHA256']==digest
        affinity=r['owner_affinity'];assert affinity['verified']and affinity['width']==4
        masks=[]
        for worker in affinity['actual_owner_report']['owners']:
            assert worker['changes']==1;mask=list(map(int,worker['verified_mask'].split(',')));assert len(mask)==4;masks+=mask
        assert len(masks)==len(set(masks))==32
        assert len(r['agent_requests'])==6;signature=[];calls=0
        for i,row in enumerate(r['agent_requests']):
            ids=row['generated_ids'];assert ids and all(type(v)is int and v>=0 for v in ids)
            assert ids[-1]in[163585,163586]and not any(v in[163585,163586]for v in ids[:-1])
            assert row['normal_reset']and row['finish']=='eos'and not row.get('error')and not row.get('cleanup_errors')
            assert row['input_tokens']==9613 and row['output_budget']==8192 and row['prefill_mode']=='W512'
            h=row['history_render_audit'];assert h['qualified']and h['Native_prompt_token_ids_SHA256']==prompt_sha
            assert h['RC_order_preserved']and h['RC_one_contribution_per_source_field']and not h['markers_sent_to_Native']
            obs=row['long_prefill_observations'];assert len(obs)==1 and obs[0]['full_prompt_consumed']and obs[0]['input_tokens_committed']==9613
            signature.append([row['request_id'],prompt_sha,ids]);calls+=math.ceil(9613/512)+len(ids)-1
            if node==0:
                fixture=card['plan'][i];c=client['rows'][i];body=read('client/request-%d.json'%i)
                assert fixture['index']==i and fixture['delivery']==c['delivery']==['P0','P1','P0','P1','P1','P0'][i]
                assert fixture['warm']==c['warm']==(i<2)
                reference=copy.deepcopy(template);reference['text_delivery']=fixture['delivery'];assert body==reference
                assert body['max_tokens']==8192 and body['stream']is True and sha(files['client/request-%d.json'%i])==fixture['actual_request_SHA256']==c['actual_request_SHA256']
                wire=files['client/response-%d.wire'%i];assert sha(wire)==c['wire_SHA256']
                ident,usage,call,t=reparse_sse(wire,c['client_clock'],expected,history)
                assert ident==c['request_id']==row['request_id']and ident not in IDs;IDs.add(ident)
                assert usage==c['usage']and usage['prompt_tokens']==9613 and usage['completion_tokens']==len(ids)and usage['total_tokens']==9613+len(ids)
                ticks=row['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(v)for v in ticks)and all(a<=b for a,b in zip(ticks,ticks[1:]))
                assert row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['finished_monotonic']
                visible=[tick for token,tick in zip(ids,ticks)if token not in[163585,163586]]
                t.update(index=i,delivery=fixture['delivery'],warm=i<2,generated_including_EOS=len(ids),visible=len(visible),backend_dequeue_to_first_emit_s=ticks[0]-row['started_monotonic'],backend_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0]))
                timings.append(t)
        assert r['agent_native_forward_calls']==calls;signatures.append(signature)
        owners.append(dict(node=node,PID=claim['PID'],keys=r['loaded_keys'],create_s=r['create_seconds'],load_s=r['loading_seconds'],Native_calls=calls))
    assert all(s==signatures[0]for s in signatures)
    assert len(watch['post'])==4
    for p in watch['post']:
        assert p['actual_step']=='910380.578'and not p['kfd_pids']and not p['matching_component_processes']and not p['expected_PIDs_still_exist']
        assert p['ports_bindable']=={'19743':True,'26743':True,'43844':True}
    accounts={r[0]:r for s in seal['accounting']if(r:=s.split('|'))}
    for step in ['910380.575','910380.576','910380.577','910380.578','910407.880']:assert accounts[step][2:4]==['COMPLETED','0:0']
    formal={mode:[r for r in timings if not r['warm']and r['delivery']==mode]for mode in ['P0','P1']}
    comparison={}
    for key in ['first_reasoning_s','first_semantic_s','first_tool_fragment_s','complete_typed_tool_s','client_completion_s','backend_visible_decode_tps']:
        a=[r[key]for r in formal['P0']];b=[r[key]for r in formal['P1']]
        comparison[key]={'P0':a,'P1':b,'P0_mean':statistics.mean(a),'P1_mean':statistics.mean(b),'P1_over_P0_mean':statistics.mean(b)/statistics.mean(a),'P0_endpoint_drift_fraction':abs(a[-1]/a[0]-1)}
    return dict(passed=True,seal_SHA256=sha(blob),raw_files_verified=len(files),original_bytes_verified=sum(len(v)for v in files.values()),source_pins=pins,actual_GPU='910380.576',actual_client='910380.577',keeper='910407.880',post='910380.578',source_claim_commit=watch['publication_commit'],external_requests=6,normal_32_workers=True,all_four_request_signatures_equal=True,natural_EOS_and_integer_tool_gold=True,post_clean=True,owners=owners,timings=timings,formal_comparison=comparison,formal_indices=[2,3,4,5],client_incremental_token_SLO_qualified=False,new_Native_compute_speedup_qualified=False,full_benchmark=False,scope='Same qualified8df TP16PP2/32cards/C1/32768/W512/AR one complete9613input tool history. Two warm+one ABBA, P0 buffered vs P1 incremental text. Real client first reasoning and complete-tool clocks; chunks are not tokens; no warm sample as formal; no cross-node clock subtraction.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive)
    with a.output.open('x',encoding='utf8')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k]for k in ['passed','raw_files_verified','external_requests','formal_comparison']}))
