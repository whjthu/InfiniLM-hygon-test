"""Original four-leader/32worker terminal consumer, quality is not an all-correct gate."""
from pathlib import Path
import hashlib,json,math,time
from client import verify_row

NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']

def collect(root,run,actual,source_sha,client_result,deadline):
    root=Path(root);folder=root/'results/public-Kimi-text-provider-r434';refs=[];signatures=[];pids={};calls=[]
    def read(path):
        while time.time()<deadline:
            try:return json.loads(Path(path).read_bytes())
            except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
        raise TimeoutError('Bounded original terminal visibility: '+str(path))
    client=read(client_result);assert client['completed']and client['execution_passed']and not client['errors']
    assert client['attempted_HTTP']==client['completed_HTTP']==len(client['rows'])==256
    enc=read(root/'candidates/Kimi-wide-fewshot-quality-r448-CPU-v1/encoding.json');encoded={c['case_id']:c for c in enc['cases']}
    for node in range(4):
        path=folder/('Kimi-public-r428-'+run+'-node%d.json'%node);q=read(path);claim=read(folder/('claim-'+run+'-node%d.json'%node))
        assert claim['actual_step']==actual and claim['node']==node and claim['run']==run
        assert claim['source_manifest_SHA256']==q['public_source_manifest_SHA256']==source_sha
        assert q['completed']and q['protocol_before_destroy_passed']and q['all8_owned_workers_shutdown_completed']and not q['cleanup_errors']
        assert q['library_sha256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
        assert q['run']==run and q['node']==node and q['TP']==16 and q['PP']==2
        assert q['layout']==dict(node=node,local_workers=8,TP=16,TP_node_rank=node%2,PP=2,PP_stage=node//2,sampling_node=2,public_graph_enabled=False,owned_metadata_enabled=False)
        assert q['PP_ranges']==[[0,46],[46,93]]and q['cache_geometry']['max_total_tokens']==32768
        assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3
        assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4
        masks=[]
        for owner in q['owner_affinity']['actual_owner_report']['owners']:
            assert owner['changes']==1;masks.extend(map(int,owner['verified_mask'].split(',')))
        assert len(masks)==len(set(masks))==32
        rows=q['agent_requests'];assert len(rows)==256 and q['serving_protocol_complete']and not q.get('service_cleanup_errors')
        signature=[]
        for i,row in enumerate(rows):
            c=client['rows'][i];e=encoded[c['case_id']]
            assert row['request_id']==c['request_id']and row['input_tokens']==e['prompt_tokens']and row['output_budget']==4096
            ids=row['generated_ids'];natural=row['finish']=='eos'
            assert natural==c['natural_EOS']and row['finish']in('eos','length')and row['normal_reset']and row['prefill_mode']=='W512'
            assert len(ids)==c['generated_including_EOS']and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==c['generated_IDs_SHA256']
            assert (ids[-1]in[163585,163586]and not any(t in[163585,163586]for t in ids[:-1]))if natural else(len(ids)==4096 and not any(t in[163585,163586]for t in ids))
            h=row['history_render_audit'];assert h['qualified']and h['generation_thinking']is False and not h['markers_sent_to_Native']and h['Native_prompt_token_ids_SHA256']==e['prompt_IDs_SHA256']
            assert h['RC_order_preserved']and h['RC_one_contribution_per_source_field']
            obs=row['long_prefill_observations'];assert len(obs)==1 and obs[0]['full_prompt_consumed']and obs[0]['input_tokens_committed']==e['prompt_tokens']
            assert obs[0]['actual_forward_records']==len(e['prefill_chunk_tokens'])and not row.get('error')and not row.get('cleanup_errors')
            signature.append([c['case_id'],row['request_id'],e['prompt_IDs_SHA256'],ids,row['finish']])
        signatures.append(signature);pids[NODES[node]]=[claim['PID']];calls.append(q['agent_native_forward_calls'])
        refs.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),claim_PID=claim['PID']))
    assert all(s==signatures[0]for s in signatures)and len(set(calls))==1 and calls[0]>0
    return dict(reports=refs,leader_count=4,Native_internal_owner_count=32,external_HTTP=256,Native_requests=256,
                expected_node_PIDs=pids,all4_prompt_output_reset_signatures_equal=True,Native_calls_per_leader=calls,
                normal32destroy=True,quality_summary=client['summary'],all_correct_gate=False,noninferiority_proven=False,
                finite_assertions_scope='Unchanged pinned Native.finish BF16/flags checks, same qualified source; no added device counter or fabricated host finite probe.',
                public_API_scope_pass=True,performance_comparison=False,online_SLO=False)
