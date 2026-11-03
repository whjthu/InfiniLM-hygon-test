"""Terminal read only; all quality outcomes count, lifecycle/identity must hold."""
from pathlib import Path
import hashlib
import json
import sys
import time


def collect(root,run,actual,source_sha,client_result,deadline,nodes,slot,client_package):
    root=Path(root);folder=root/'results/GLM-W4-fewshot-quality-r444'/run
    def read(path):
        while time.time()<deadline:
            try:return json.loads(path.read_bytes())
            except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
        raise TimeoutError('Original bounded terminal visibility: '+str(path))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    client=read(Path(client_result))
    assert client['completed']and client['execution_passed']and not client['errors']
    assert client['actual_HTTP']==client['planned_HTTP']==len(client['rows'])==64
    assert (client['shard_index'],client['shards'],client['planned_global_HTTP'])==(slot,8,512)
    assert client['provider_source_SHA256']==source_sha and client['actual_step']==actual and client['run']==run
    sys.path.insert(0,str(Path(client_package)/'client-v1'))
    from owner_validation import verify_trace,verify_native_calls
    encoded=json.loads((root/'candidates/GLM-W4-fewshot-quality-r444-CPU-v1/encoding.json').read_bytes())
    encoded={r['case_id']:r for r in encoded['cases']}
    refs=[];pids={};signatures=[];memory=[];calls=0
    for stage,node in enumerate(nodes):
        leader=read(folder/('leader-%d.json'%stage));actors=read(folder/('actors-%d.json'%stage))
        assert leader['identity']['actual_step']==actors['identity']['actual_step']==actual
        pids[node]=sorted(set([leader['PID']]+actors['owned_process_PIDs']))
        for rank in range(8):
            path=folder/('owner-%d-%d.json'%(stage,rank));q=read(path)
            lp=folder/('lifecycle-%d-%d.json'%(stage,rank));life=read(lp)
            assert q['actual_step']==actual and q['source_manifest_SHA256']==q['service_profile_manifest_SHA256']==source_sha
            assert q['completed']and q['normal_shutdown']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']
            assert life['normal']and life['completed']and life['destroy_attempts']==life['shutdown_attempts']==1
            assert life['destroy_RC']==life['shutdown_RC']==0 and not life['cleanup_errors']and life['primary_error']is None
            assert q['all16_preweight_8GiB_passed']and q['all16_loaded_and_postload_4GiB_passed']and q['constructor_noV_verified']
            assert q['constructor_noV_census']['cache_category_bytes'].get('projected_value',0)==0
            def masks(v):return v['width'],sorted((r['device'],r['changes'],r['verified_mask'])for r in v['actual_owner_report']['owners'])
            assert masks(q['owner_affinity_after_load'])==masks(q['owner_affinity_final'])
            rows=q['service_requests'];assert q['attempted_Native']==q['completed_Native']==len(rows)==64
            signature=[]
            for index,row in enumerate(rows):
                c=client['rows'][index];e=encoded[c['case_id']]
                assert row['request_index']==row['admitted_index']==index==c['local_owner_index']
                assert row['native_profile']=='glm-nov16-w'+str(c['window'])and row['prefill_control_window']==c['window']
                assert row['original_reset_all16_completed']and row['generated_natural_EOS']==c['natural_EOS']
                assert row['reason']in('eos_token','length')and row['generated_tokens']==c['generated_tokens']
                assert row['actual_prompt_token_ids_SHA256']==e['prompt_IDs_SHA256']
                assert row['generated_token_ids_SHA256']==c['generated_IDs_SHA256']
                assert row['prompt_tokens']==e['prompt_tokens']and row['requested_output_budget']==4096
                verify_trace(row,c['window'],e);verify_native_calls(row,finite_expected=stage==1)
                signature.append((c['case_id'],c['window'],e['prompt_IDs_SHA256'],row['generated_token_ids_SHA256'],row['generated_natural_EOS']))
                calls+=len(row['native_calls'])
            signatures.append(signature)
            refs.append(dict(stage=stage,rank=rank,report_path=str(path),report_SHA256=sha(path),lifecycle_path=str(lp),lifecycle_SHA256=sha(lp)))
            memory.append(dict(stage=stage,rank=rank,minimum_sampled_forward_free_bytes=q['minimum_driver_free_bytes'],memory_after_load=q['memory_after_load'],snapshot_is_not_peak=True))
    assert len(signatures)==16 and all(s==signatures[0]for s in signatures)
    wires=[]
    for index,c in enumerate(client['rows']):
        directory=Path(client_result).parent/('%03d-%s-W%d'%(index,c['case_id'],c['window']))
        body=directory/'request.body.json';wire=directory/'response.wire'
        assert body.is_file()and wire.is_file()
        assert sha(body)==c['request_body_SHA256']and sha(wire)==c['response_wire_SHA256']
        wires.append(dict(request_body_path=str(body),request_body_SHA256=sha(body),response_wire_path=str(wire),response_wire_SHA256=sha(wire)))
    assert len(wires)==64 and {w['response_wire_SHA256']for w in wires}=={c['response_wire_SHA256']for c in client['rows']}
    return dict(actual_owner_count=16,external_requests=64,logical_requests_per_owner=64,logical_requests_total_owners=1024,
        all16_request_token_EOS_profile_reset_signatures_equal=True,reports=refs,wires=wires,expected_node_PIDs=pids,
        Native_CAPI_calls=calls,normal16destroy=True,memory_snapshots=memory,quality_summary=client['summary'],
        all_correct_gate=False,noninferiority_proven=False,full_benchmark=False,performance_comparison=False,online_SLO=False)
