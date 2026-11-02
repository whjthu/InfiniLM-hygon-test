"""Actual four-arm clocks/costs, without changing the failed controller verdict."""
from pathlib import Path
import hashlib,json,statistics,collections

D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;O=G/'results/bw1000/2026-11-02'
q=json.loads((L/'Kimi364365-run-d-full-terminal-independent-post.json').read_bytes())
watch=q['files'][0]['report'];targets=[x['report']for x in q['files'][1:5]];draft=q['files'][5]['report']
assert watch['completed']and not watch['passed']and watch['error_type']=='AssertionError'and'final_evidence'not in watch
assert all(t['completed']and t['all8_owned_workers_shutdown_completed']and not t['cleanup_errors']for t in targets)
assert all(t['pair_report']['completed']and t['pair_report']['normal_draft_protocol_close']for t in targets)
assert draft['completed']and draft['passed']and draft['normal_protocol_close']and q['all5_independent_post_clean']
assert len(targets)==4 and all(len(t['pair_report']['requests'])==4 for t in targets)
expected=[('retained8635','AR'),('near32k','AR'),('near32k','spec'),('retained8635','spec')]
signatures=[]
for t in targets:
    assert t['source_manifest_SHA256']=='909cea03171e52e81f377d1f2d61425a7d46778522fb28c4be989b46f41436b1'
    assert t['Native_SHA256']=='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
    assert [(r['name'],r['mode'])for r in t['pair_report']['requests']]==expected
    assert all(r['completed']and r['normal_reset']and r['finish']=='eos'and r['quality']['passed']for r in t['pair_report']['requests'])
    signatures.append([(r['ids_SHA256'],r['tokens'],r['finish'])for r in t['pair_report']['requests']])
assert all(x==signatures[0]for x in signatures)


def percentile(values,p):
    values=sorted(values);i=(len(values)-1)*p;a=int(i);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(i-a)


arms=[]
for index,r in enumerate(targets[0]['pair_report']['requests']):
    ticks=[(token,event['at_s'])for event in r['emitter_events']for token in event['tokens']]
    assert [token for token,tick in ticks]==r['generated_ids']
    visible=[tick for token,tick in ticks if token not in [163585,163586]]
    assert len(visible)==r['visible_tokens']==58 and len(ticks)==r['tokens']==59
    first,last=visible[0],visible[-1];rate=(len(visible)-1)/(last-first)
    assert abs(rate-r['backend_decode_tok_s'])<1e-12 and abs(first-r['TTFT_s'])<1e-12
    itl=[b-a for a,b in zip(visible,visible[1:])]
    calls=r['Native_calls'];cycles=r['cycles'];costs=r.get('publisher_costs',[])
    proposed=sum(c['width'][0]-1 for c in calls if c['kind']=='verify')if r['mode']=='spec'else 0
    matched=sum(c['raw_matched_proposals']for c in cycles)
    sums={op:sum(c['local_wall_s']for c in costs if c['operation']==op)for op in sorted({x['operation']for x in costs})}
    arms.append(dict(ordinal=index,name=r['name'],mode=r['mode'],input_tokens=r['input_tokens'],output_budget=r['output_budget'],
        source_report=q['files'][1]['path'],source_report_SHA256=q['files'][1]['SHA256'],prompt_SHA256=r['prompt_SHA256'],
        output_ID_SHA256=r['ids_SHA256'],generated_ids=r['generated_ids'],emitter_events=r['emitter_events'],
        emitter_clock=dict(domain='node0 request-relative monotonic; admission is run337 start after one-time draft connect/load handshake',
            request_start_s=0,first_visible_s=first,last_visible_s=last,first_to_last_visible_s=last-first,
            TTFT_s=first,generated_tokens_including_EOS=len(ticks),useful_non_EOS_tokens=len(visible),
            EOS_ids_excluded=[163585,163586],padding_generated=False,
            rate_formula='(visible_count-1)/(last_visible-first_visible)',backend_first_last_tok_s=rate,
            full_service33_card_backend_rate=rate/33,
            ITL_P50_s=percentile(itl,.5),ITL_P95_s=percentile(itl,.95),ITL_P99_s=percentile(itl,.99),ITL_max_s=max(itl),
            last_prompt_harvest_append_seal_after_firstemit_included=True,
            earlier_prompt_capture_append_inside_TTFT=True,proposal_only_window=False,HTTP_or_online_SLO_qualified=False),
        generation_local_wall_s=r['generation_local_wall_s'],request_through_reset_local_wall_s=r['request_through_reset_local_wall_s'],
        max4leader_generation_local_wall_s=max(t['pair_report']['requests'][index]['generation_local_wall_s']for t in targets),
        max4leader_request_through_reset_local_wall_s=max(t['pair_report']['requests'][index]['request_through_reset_local_wall_s']for t in targets),
        actual_source_report_service_cards=r['all_service_cards'],all33_capacity_generation_tok_per_card_s=len(visible)/r['generation_local_wall_s']/33,
        quality=r['quality'],finish=r['finish'],normal_reset=r['normal_reset'],cycles=cycles,
        proposal_diagnostics=dict(actual_proposals=proposed,raw_matched_proposals=matched,
            accepted_useful_visible_proposals=sum(c['accepted_visible_proposals']for c in cycles),
            cycle_useful_emitted_tokens=sum(c['useful_emitted_tokens']for c in cycles),
            cycle_emitted_tokens_including_EOS=sum(c['emitted_tokens']for c in cycles),
            prefill_visible_tokens=len(visible)-sum(c['useful_emitted_tokens']for c in cycles)if cycles else len(visible),
            raw_matched_fraction=None if not proposed else matched/proposed,
            denominator_includes_proposed_suffix_after_EOS_stop=True if cycles else False,
            replay_cycles=sum(c['needs_replay']for c in cycles),terminal_cycles=sum(c['terminal']for c in cycles)),
        phase_costs=dict(clock='Local elapsed scopes; no absolute timestamps subtracted across nodes',
            cycle_local_wall_sum_s=sum(c['cycle_local_wall_s']for c in cycles),
            publisher_operation_local_wall_s=sums,
            publisher_operations_additive=False,
            overlap='commit_replay_publish encloses append_committed or retire RPC and row checks; do not add them twice. QueryRPC includes worker handler+wire+fourleader gather.',
            sum_Native_prefill_CAPI_maxhost_s=sum(c['CAPI_maxhost_s']for c in calls if c['kind']in ['prefill','append512']),
            sum_Native_verify_CAPI_maxhost_s=sum(c['CAPI_maxhost_s']for c in calls if c['kind']=='verify'),
            sum_local_feature_harvest_s=sum(c['feature_harvest_local_s']for c in calls),
            first_Native_prefill_record=calls[0],Native_call_count=len(calls),
            CAPI_enclosures_include_waits_not_pure_GPU=True,harvest_not_added_to_CAPI_without_boundary_proof=True)))

pairs=[]
for name,ar,spec in [('retained8635',arms[0],arms[3]),('near32k',arms[1],arms[2])]:
    pairs.append(dict(name=name,AR_ordinal=ar['ordinal'],spec_ordinal=spec['ordinal'],
        backend_first_last_AR_tok_s=ar['emitter_clock']['backend_first_last_tok_s'],
        backend_first_last_spec_tok_s=spec['emitter_clock']['backend_first_last_tok_s'],
        backend_first_last_rate_gain_fraction=spec['emitter_clock']['backend_first_last_tok_s']/ar['emitter_clock']['backend_first_last_tok_s']-1,
        full_generation_AR_s=ar['generation_local_wall_s'],full_generation_spec_s=spec['generation_local_wall_s'],
        full_generation_rate_gain_fraction=ar['generation_local_wall_s']/spec['generation_local_wall_s']-1,
        spec_backend_first_last_over15=spec['emitter_clock']['backend_first_last_tok_s']>15,
        HTTP_or_online_SLO_qualified=False,stable_Model_gain_qualified=False,
        order_cold_prefill_confound='FirstAR retained firstNativeprefill33.812s versus laterSpec4.696s; do not attribute all retained complete-generation improvement to speculation'if name=='retained8635'else'One AR/spec paired diagnostic; completegeneration is nearly flat despite decode gain'))

watch_path=O/'Kimi364365-run-d-watch719-FAIL-sealed.json';draft_path=O/'Kimi364365-run-d-draft672-PASS-sealed.json'
for f,p in [(q['files'][0],watch_path),(q['files'][5],draft_path)]:
    raw=(json.dumps(f['report'],indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==f['SHA256'];p.write_bytes(raw)
post_path=O/'Kimi364365-run-d-delayed554680-clean.json'
post=dict(independent_post=q['independent_post'],all5_independent_post_clean=q['all5_independent_post_clean'],accounting=q['accounting'],
    original_post_failure='552/553 node0a14r2n08 port39947 false; KFDempty and allotherports/nodes true; original TCPstate/PIDowner unknown; delayed clean does not rewrite719FAILED1')
post_path.write_bytes((json.dumps(post,indent=2)+'\n').encode())
metadata=[]
for f,t in zip(q['files'][1:5],targets):
    keys=['node','actual_step','source_manifest_SHA256','Native_SHA256','qualification_scope','layout','cache_geometry','actual_CAPI_providers','actual_loaded_libraries','first_capture_validation','draft_wire_source_binding','loading_seconds','head_after_finish','after_finish_memory_guard','all8_owned_workers_shutdown_completed','capture_CPU_cleanup_completed','sampler_CPU_cleanup_completed','control_CPU_cleanup_completed','cleanup_errors']
    metadata.append(dict(path=f['path'],SHA256=f['SHA256'],bytes=f['bytes'],fields={k:t[k]for k in keys if k in t}))
total_useful=sum(a['emitter_clock']['useful_non_EOS_tokens']for a in arms)
out=dict(schema=1,state='ENGINE_FOUR_ARMS_QUALITY_NORMAL_PASS_CONTROLLER_POST_FAIL_DELAYED5CLEAN_PENDING_ROOT_AUDIT',
    run='longpair-20261102-d',source_claim_commit=watch['publication_commit'],controller_verdict=dict(actual_CPU='910407.719',state='FAILED1',
        watch_passed=watch['passed'],error_type=watch['error_type'],error_frames=watch['error_frames'],final_evidence_present=False,
        original_watch_path=q['files'][0]['path'],original_watch_SHA256=q['files'][0]['SHA256']),
    GPU_terminal=dict(target='910380.550 COMPLETED0 /46m11s',draft='910371.672 COMPLETED0 /7m13s'),
    all4leader_request_signatures_equal=True,all4requests59EOS_tool_gold=True,all32_normal_destroy=True,draft_normal_protocol_close=True,
    strict_worker_source_manifest_SHA256=draft['source_manifest_SHA256'],draft_phase_counts=draft['phase_counts'],draft_commands=draft['commands'],
    draft_memory_fences=len(draft['memory_fences']),all_fences_original4GiB=all(f['floor_passed']and f['floor_bytes']==4<<30 for f in draft['memory_fences']),
    draft_allocator_trim_count=draft['allocator_trim_count'],draft_pressure_trims=[f for f in draft['memory_fences']if f['trimmed']],
    clocks=dict(emitter='Each armnode0 request-relative monotonic; all59ticks retained',
        generation='run337 start after draft-connect before publisher.prefill; completeprompt capture/append and query/commit/retire included',
        one_time_draft_connect_load_handshake_local_s=targets[0]['pair_report']['draft_connect_load_handshake_local_s'],
        one_time_draft_init_before_first_spec_request_start=True,
        real_user_admission_before_bootstrap_not_recorded=True,HTTP_or_online_SLO_qualified=False,
        no_posthoc_subtraction_from_reported_generationwall=True,
        cross_node_absolute_subtraction=False),
    service_cost=dict(all_service_capacity_cards=33,target_collective_cards=32,deferred_draft_cards=1,
        AR_source_record_reports32_spec33=True,all_arms_full33_normalizations_provided=True,
        target_Slurm_residence_s=2771,draft_Slurm_residence_s=433,
        actual_Slurm_GPU_card_seconds=32*2771+433,
        full33capacity_for_target_residency_card_seconds=33*2771,
        useful_non_EOS_tokens_all4arms=total_useful,
        all4tokens_over_actual_Slurm_card_seconds=total_useful/(32*2771+433),
        denominator_includes_cold_target_load_and_draft_init=True,
        scope='One mixed4arm diagnosis, not counterfactual whole-service ARbaseline or onlinegoodput'),
    arms=arms,pairs=pairs,target_metadata_and_original_refs=metadata,
    draft_original=dict(path=q['files'][5]['path'],SHA256=q['files'][5]['SHA256']),
    independent_delayed_post=post,all_resources_released=True,
    limitations=['Two input-length variants of one calculate intent; no full benchmark/no nondrop proof',
        'No extra warmups,onepair perlength; cold firstARprefill confounds retained completewall',
        'Near32k completegeneration not improved; backenddecode does not replace fullModel or servicewall',
        '719FAILED1 and missingfinal_evidence preserved; independent delayed cleanup is a separate observation',
        'Original39947 transient state/owner not recorded, so TIME_WAIT remains a hypothesis',
        'Same327qualified8df diagnostic only; no latest99/ec inherited Modelqualification'],
    decision='Stop this once-only4request scope; root independently audits actualtimelines/costs before any promotion/newGPU.')
p=O/'Kimi364365-run-d-4arm-emitter-phase-audit.json';p.write_bytes((json.dumps(out,indent=2)+'\n').encode())
task=G/'coordination/tasks/Kimi-long-pair-run-d-r364365.json';t=json.loads(task.read_bytes());t.update(state=out['state'],GPU=0,GPU_GO=False,
    actual_draft='910371.672',all_resources_released=True,controller_terminal='FAILED1 post39947 assertion',
    target_and_draft_GPU_terminal='COMPLETED0',final_evidence_generated_by_original_controller=False,
    terminal_audit=str(p.relative_to(G)).replace('\\','/'),new_GPU_or_target_reload_authorized=False)
task.write_bytes((json.dumps(t,indent=2)+'\n').encode())
checkpoint=D/'RUN_D_CHECKPOINT.json';c=json.loads(checkpoint.read_bytes());c.update(state=out['state'],all_resources_released=True,
    actual_draft='910371.672',next='No active handles to resume/replay. Rootaudit terminal_audit; no newGPU authorized.',terminal_audit=t['terminal_audit'])
checkpoint.write_bytes((json.dumps(c,indent=2)+'\n').encode())
print(json.dumps(dict(audit=str(p),pairs=pairs,bootstrap_s=out['clocks']['one_time_draft_connect_load_handshake_local_s'],all5clean=True)))
