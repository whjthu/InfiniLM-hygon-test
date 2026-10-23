"""Resident full-model quality and serving matrix; no model/weight mutations."""
import ctypes
import hashlib
import json
import os
import time
import shutil

import numpy as np
import torch


def run(g):
    r, tag, lib, ctx, cfg, forward, check, report, path = [g[k] for k in
        ['r', 'tag', 'lib', 'ctx', 'cfg', 'forward', 'check', 'report', 'path']]
    policy = lib.modern_deep_precision_policy_r3443
    policy.argtypes = [ctypes.c_int]
    os.environ.pop('INFINILM_DEEP_LAYER_TRACE_R4026', None)
    g['projection_tile_mode'] = 0
    g['c4_tile_mode'] = 0
    g['precision_mode'] = 5
    check(policy(5))
    g['layout_mode']=0;g['HC_mode']=0;g['tail_SUM_mode']=0;check(lib.deep_moe_tail_set_r7211(0));check(lib.modern_deep_layout_policy_r7052(0));check(lib.deep_hc_canonical_policy_r7130(0));g['run_chat']('before')
    check(policy(5));g["precision_mode"]=5
    natural = json.loads((r / 'results' / f'deep-serving-cases-r2130-{tag}.json').read_text())
    audit = json.loads((r / 'results/deep-attention-r3511-B8-audit.json').read_text())
    ref_path = r / 'results' / f'deep-attention-r3509-{audit["producer"][0]}.json'
    reference = json.loads(ref_path.read_text())
    assert [x['input_ids'] for x in natural['cases']] == [x['input_ids'] for x in reference['natural_input_provenance']['cases']]
    frozen = next(c for c in reference['attention_F64_cases'] if c['phase'] == 'natural-l0-r1')['generated_ids']
    assert len(frozen) == 8 and all(len(x) == 512 for x in frozen)
    eos = cfg['eos_token_id']
    qa_ref = json.loads((r / 'results/native-engine-allselected-r816-910373.442.json').read_text())
    qa_gold = {x['phase'].rsplit('_repeat', 1)[0]: x['generated_ids'] for x in qa_ref['chat_cases_by_epoch']['before']}
    qa_prompts = {x['label']: x['input_ids'] for x in g['chat_payload']['cases']}
    assert {'arithmetic', 'retrieval', 'beijing'} <= set(qa_prompts) & set(qa_gold)
    report.update(serving_cases_r4842=[], teacher_OFF_diagnostic_r4842=[],
        reference_file=str(ref_path), reference_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        serving_whole_candidate_qualified=False, performance_adopted=False,
        scope='Exact e8f0 parent plus strict past0 single512 MoE SUM padding2048 2TU candidate once-load; OFF layout0HC0tail0 vs ON layout3HC1tail1. T1/T16 decode bypasses new tail path. PureB32 historical128 anchor, candidate mixed33 andB1 full128 IDs/EOS plusQA816, thenonlyifpassed pureB32 ABBA profile. B1 independenthistoricalgold unavailable; crossB3509 gold unchanged. Rejected outputs neverTPS; allintermediate bitsdiagnosticonly.',
        source_sha256={n: hashlib.sha256((r / 'incoming' / n).read_bytes()).hexdigest() for n in
            ['modern_Deep_tailSUM_full_r7224.py', 'Deep_tailSUM_full_matrix_r7224.py', 'Deep_tailSUM_full_r7224_inside.sh']})

    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        tmp.replace(path)

    def sync():
        for device in range(8):
            torch.cuda.synchronize(device)

    def invoke(inputs, rows, past):
        widths = list(map(len, inputs))
        positions = sum([list(range(p, p + width)) for p, width in zip(past, widths)], [])
        tick = time.perf_counter()
        code, logits, ids = forward(sum(inputs, []), positions, rows, past,
            [p + width for p, width in zip(past, widths)], [0] + np.cumsum(widths).tolist())
        check(code)
        elapsed = time.perf_counter() - tick
        free = [torch.cuda.mem_get_info(device)[0] for device in range(8)]
        report['minimum_driver_free_after_forward_sampled_not_peak'] = min(
            report.get('minimum_driver_free_after_forward_sampled_not_peak', min(free)), min(free))
        assert min(free) >= 6 * 1024**3, ('Model_free_reserve_6GiB', free)
        values = (logits.astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(values).all() and values.argmax(-1).tolist() == ids.tolist()
        return ids.tolist(), logits, elapsed

    repeats = {}
    began_matrix = time.perf_counter()

    def c4_snapshot():return [[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]
    csa_layers=sum(int(v)==4 for v in cfg['compress_ratios'][:cfg['num_hidden_layers']])
    def trial(B, M, tile, repeat, maximum=128, mixed=False):
        assert B <= 128 and 0 < M <= 64 and tile in [0, 16]
        g['projection_tile_mode'] = tile
        g['c4_tile_mode'] = 16 if tile==16 else 0
        check(lib.deep_hc_canonical_policy_r7130(g['HC_mode']));assert lib.deep_hc_canonical_get_r7130()==g['HC_mode']
        c4_before=c4_snapshot()
        check(lib.modern_native_engine_reset(ctx))
        prompts = [natural['cases'][i % 2]['input_ids'] for i in range(B)]
        gold = [frozen[i % 8][:maximum] for i in range(B)]
        families = [f'natural-{i % 2}' for i in range(B)]
        if mixed:
            # At EOS step3, 33 requests shrink to17; at step21 they shrink to16.
            # Keep the sixteen natural requests at original row17..32, preserving
            # row/family reference mapping rather than shifting/reusing caches.
            assert B == 33
            for i in range(17):
                label = 'arithmetic' if i == 0 else 'retrieval' if i <= 15 else 'beijing'
                prompts[i] = qa_prompts[label]
                gold[i] = qa_gold[label][:maximum]
                families[i] = label
        label = f'Layout{g["layout_mode"]}-HC{g["HC_mode"]}-Tail{g["tail_SUM_mode"]}-B{B}-M{M}-tile{tile}-r{repeat}-max{maximum}-mixed{int(mixed)}'
        report['phase'] = label
        save()
        generated = [[] for _ in range(B)]
        timestamps = [[] for _ in range(B)]
        events = []
        snapshots = []
        active = list(range(B))
        sync()
        start = time.perf_counter()
        prefill_end = None
        for step in range(maximum):
            width = 4 if step == 0 else M
            for begin in range(0, len(active), width):
                rows = active[begin:begin + width]
                inputs = [prompts[i] if step == 0 else [generated[i][-1]] for i in rows]
                past = [0 if step == 0 else len(prompts[i]) + len(generated[i]) - 1 for i in rows]
                selected, logits, elapsed = invoke(inputs, rows, past)
                stamp = time.perf_counter() - start
                for row, token in zip(rows, selected):
                    generated[row].append(token)
                    timestamps[row].append(stamp)
                events.append({'step': step, 'rows': rows, 'input_tokens': sum(map(len, inputs)),
                    'CAPI_wall_seconds': elapsed, 'emission_seconds': stamp,
                    'tile_applied': bool(tile == 16 and step > 0 and 16 < len(rows) <= 64),'C4_applied':bool(g['c4_tile_mode']==16 and step>0 and 16<len(rows)<=64)})
                if step in [0, 25, maximum - 1]:
                    # Hashes retain output provenance without timed disk I/O or
                    # accumulating full vocabulary tensors in host memory.
                    snapshots.append({'step': step, 'rows': rows,
                        'shape': list(logits.shape), 'sha256': hashlib.sha256(logits.tobytes()).hexdigest()})
            if step == 0:
                sync()
                prefill_end = time.perf_counter() - start
            active = [i for i in active if generated[i][-1] != eos]
            if not active:
                break
        sync()
        end = time.perf_counter() - start
        assert prefill_end is not None
        assert all(v and all(x != eos for x in v[:-1]) and (v[-1] == eos or len(v) == maximum) for v in generated)
        frozen_exact = generated == gold
        key = (g['layout_mode'],g['HC_mode'],g['tail_SUM_mode'], B, M, tile, maximum, mixed)
        repeat_exact = generated == repeats.setdefault(key, generated)
        common_anchor = max(t[0] for t in timestamps)
        decode_end = max(t[-1] for t in timestamps)
        common_decode_tokens = sum(token != eos and stamp > common_anchor
            for ids, stamps in zip(generated, timestamps) for token, stamp in zip(ids, stamps))
        useful = sum(x != eos for ids in generated for x in ids)
        user = [sum(x != eos for x in ids[1:]) / (stamps[-1] - stamps[0])
            if len(stamps) > 1 and stamps[-1] > stamps[0] else None
            for ids, stamps in zip(generated, timestamps)]
        mismatches = []
        for row, (actual, expected) in enumerate(zip(generated, gold)):
            if actual != expected:
                first = next((i for i, (x, y) in enumerate(zip(actual, expected)) if x != y), min(len(actual), len(expected)))
                mismatches.append({'row': row, 'family': families[row], 'first_different_token': first,
                    'actual': actual[first] if first < len(actual) else None,
                    'expected': expected[first] if first < len(expected) else None,
                    'actual_length': len(actual), 'expected_length': len(expected)})
        c4_after=c4_snapshot();c4_delta=[[c4_after[d][k]-c4_before[d][k]for k in range(2)]for d in range(8)]
        c4_expected=2*csa_layers*sum(e['C4_applied']for e in events)
        assert c4_delta==[[c4_expected]*2 for d in range(8)],('C4_case_counter_total',label,c4_delta,c4_expected)
        result = {'C4_compressor_tile':g['c4_tile_mode'],'C4_counter_before':c4_before,'C4_counter_after':c4_after,'C4_counter_delta':c4_delta,'C4_expected_per_kind':c4_expected,'phase': label, 'batch': B, 'micro': M, 'projection_tile': tile,
            'repeat': repeat, 'max_output_tokens': maximum, 'mixed_EOS': mixed,
            'input_ids': prompts, 'input_lengths': list(map(len, prompts)), 'families': families,
            'generated_ids': generated, 'timestamps': timestamps, 'events': events,
            'full_frozen_IDs_exact': frozen_exact, 'repeat_IDs_exact': repeat_exact,
            'mismatches': mismatches, 'finite_CPUargmax_every_forward': True,
            'no_forward_after_EOS': True, 'all_service_cards': 8,
            'prefill_seconds': prefill_end, 'prefill_input_tokens_s': sum(map(len, prompts)) / prefill_end,
            'TTFT_seconds': [x[0] for x in timestamps], 'end_to_end_seconds': end,
            'common_decode_anchor_seconds': common_anchor, 'decode_seconds': decode_end - common_anchor,
            'common_useful_decode_tokens': common_decode_tokens,
            'valid_decode_window':decode_end>common_anchor,'aggregate_useful_decode_tok_s': common_decode_tokens / (decode_end - common_anchor) if decode_end>common_anchor else None,
            'useful_output_tokens': useful, 'including_prefill_useful_tok_s': useful / end,
            'per_user_decode_tok_s': user, 'output_signatures': snapshots,
            'layout_mode':g['layout_mode'],'HC_mode':g['HC_mode'],'tail_SUM_mode':g['tail_SUM_mode'],'candidate_case_quality_passed': frozen_exact and repeat_exact,'historical_independent_B1_reference':'NOT_AVAILABLE' if B==1 else None,'reference_scope':'3509 B8 complete natural gold crossB for natural rows; mixedQA816 ownreference; not independent B1historicalgold',
            'limits': 'Closed batch, two distinct natural prompts repeated; no queue/HTTP arrival test. Token1 included in prefill/TTFT; decode anchor is last initial token emission. EOS excluded from useful counts. Full source IDs/EOS and all8 normal shutdown still require terminal independent audit.'}
        result['profile_measurement_requested']=bool(g.get('profile_mode',False))
        if not(result['candidate_case_quality_passed']and g.get('profile_mode',False)):
            for k in ['aggregate_useful_decode_tok_s','including_prefill_useful_tok_s','per_user_decode_tok_s','prefill_input_tokens_s']:result[k]=None
        report['serving_cases_r4842'].append(result)
        save()
        print('DEEP_PROJECTION_SERVING_R4842_CASE', label, 'IDs', frozen_exact,
            'decode', result['aggregate_useful_decode_tok_s'], 'E2E', result['including_prefill_useful_tok_s'], flush=True)
        # Numeric gold rejection recorded above; keep controlled restores with same frozen gold. GPU/CAPI/finite/ownership errors remain fatal.
        return result


    def switch(layout,hc,tail):
        sync();check(lib.modern_deep_layout_policy_r7052(layout));check(lib.deep_hc_canonical_policy_r7130(hc));check(lib.deep_moe_tail_set_r7211(tail));assert lib.deep_moe_tail_get_r7211()==tail;g['layout_mode']=layout;g['HC_mode']=hc;g['tail_SUM_mode']=tail
    switch(0,0,0)
    anchor=trial(32,32,16,0,128,False)
    assert anchor['candidate_case_quality_passed'],('qualifiedpureB32historicalanchor',anchor['mismatches'])
    switch(3,1,1);g['run_chat']('after')
    mixed=trial(33,16,0,0,128,True)
    single=trial(1,8,0,0,128,False)
    candidate_quality=mixed['candidate_case_quality_passed']and single['candidate_case_quality_passed']and report.get('finite_after_passed',False)
    report['candidate_full128_ID_EOS_QA_passed']=candidate_quality
    report['profile_ABBA_skipped']=not candidate_quality
    report['profile_skip_reason']=None if candidate_quality else'Candidate final IDs/EOS/QA rejected; profile forbidden, only OFFrestore control follows'
    if candidate_quality:
        g['profile_mode']=True
        for layout,hc,tail,repeat in [(0,0,0,1),(3,1,1,1),(3,1,1,2),(0,0,0,2)]:
            switch(layout,hc,tail);trial(32,32,16,repeat,128,False)
    else:
        switch(0,0,0);trial(32,32,16,1,128,False)
    switch(0,0,0);g['projection_tile_mode']=0;g['c4_tile_mode']=0;g['precision_mode']=5;check(policy(5));g['run_chat']('after_restore')
    report['historical_independent_B1_reference']='NOT_AVAILABLE'
    report['serving_whole_candidate_qualified']=candidate_quality and all(c['candidate_case_quality_passed']for c in report['serving_cases_r4842'])and all(report.get('finite_'+epoch+'_passed',False)for epoch in ['before','after','after_restore'])
    for c in report['serving_cases_r4842']:
        c['performance_adopted']=False
        if not c['candidate_case_quality_passed']:
            c['rejected_diagnostic_timings']={k:c[k]for k in ['aggregate_useful_decode_tok_s','including_prefill_useful_tok_s','per_user_decode_tok_s']}
            for k in ['aggregate_useful_decode_tok_s','including_prefill_useful_tok_s','per_user_decode_tok_s']:c[k]=None
    report['matrix_wall_seconds']=time.perf_counter()-began_matrix;report['serving_collection_completed_r4842']=True;report['performance_adopted']=False;save()
