"""Occurrence-aware attention envelopes; no exclusive GPU/kernel/provider inference."""
import math
import re

PAIRS = frozenset((
    'ids.H2D', 'prepare.forward_latent', 'fresh.expand_cached', 'selection.D2H',
    'selection.host_remap', 'selection.ids_alloc_hold_H2D', 'history.selected_value',
    'gather.key', 'gather.value_conditional', 'attention.certified', 'attention.selected',
    'attention.result_copy', 'output.forward', 'output.original_stage_finish_or_stream_wait',
    'projection.original_success_fence', 'prepare.latent_original_stream_wait',
    'expand.original_stream_wait', 'output.quant', 'output.scale_scatter',
    'output.projection', 'output.copy', 'output.allreduce', 'output.original_success_fence',
))
BRANCHES = frozenset(('selection.complete_prefix', 'selection.partial_prefix', 'v398.enabled',
                      'v398.disabled', 'history.dispatch.packed', 'history.dispatch.standard'))
TUPLE = re.compile(r'(projection|output)\.tuple\.M(\d+)\.N(\d+)\.K(\d+)\.row_offset(\d+)'
                   r'\.column_offset(\d+)\.input_BF16\.contiguous_checked\.canonical_stride_inferred_'
                   r'(\d+)_1\.raw_stride_UNKNOWN\.kernel_provider_UNKNOWN\Z')
DISPATCH = re.compile(r'(projection|output)\.dispatch\.(cached_r809|fallback_Fp8BlockLinear)'
                      r'\.kernel_provider_UNKNOWN\Z')


def require(test, message):
    if not test:
        raise ValueError(message)


def validate_tuple(site, m, n, k, row, column, stride, frame, global_rank, stack, branches):
    require(type(global_rank) is int and 0 <= global_rank < 16, 'attention global rank differs')
    require(m > 0 and n > 0 and k > 0 and stride == k, 'attention canonical tuple differs')
    history = site == 'projection' and any(stem == 'history.selected_value' for stem, _ in stack)
    if history:
        require('selection.complete_prefix' in branches and 'selection.partial_prefix' not in branches
                and 'v398.enabled' in branches and 'v398.disabled' not in branches
                and 'history.dispatch.standard' in branches and 'history.dispatch.packed' not in branches,
                'history tuple requires observed complete-prefix standard v398 branch')
        rows = frame['row_past_count']
        require(frame['tokens'] == 1 and len(rows) == 1 and rows[0][0] == 0 and rows[0][2] == 1
                and type(rows[0][1]) is int and 0 < rows[0][1] < 8192,
                'history projection requires bound trueT1 prefix within one frozen8192 block')
        require(m == rows[0][1] + 1 and n == 1792 and k == stride == 512
                and row == global_rank * 1792 and column == 0,
                'history projection tuple differs from frozen selected-prefix contract')
    else:
        require(m == 1, 'nonhistory attention projection must retain M1')
    return history


def validate_layout(tuples, dispatches, counts, layer, rank, past):
    # Exact frozen C1 complete-prefix path, not a general projection allowance.
    expected=[('projection',1,2048,6144,0,0,'attention_prepare'),
              ('projection',1,1024,2048,rank*1024,0,'attention_prepare'),
              ('projection',1,576,6144,0,0,'attention_prepare')]
    if layer==0:
        expected += [('projection',1,128,6144,0,0,'attention_indexer'),
                     ('projection',1,256,2048,rank*256,0,'attention_indexer')]
    expected += [('projection',1,1792,512,rank*1792,0,'attention_fresh'),
                 ('projection',past+1,1792,512,rank*1792,0,'attention_history'),
                 ('output',1,6144,1024,0,rank*1024,'attention_output')]
    n=768 if layer==0 else 128
    expected += [('projection',1,n,6144,rank*n,0,'FFN_gate_up')]*2
    actual=[tuple(t[k] for k in ('site','M','N','K','row_offset','column_offset','caller_phase')) for t in tuples]
    require(actual==expected, 'exact frozen projection caller sequence or geometry differs')
    require([d['site'] for d in dispatches]==[t[0] for t in expected], 'exact frozen dispatch sequence differs')
    require(counts.get('projection.original_success_fence')==(9 if layer==0 else 7),
            'exact generic projection fence count differs')


def attention_windows(frame, global_rank):
    require(type(global_rank) is int and 0 <= global_rank < 16 and frame['rank'] == global_rank % 8, 'attention local/global rank differs')
    require([frame['full_detail_layer'], frame['shared_detail_layer']] == [0, 3]
            and frame['event_count'] == len(frame['marks']) <= 440, 'attention layer/event identity differs')
    names = [mark['name'] for mark in frame['marks']]
    begins = [(int(match[1]), index) for index, name in enumerate(names)
              if (match := re.fullmatch(r'layer\.(\d+)\.(?:full|shared)\.begin', name))]
    ends = [(int(match[1]), index) for index, name in enumerate(names)
            if (match := re.fullmatch(r'layer\.(\d+)\.end', name))]
    require([layer for layer, _ in begins] == list(range(78))
            and [layer for layer, _ in ends] == list(range(78))
            and all(begins[layer][1] < ends[layer][1] < begins[layer+1][1] for layer in range(77))
            and begins[-1][1] < ends[-1][1], 'incomplete/duplicate/misordered78-layer frame')
    selected = []
    for mark_index, mark in enumerate(frame['marks']):
        if '.attnd1.' not in mark['name']:
            continue
        match = re.fullmatch(r'detail\.layer\.(0|3)\.attnd1\.(.+)', mark['name'])
        require(match is not None, 'attention detail emitted outside selected0/3')
        layer = int(match.group(1))
        require(begins[layer][1] < mark_index < ends[layer][1],
                'attention detail marker lies outside its named layer window')
        selected.append((layer, match.group(2), mark, mark_index))
    require(0 < len(selected) <= 162, 'missing or over-budget new attention marks')
    reports = []
    for layer in (0, 3):
        events = [(name, mark, index) for current, name, mark, index in selected if current == layer]
        def anchor(suffix):
            full=f'detail.layer.{layer}.'+suffix
            require(names.count(full)==1, 'missing or duplicate caller phase anchor')
            return names.index(full)
        ffn_begin, gate_end, ffn_end = (anchor(x) for x in ('FFN.begin','FFN.local.gate_up.return','FFN.end'))
        require(ffn_begin < gate_end < ffn_end, 'caller FFN anchor ordering differs')
        ib,ie=(anchor('attn.indexer.features.begin'),anchor('attn.indexer.features.return')) if layer==0 else (-1,-1)
        if layer==0:require(ib<ie<ffn_begin,'caller indexer anchor ordering differs')
        def phase(index, stack):
            if ffn_begin<index<gate_end:return 'FFN_gate_up'
            require(index<ffn_begin,'generic marker outside supported caller phases')
            for stem,label in (('prepare.forward_latent','attention_prepare'),('fresh.expand_cached','attention_fresh'),('history.selected_value','attention_history'),('output.forward','attention_output')):
                if any(s==stem for s,_ in stack):return label
            if ib<index<ie:return 'attention_indexer'
            return 'attention_outer'
        fence_phases=[]
        stack, windows, tuples, dispatches, branches = [], [], [], [], []
        counts = {}
        for name, mark, mark_index in events:
            host, stream = mark['owner_host_s'], mark['gpu_from_first_s']
            require(all(math.isfinite(x) and x >= 0 for x in (host, stream)), 'invalid attention clocks')
            if name.endswith('.begin') or name.endswith('.return'):
                stem, edge = name.rsplit('.', 1)
                require(stem in PAIRS, 'unknown attention window marker')
                if edge == 'begin':
                    if stem=='projection.original_success_fence':
                        require(tuples and phase(mark_index,stack)==tuples[-1]['caller_phase'], 'projection fence caller phase differs')
                        fence_phases.append(tuples[-1]['caller_phase'])
                    stack.append((stem, mark))
                else:
                    require(stack and stack[-1][0] == stem, 'missing/misordered attention occurrence')
                    _, begin = stack.pop()
                    occurrence = counts.get(stem, 0)
                    counts[stem] = occurrence+1
                    upper = (9 if layer==0 else 7) if stem == 'projection.original_success_fence' else 2 if stem == 'expand.original_stream_wait' else 1
                    require(counts[stem] <= upper, 'attention occurrence exceeds static C1 bound')
                    host_delta = host-begin['owner_host_s']
                    stream_delta = stream-begin['gpu_from_first_s']
                    require(host_delta >= 0 and stream_delta >= -2e-6, 'inverted attention window')
                    caller_phase = phase(mark_index, stack)
                    if stem=='projection.original_success_fence':
                        require(caller_phase==fence_phases[occurrence], 'projection fence crosses caller phase')
                    windows.append(dict(name=stem, occurrence=occurrence, caller_phase=caller_phase, host_envelope_seconds=host_delta,
                                        stream_envelope_seconds=stream_delta if stream_delta >= 0 else None))
            elif name in BRANCHES:
                branches.append(name)
            elif TUPLE.fullmatch(name):
                match = TUPLE.fullmatch(name)
                site, m, n, k, row, column, stride = match.groups()
                validate_tuple(site, int(m), int(n), int(k), int(row), int(column), int(stride), frame, global_rank, stack, branches)
                tuples.append(dict(site=site, occurrence=sum(x['site'] == site for x in tuples),
                                   M=int(m), N=int(n), K=int(k), row_offset=int(row), column_offset=int(column), caller_phase=phase(mark_index,stack),
                                   input='BF16_contiguous_checked', canonical_stride_inferred=[int(k), 1],
                                   raw_stride='UNKNOWN', actual_kernel_provider='UNKNOWN'))
            elif DISPATCH.fullmatch(name):
                match = DISPATCH.fullmatch(name)
                dispatches.append(dict(site=match[1], observed_branch=match[2], actual_kernel_provider='UNKNOWN'))
            else:
                raise ValueError('unknown attention detail metadata marker')
        require(not stack, 'unfinished attention window')
        mandatory = ['prepare.forward_latent', 'fresh.expand_cached', 'attention.result_copy',
                     'output.forward', 'output.original_stage_finish_or_stream_wait']
        require(all(counts.get(name) == 1 for name in mandatory)
                and counts.get('attention.selected', 0)+counts.get('attention.certified', 0) == 1,
                'missing or duplicate mandatory attention windows/branch')
        for site in ('projection', 'output'):
            nt = sum(x['site'] == site for x in tuples)
            nd = sum(x['site'] == site for x in dispatches)
            require(nt == nd and (nt == (9 if layer==0 else 7) if site == 'projection' else nt == 1),
                    'projection tuple/dispatch occurrence count differs')
        validate_layout(tuples, dispatches, counts, layer, global_rank, frame['row_past_count'][0][1])
        require(fence_phases==[t['caller_phase'] for t in tuples if t['site']=='projection'], 'projection fence phase sequence differs')
        for options in [('selection.complete_prefix', 'selection.partial_prefix'),
                        ('v398.enabled', 'v398.disabled'), ('history.dispatch.packed', 'history.dispatch.standard')]:
            require(sum(branches.count(name) for name in options) <= 1, 'contradictory/duplicate attention branch')
        reports.append(dict(layer=layer, new_marks=len(events), windows=windows, tuples=tuples,
                            dispatches=dispatches, observed_branches=branches,
                            missing_branch_is_not_zero_cost=True, exclusive_GPU_time=False,
                            residual_between_nested_return_and_outer_return='UNKNOWN_UNATTRIBUTED'))
    return reports
