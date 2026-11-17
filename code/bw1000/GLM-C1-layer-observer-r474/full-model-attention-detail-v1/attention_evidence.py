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


def attention_windows(frame):
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
        selected.append((layer, match.group(2), mark))
    require(0 < len(selected) <= 162, 'missing or over-budget new attention marks')
    reports = []
    for layer in (0, 3):
        events = [(name, mark) for current, name, mark in selected if current == layer]
        stack, windows, tuples, dispatches, branches = [], [], [], [], []
        counts = {}
        for name, mark in events:
            host, stream = mark['owner_host_s'], mark['gpu_from_first_s']
            require(all(math.isfinite(x) and x >= 0 for x in (host, stream)), 'invalid attention clocks')
            if name.endswith('.begin') or name.endswith('.return'):
                stem, edge = name.rsplit('.', 1)
                require(stem in PAIRS, 'unknown attention window marker')
                if edge == 'begin':
                    stack.append((stem, mark))
                else:
                    require(stack and stack[-1][0] == stem, 'missing/misordered attention occurrence')
                    _, begin = stack.pop()
                    occurrence = counts.get(stem, 0)
                    counts[stem] = occurrence+1
                    upper = 7 if stem == 'projection.original_success_fence' else 2 if stem == 'expand.original_stream_wait' else 1
                    require(counts[stem] <= upper, 'attention occurrence exceeds static C1 bound')
                    host_delta = host-begin['owner_host_s']
                    stream_delta = stream-begin['gpu_from_first_s']
                    require(host_delta >= 0 and stream_delta >= -2e-6, 'inverted attention window')
                    windows.append(dict(name=stem, occurrence=occurrence, host_envelope_seconds=host_delta,
                                        stream_envelope_seconds=stream_delta if stream_delta >= 0 else None))
            elif name in BRANCHES:
                branches.append(name)
            elif TUPLE.fullmatch(name):
                match = TUPLE.fullmatch(name)
                site, m, n, k, row, column, stride = match.groups()
                require(int(m) == 1 and int(n) > 0 and int(k) > 0 and int(stride) == int(k),
                        'attention canonical tuple differs')
                tuples.append(dict(site=site, occurrence=sum(x['site'] == site for x in tuples),
                                   M=int(m), N=int(n), K=int(k), row_offset=int(row), column_offset=int(column),
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
            require(nt == nd and (1 <= nt <= 7 if site == 'projection' else nt == 1),
                    'projection tuple/dispatch occurrence count differs')
        for options in [('selection.complete_prefix', 'selection.partial_prefix'),
                        ('v398.enabled', 'v398.disabled'), ('history.dispatch.packed', 'history.dispatch.standard')]:
            require(sum(branches.count(name) for name in options) <= 1, 'contradictory/duplicate attention branch')
        reports.append(dict(layer=layer, new_marks=len(events), windows=windows, tuples=tuples,
                            dispatches=dispatches, observed_branches=branches,
                            missing_branch_is_not_zero_cost=True, exclusive_GPU_time=False,
                            residual_between_nested_return_and_outer_return='UNKNOWN_UNATTRIBUTED'))
    return reports
