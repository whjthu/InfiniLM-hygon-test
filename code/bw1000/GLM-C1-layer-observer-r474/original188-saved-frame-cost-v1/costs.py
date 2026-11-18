"""Derive same-owner envelopes, never exclusive device time or GPU busy."""
import hashlib
import json
import math
import re
import statistics
from attention_evidence import attention_windows
from ffn_evidence import MARKS, check_frame, indexer_windows

CLOCK_SCOPE = 'one owner original stream; GPU event intervals overlap host spans'
UNITS = {'owner_host_s': 'seconds', 'gpu_from_first_s': 'seconds'}


def need(test, reason):
    if not test:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def interval(marks, begin, end, units=None):
    """Offsets use each clock's own first mark; never subtract host and event."""
    need((UNITS if units is None else units) == UNITS, 'CLOCK_UNIT')
    need(type(begin) is int and type(end) is int and 0 <= begin < end < len(marks), 'MISSING_OR_INVERTED_MARKER')
    out = {}
    for key in UNITS:
        values = [marks[i][key] for i in (0, begin, end)]
        need(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values), 'NONFINITE_CLOCK')
        delta = values[2] - values[1]
        need(delta >= (0 if key == 'owner_host_s' else -2e-6), 'NEGATIVE_CLOCK_DELTA')
        prefix = 'host' if key == 'owner_host_s' else 'event'
        out[prefix + '_start_s'] = values[1] - values[0]
        out[prefix + '_end_s'] = values[2] - values[0]
        out[prefix + '_envelope_s'] = delta if delta >= 0 else None
    return out


def overlap(a, b, clock):
    need(clock in ('host', 'event') and a['clock_identity'] == b['clock_identity'], 'CROSS_CLOCK_OR_FRAME_OVERLAP')
    if a[clock + '_envelope_s'] is None or b[clock + '_envelope_s'] is None:
        return None
    return max(0, min(a[clock + '_end_s'], b[clock + '_end_s']) - max(a[clock + '_start_s'], b[clock + '_start_s']))


def derive_frame(frame, diagnostic, rank, index):
    need(frame['clock_scope'] == CLOCK_SCOPE, 'CLOCK_DOMAIN')
    if 'units' in frame:
        need(frame['units'] == UNITS, 'CLOCK_UNIT')
    check_frame(frame, diagnostic, rank, index)
    marks = frame['marks']
    names = [m['name'] for m in marks]
    identity = [rank, index, frame['sequence']]
    regions = []

    def add(name, layer, a, b, **metadata):
        row = dict(name=name, layer=layer, clock_identity=identity, **interval(marks, a, b), **metadata)
        regions.append(row)
        return row

    def position(name):
        need(names.count(name) == 1, 'MISSING_OR_DUPLICATE_MARKER')
        return names.index(name)

    add('first_to_last_observed_marker', None, 0, len(marks) - 1)
    for layer in range(78):
        begin = [i for i, n in enumerate(names) if re.fullmatch(r'layer\.' + str(layer) + r'\.(full|shared)\.begin', n)]
        need(len(begin) == 1, 'LAYER_BEGIN')
        add('decoder_layer', layer, begin[0], position('layer.%d.end' % layer), observed_kind=names[begin[0]].split('.')[2])
    attention = attention_windows(frame, rank)
    exposures = []
    for detail in attention:
        layer = detail['layer']
        prefix = 'detail.layer.%d.' % layer
        anchors = ['FFN.begin', *MARKS, 'FFN.end']
        add('FFN.enclosing', layer, position(prefix + anchors[0]), position(prefix + anchors[-1]))
        for left, right in zip(anchors, anchors[1:]):
            add('FFN.segment.' + left + '__' + right, layer, position(prefix + left), position(prefix + right))
        # Pair the exact occurrence stack accepted by the copied qualified parser.
        stack, counts, pairs = [], {}, {}
        for i, name in enumerate(names):
            if not name.startswith(prefix + 'attnd1.'):
                continue
            short = name[len(prefix + 'attnd1.'):]
            if short.endswith('.begin'):
                stack.append((short[:-6], i))
            elif short.endswith('.return'):
                stem = short[:-7]
                need(stack and stack[-1][0] == stem, 'ATTENTION_STACK')
                _, start = stack.pop()
                occurrence = counts.get(stem, 0)
                counts[stem] = occurrence + 1
                pairs[(stem, occurrence)] = (start, i)
        need(not stack and len(pairs) == len(detail['windows']), 'ATTENTION_OCCURRENCE_COUNT')
        for window in detail['windows']:
            row = add('attention.' + window['name'], layer, *pairs[(window['name'], window['occurrence'])],
                      occurrence=window['occurrence'], caller_phase=window['caller_phase'])
            need(row['host_envelope_s'] == window['host_envelope_seconds']
                 and row['event_envelope_s'] == window['stream_envelope_seconds'], 'QUALIFIED_WINDOW_DIFFERS')
        branches = set(detail['observed_branches'])
        history = [t for t in detail['tuples'] if t['caller_phase'] == 'attention_history']
        exposed = bool({'selection.complete_prefix', 'v398.enabled', 'history.dispatch.standard'} <= branches
                       and len(history) == 1 and history[0]['M'] == frame['row_past_count'][0][1] + 1
                       and 1 < history[0]['M'] <= 8192 and history[0]['N'] == 1792
                       and history[0]['K'] == 512 and history[0]['row_offset'] == rank * 1792)
        exposures.append(dict(layer=layer, source_derived_exposure=exposed, direct_call_counter=False,
                              observed_branches=sorted(branches), history_tuples=history))
    indexer = indexer_windows(frame)
    for name, suffixes in (('features', ('attn.indexer.features.begin', 'attn.indexer.features.return')),
                           ('scored_selector', ('attn.indexer.selector.forward.begin', 'attn.indexer.selector.forward.return'))):
        if indexer[name] is not None:
            prefix = 'detail.layer.%d.' % indexer['layer']
            add('indexer.' + name, indexer['layer'], position(prefix + suffixes[0]), position(prefix + suffixes[1]))
    # Only named enclosing/child relations, not a sum of overlapping regions.
    nested = []
    for layer in (0, 3):
        enclosing = next(r for r in regions if r['name'] == 'decoder_layer' and r['layer'] == layer)
        for child in regions:
            if child['layer'] == layer and child is not enclosing:
                nested.append(dict(parent='decoder_layer', child=child['name'], layer=layer,
                                   occurrence=child.get('occurrence'), caller_phase=child.get('caller_phase'),
                                   host_intersection_s=overlap(enclosing, child, 'host'),
                                   event_intersection_s=overlap(enclosing, child, 'event')))
    return dict(global_rank=rank, local_rank=frame['rank'], AR_index=index, sequence=frame['sequence'],
                past=frame['row_past_count'][0][1], marker_count=len(marks), units=UNITS,
                clock_scope=CLOCK_SCOPE, regions=regions, nested_relations=nested, source_exposure=exposures)


def derive_owner(owner, rank):
    need(owner['global_rank'] == rank and len(owner['requests']) == 66, 'OWNER_IDENTITY')
    diagnostic = owner['requests'][1]
    obs = diagnostic['observer_report']
    ids = diagnostic['generated_token_ids']
    count = min(3, max(0, len(ids) - 1))
    need(diagnostic['arm'] == 'ON' and diagnostic['diagnostic'] is True
         and obs['global_tp_rank'] == rank and obs['total_frames_used'] == len(obs['samples']) == count
         and obs['frame_cap'] == 3 and obs['events_allocated'] == 512 and not obs['pending_next_decode'], 'OBSERVER_IDENTITY')
    need(digest(ids) == diagnostic['generated_token_ids_SHA256'], 'DIAGNOSTIC_TOKEN_PIN')
    identities, frames = [], []
    for i, frame in enumerate(obs['samples']):
        need(all(type(t) is int and 0 <= t < 154880 for t in ids[:i+2]), 'CAPTURE_TOKEN_DOMAIN')
        prefix_hash = digest(ids[:i+1])
        if rank == 0:
            matches = [r for r in diagnostic['diagnostic_logits'] if r['kind'] == 'decode'
                       and r['past'] == diagnostic['prompt_tokens'] + i and r['prefix_IDs_SHA256'] == prefix_hash]
            need(len(matches) == 1, 'ACTUAL_AR_CAPTURE_IDENTITY')
        identities.append(dict(AR_index=i, prompt_IDs_SHA256=diagnostic['prompt_IDs_SHA256'],
                               past=diagnostic['prompt_tokens'] + i, prefix_IDs_SHA256=prefix_hash,
                               input_token_SHA256=digest([ids[i]]), output_token_SHA256=digest([ids[i+1]])))
        frames.append(derive_frame(frame, diagnostic, rank, i))
    return dict(rank=rank, frames=frames, capture_identities=identities)


def summarize(owners):
    groups = {}
    for owner in owners:
        for frame in owner['frames']:
            for region in frame['regions']:
                key = (region['name'], region['layer'], region.get('caller_phase'))
                groups.setdefault(key, []).append(region)
    rows = []
    for (name, layer, phase), values in groups.items():
        row = dict(name=name, layer=layer, caller_phase=phase, rank_frame_occurrences=len(values))
        for clock in ('host', 'event'):
            measured = [r[clock + '_envelope_s'] for r in values if r[clock + '_envelope_s'] is not None]
            row[clock + '_envelope_s'] = dict(count=len(measured), unknown=len(values)-len(measured),
                                             min=min(measured) if measured else None,
                                             median=statistics.median(measured) if measured else None,
                                             max=max(measured) if measured else None)
        rows.append(row)
    return sorted(rows, key=lambda r: (r['name'], -1 if r['layer'] is None else r['layer'], r['caller_phase'] or ''))
