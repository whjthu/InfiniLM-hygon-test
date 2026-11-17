"""Check the same-prompt observer-on/off requests from the r474 GLM owner."""

import hashlib
import json
import math
from attention_evidence import attention_windows

MARKS = (
    'FFN.local.gate_up.return',
    'FFN.local.activation.end',
    'FFN.local.down.begin',
    'FFN.local.down.before_original_fence',
    'FFN.local.down.after_original_fence',
    'FFN.tp.copy.begin',
    'FFN.tp.allreduce.begin',
    'FFN.tp.allreduce.return',
    'FFN.out.before_original_fence',
    'FFN.out.after_original_fence',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_frame(frame, diagnostic, rank, index):
    marks = frame['marks']
    require(frame['rank'] == rank % 8 and frame['tokens'] == 1
            and frame['stage'] == 0 and frame['row_past_count'] == [[0, diagnostic['prompt_tokens'] + index, 1]]
            and frame['sequence'] == len(diagnostic['prefill_fragment_plan']) + index + 1,
            'wrong FFN diagnostic frame')
    require(157 <= len(marks) == frame['event_count'] <= 440, 'wrong FFN event bound')
    names = [m['name'] for m in marks]
    require(names[0] == 'CAPI.owner.begin', 'wrong FFN clock origin')
    host = [m['owner_host_s'] for m in marks]
    stream = [m['gpu_from_first_s'] for m in marks]
    require(all(math.isfinite(x) and x >= 0 for x in host + stream)
            and abs(stream[0]) <= 2e-6
            and all(a <= b for a, b in zip(host, host[1:]))
            and all(a <= b + 2e-6 for a, b in zip(stream, stream[1:])), 'invalid FFN event clocks')
    observed = []
    selected = [frame['full_detail_layer'], frame['shared_detail_layer']]
    require(len(set(selected)) == 2 and all(type(i) is int and 0 <= i < 78 for i in selected),
            'wrong detailed layer selection')
    require(sum(any(name.endswith('.' + mark) for mark in MARKS) for name in names) == 2 * len(MARKS),
            'unexpected number of FFN split marks')
    for layer, kind in zip(selected, ('full', 'shared')):
        prefix = f'detail.layer.{layer}.'
        expected = (f'layer.{layer}.{kind}.begin', prefix + 'FFN.begin',
                    *(prefix + name for name in MARKS),
                    prefix + 'FFN.end', f'layer.{layer}.end')
        expected_set = set(expected)
        require(tuple(n for n in names if n in expected_set) == expected,
                f'missing, duplicate or misordered FFN split on layer {layer}')
        observed.append(layer)
    attention_windows(frame, rank)
    return observed


INDEXER_WINDOWS = {
    'features': ('attn.indexer.features.begin', 'attn.indexer.features.return'),
    'scored_selector': ('attn.indexer.selector.forward.begin', 'attn.indexer.selector.forward.return'),
}


def indexer_windows(frame):
    """Return measurements only for split call pairs seen on the full attention layer."""
    marks = frame['marks']
    layer = frame['full_detail_layer']
    prefix = f'detail.layer.{layer}.'
    expected = [prefix + suffix for pair in INDEXER_WINDOWS.values() for suffix in pair]
    split_suffixes = tuple('.' + suffix for pair in INDEXER_WINDOWS.values() for suffix in pair)
    actual = [m for m in marks if m['name'].endswith(split_suffixes)]
    labels = [m['name'] for m in actual]
    require(labels in ([], expected[:2], expected), 'missing, foreign, duplicate or misordered indexer split markers')
    if labels:
        positions = [i for i, m in enumerate(marks) if m['name'] in labels]
        names = [m['name'] for m in marks]
        anchors = (f'layer.{layer}.full.begin', prefix + 'FFN.begin')
        require(all(names.count(anchor) == 1 for anchor in anchors)
                and names.index(anchors[0]) < positions[0] <= positions[-1] < names.index(anchors[1]),
                'indexer marks outside full attention layer window')
    windows = {}
    for i, name in enumerate(INDEXER_WINDOWS):
        if len(actual) < 2 * (i + 1):
            windows[name] = None
            continue
        begin, end = actual[2*i:2*i+2]
        host_s = end['owner_host_s'] - begin['owner_host_s']
        stream_s = end['gpu_from_first_s'] - begin['gpu_from_first_s']
        require(math.isfinite(host_s) and math.isfinite(stream_s) and host_s >= 0 and stream_s >= -2e-6,
                'invalid indexer window clocks')
        windows[name] = dict(host_s=host_s, enclosing_stream_s=stream_s if stream_s >= 0 else None)
    return dict(layer=layer, past=frame['row_past_count'][0][1], **windows)
