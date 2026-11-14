"""Summarize original matched GLM diagnostic frames by layer and detailed mark."""
import math
import re
from pathlib import Path
from statistics import median

from read_decode_cost import AUDIT_SHA, BINDING_SHA, GATES, LABEL, read


def frame_windows(frame):
    marks = frame['marks']
    if len(marks) != frame['event_count'] or not 157 <= len(marks) <= 250:
        raise ValueError('unexpected mark count')
    names = tuple(m['name'] for m in marks)
    host = [m['owner_host_s'] for m in marks]
    stream = [m['gpu_from_first_s'] for m in marks]
    if (names[0] != 'CAPI.owner.begin' or
            any(not math.isfinite(v) or v < 0 for v in host + stream)
            or any(b < a for a, b in zip(host, host[1:]))
            or any(b + 2e-6 < a for a, b in zip(stream, stream[1:]))):
        raise ValueError('invalid original event clock')
    layers = {}
    opened = None
    by_mark = {}
    for i, name in enumerate(names):
        begin = re.fullmatch(r'layer\.(\d+)\.(full|shared)\.begin', name)
        end = re.fullmatch(r'layer\.(\d+)\.end', name)
        if begin:
            if opened is not None or int(begin[1]) != len(layers):
                raise ValueError('unordered layer markers')
            opened = (i, begin[2])
        if end:
            if opened is None or int(end[1]) != len(layers):
                raise ValueError('unmatched layer marker')
            start, kind = opened
            layer = int(end[1])
            layers[layer] = (kind, (host[i]-host[start])*1000, max(0.,stream[i]-stream[start])*1000)
            if layer in (frame['full_detail_layer'], frame['shared_detail_layer']):
                for j in range(start+1, i+1):
                    key = f'{layer}:{names[j-1]} -> {names[j]}'
                    values = by_mark.setdefault(key, [0., 0.])
                    values[0] += (host[j]-host[j-1])*1000
                    values[1] += max(0.,stream[j]-stream[j-1])*1000
            opened = None
    if opened or len(layers) != 78:
        raise ValueError('missing layer boundaries')
    return names, layers, by_mark


def paired_stats(rows):
    return {axis: dict(socket_ms=median(a[i] for a,b in rows),
                       shca_ms=median(b[i] for a,b in rows),
                       shca_minus_socket_ms=median(b[i]-a[i] for a,b in rows))
            for i, axis in enumerate(('host','stream'))}


def original(root, run, pin, arm, rank):
    path = root / (LABEL + '-' + arm) / f'owner-{rank:02d}.json'
    item = run['arms'][arm]
    owner = read(path, item['owners'][str(rank)]['SHA256'], 96 * 1024**2)
    if (owner['global_rank'],owner['actual_GPU'],owner['Native_arm']) != (rank,GATES[arm],arm):
        raise ValueError('owner identity differs')
    row, = [r for r in owner['requests'] if r['diagnostic'] is True]
    if (row['completed'] is not True or row['reset_after'] is not True
            or any(row[k] != pin[k] for k in ('ordinal','case_id','suite','prompt_tokens',
                                              'prompt_IDs_SHA256','body_SHA256'))):
        raise ValueError('diagnostic input differs')
    report = row['observer_report']
    if (report != owner['observer_diagnostic'] or report['global_tp_rank'] != rank
            or report['total_frames_used'] != len(report['samples']) or len(report['samples']) > 3):
        raise ValueError('diagnostic frame inventory differs')
    for i, frame in enumerate(report['samples']):
        if (frame['rank'] != rank % 8 or frame['tokens'] != 1 or frame['stage'] != 0
                or frame['row_past_count'] != [[0,row['prompt_tokens']+i,1]]
                or frame['sequence'] != len(row['prefill_CAPI_s']) + i + 1):
            raise ValueError('diagnostic frame identity differs')
    return row['generated_token_ids'], report['samples']


def audit(root):
    run = read(root / (LABEL + '-independent-terminal-audit-v1.json'), AUDIT_SHA, 65536)
    if not run['complete_engineering_pair']:
        raise ValueError('paired audit required')
    binding = read(root / (LABEL+'-dispatch')/'src'/'INPUT_BINDING.json', BINDING_SHA, 1 << 20)
    pin = binding['private_plan']['requests'][0]
    layer_rows = {i: [] for i in range(78)}
    detailed = {}
    kinds = {}
    counts = []
    first_ids = None
    for rank in range(16):
        (a_ids,a),(b_ids,b) = [original(root,run,pin,arm,rank) for arm in GATES]
        if a_ids != b_ids or (first_ids is not None and a_ids != first_ids):
            raise ValueError('diagnostic output differs')
        first_ids = a_ids
        if len(a) != len(b):
            raise ValueError('different sample count')
        counts.append(len(a))
        for left, right in zip(a,b):
            names, layers, detail = frame_windows(left)
            other_names, other_layers, other_detail = frame_windows(right)
            if names != other_names or detail.keys() != other_detail.keys():
                raise ValueError('paired marks differ')
            for i,(kind,*span) in layers.items():
                other_kind,*other_span = other_layers[i]
                if kind != other_kind or kinds.setdefault(i,kind) != kind:
                    raise ValueError('paired layer kind differs')
                layer_rows[i].append((span,other_span))
            for key in detail:
                detailed.setdefault(key,[]).append((detail[key],other_detail[key]))
    if not sum(counts):
        raise ValueError('no matching diagnostics')
    return dict(steps=GATES, terminal_audit_SHA256=AUDIT_SHA, frames_per_rank=counts,
                identical_diagnostic_output=True, layers={i:dict(kind=kinds[i],**paired_stats(rows))
                                                          for i,rows in layer_rows.items()},
                detailed_intervals={key:dict(frames=len(rows),**paired_stats(rows)) for key,rows in detailed.items()},
                fixed_order=True, exclusive_GPU_time=False, coverage='single prompt, first decode frames')


if __name__ == '__main__':
    import json
    import sys

    print(json.dumps(audit(Path(sys.argv[1])), allow_nan=False))
