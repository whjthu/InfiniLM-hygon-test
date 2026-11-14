"""Check the same-prompt observer-on/off requests from the r474 GLM owner."""

import hashlib
import json
import math

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
    require(157 <= len(marks) == frame['event_count'] <= 270, 'wrong FFN event bound')
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
    return observed


def audit_pair(diagnostic, control, rank, eos_ids):
    require(type(rank) is int and 0 <= rank < 16, 'wrong global TP rank')
    require(diagnostic['diagnostic'] is True and diagnostic['screen_index'] is None
            and control['diagnostic'] is False and control['screen_index'] == 0
            and diagnostic['observer_diagnostic'] is True and control['observer_diagnostic'] is False,
            'wrong FFN on/off request roles')
    fields = ('ordinal', 'case_id', 'suite', 'prompt_tokens', 'prompt_IDs_SHA256', 'body_SHA256',
              'prefill_fragment_plan', 'runtime_prefill_chunk', 'original_request_prefill_window')
    require(all(diagnostic[k] == control[k] for k in fields)
            and diagnostic['runtime_prefill_chunk'] == 512
            and diagnostic['original_request_prefill_window'] == 4096
            and [(p['offset'], p['tokens']) for p in diagnostic['prefill_fragment_plan']]
            == [(offset, min(512, diagnostic['prompt_tokens'] - offset))
                for offset in range(0, diagnostic['prompt_tokens'], 512)]
            and diagnostic['prompt_tokens'] > 0,
            'FFN on/off inputs differ')
    hashes = []
    for row in (diagnostic, control):
        ids = row['generated_token_ids']
        digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        natural = bool(ids and ids[-1] in eos_ids)
        require(row['completed'] is True and row['reset_before'] is True and row['reset_after'] is True
                and row['planned_max_new_tokens'] == 4096 and 0 < len(ids) <= 4096
                and all(type(i) is int and 0 <= i < 154880 for i in ids)
                and row['generated_token_ids_SHA256'] == digest
                and not any(token in eos_ids for token in ids[:-1])
                and (natural or len(ids) == 4096)
                and row['natural_EOS'] is natural and row['cap_hit'] is (not natural),
                'incomplete FFN request or invalid saved output')
        hashes.append(digest)
    require(hashes[0] == hashes[1] and diagnostic['natural_EOS'] == control['natural_EOS'],
            'FFN observer on/off outputs differ')
    count = min(3, len(diagnostic['generated_token_ids']) - 1)
    report = diagnostic['observer_report']
    require(report['enabled'] is True and report['global_tp_rank'] == rank
            and report['event_cap_per_frame'] == report['events_allocated'] == 512
            and report['frame_cap'] == 3 and report['pending_next_decode'] is False
            and report['total_frames_used'] == count == len(report['samples'])
            and report['coverage'] == [0, 0, 0, count]
            and diagnostic['observer_mark_calls_before'] == 0 and diagnostic['observer_mark_calls_after'] == count
            and control['observer_mark_calls_before'] == control['observer_mark_calls_after'] == count
            and 'observer_report' not in control, 'invalid FFN observer coverage or off request')
    layers = [check_frame(frame, diagnostic, rank, i) for i, frame in enumerate(report['samples'])]
    return dict(rank=rank, matching_outputs=True, output_SHA256=hashes[0], observed_frames=count,
                expected_full_frames=3, full_coverage=count == 3, detailed_layers=layers,
                mark_counts=[frame['event_count'] for frame in report['samples']])


def audit_owners(owners, pin, eos_ids):
    require(len(owners) == 16, 'expected all 16 original owners')
    evidence = []
    for rank, owner in enumerate(owners):
        requests = owner['requests']
        require(owner['global_rank'] == rank and owner['completed'] is True
                and owner['passed'] is True and owner['normal_close'] is True
                and owner['observer_pool_events'] == 512 and owner['observer_source_event_bound'] == 270
                and len(requests) == 33 and len({p['ordinal'] for p in requests[1:]}) == 32,
                'wrong FFN owner inventory or teardown')
        require(all(row[k] == pin[k] for row in requests[:2]
                    for k in ('ordinal', 'case_id', 'suite', 'prompt_tokens', 'prompt_IDs_SHA256', 'body_SHA256')),
                'FFN observer requests differ from original input pin')
        row = audit_pair(requests[0], requests[1], rank, eos_ids)
        count = row['observed_frames']
        require(owner['observer_diagnostic'] == requests[0]['observer_report']
                and all(req['diagnostic'] is False and req['observer_diagnostic'] is False
                        and req['observer_mark_calls_before'] == req['observer_mark_calls_after'] == count
                        and 'observer_report' not in req for req in requests[1:]),
                'FFN observer leaked outside the diagnostic request')
        evidence.append(row)
    require(len({p['output_SHA256'] for p in evidence}) == 1, 'FFN output differs across ranks')
    return dict(planned_ranks=16, planned_frames=48, observed_frames=sum(p['observed_frames'] for p in evidence),
                full_coverage=all(p['full_coverage'] for p in evidence), owners=evidence)
