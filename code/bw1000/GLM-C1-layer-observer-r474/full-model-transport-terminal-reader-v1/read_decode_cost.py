"""Derive phase costs from the original GLM Socket/SHCA terminal evidence."""

import hashlib
import json
import math
from pathlib import Path
import statistics


LABEL = 'GLM474-pair-6dae70744fdc40e3ba0ade61a5b1143c'
AUDIT_SHA = '689103ffd834ea85bf28f185a8403c3d43627932e5e55e4f15cb993b0d4afcad'
BINDING_SHA = '81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e'
GATES = {'socket': '1103606.102', 'shca': '1103606.105'}
PHASES = ('incoming_host_s', 'layers_host_s', 'outgoing_host_s',
          'incoming_stream_s', 'layers_stream_s', 'outgoing_stream_s')


def read(path, digest, maximum):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('frozen artifact differs: ' + path.name)
    return json.loads(raw)


def valid_seconds(values):
    return all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values)


def decode_cell(request, pin):
    if (request['diagnostic'] is not False or request['completed'] is not True
            or request['reset_before'] is not True or request['reset_after'] is not True
            or any(request[k] != pin[k] for k in ('ordinal', 'case_id', 'suite',
                                                  'prompt_IDs_SHA256', 'body_SHA256'))):
        raise ValueError('invalid frozen screen request')
    ids = request['generated_token_ids']
    n = len(ids) - 1
    digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    values = request['decode_CAPI_s']
    bucket = request['pipeline_decode']
    if (digest != request['generated_token_ids_SHA256'] or not 0 <= n < 4096
            or len(values) != n or tuple(request['pipeline_profile_names']) != PHASES
            or request['Native_forward_end'] - request['Native_forward_begin']
            != n + len(request['prefill_CAPI_s']) or not valid_seconds(values)):
        raise ValueError('invalid original decode counts or output hash')
    if n == 0:
        if bucket:
            raise ValueError('unexpected profile for zero decode calls')
        return digest, n, None
    sums = bucket['sum_s']
    if bucket['count'] != n or len(sums) != len(PHASES) or not valid_seconds(sums):
        raise ValueError('invalid original decode phase profile')
    return digest, n, [(v / n) * 1000 for v in (math.fsum(values), *sums)]


def read_arm(root, arm, item, pins):
    if (item['accounting']['gate'][:4] != [GATES[arm], LABEL + '-' + arm + '-gate', 'COMPLETED', '0:0']
            or not all(item.get(k) is True for k in ('post_verified', 'all_rank_normal_close', 'transport_verified'))
            or set(item['owners']) != set(map(str, range(16)))):
        raise ValueError('incomplete frozen transport arm')
    cells = []
    for rank in range(16):
        path = root / (LABEL + '-' + arm) / f'owner-{rank:02d}.json'
        owner = read(path, item['owners'][str(rank)]['SHA256'], 96 * 1024**2)
        if (owner['global_rank'], owner['actual_GPU'], owner['Native_arm']) != (rank, GATES[arm], arm):
            raise ValueError('wrong original owner identity')
        screen = [v for v in owner['requests'] if v['diagnostic'] is False]
        if len(screen) != 32 or len(owner['requests']) != 33 or {v['ordinal'] for v in screen} != set(pins):
            raise ValueError('incomplete original request inventory')
        cells.append({v['ordinal']: decode_cell(v, pins[v['ordinal']]) for v in screen})
    if any(len({rank[k][:2] for rank in cells}) != 1 for k in pins):
        raise ValueError('generated outputs differ across ranks')
    return cells


def distributions(rows):
    return dict(median=statistics.median(rows), minimum=min(rows), maximum=max(rows))


def audit(root):
    run = read(root / (LABEL + '-independent-terminal-audit-v1.json'), AUDIT_SHA, 65536)
    if run['complete_engineering_pair'] is not True or run['run_id'] != LABEL.removeprefix('GLM474-pair-'):
        raise ValueError('original engineering audit identity differs')
    binding = read(root / (LABEL + '-dispatch') / 'src' / 'INPUT_BINDING.json', BINDING_SHA, 1 << 20)
    pins = {v['ordinal']: v for v in binding['private_plan']['requests']}
    if len(pins) != 32:
        raise ValueError('frozen screen count differs')
    a, b = (read_arm(root, k, run['arms'][k], pins) for k in GATES)
    match = [k for k in pins if a[0][k][0] == b[0][k][0]]
    pairs = [k for k in match if a[0][k][1] > 0]
    if not pairs:
        raise ValueError('no matching request with decode work')
    result = {}
    for rank_group, ranks in (('all', range(16)), ('lower_global_ranks', range(8)),
                              ('upper_global_ranks', range(8))):
        group = {}
        for i, phase in enumerate(('forward_enclosure_s', *PHASES)):
            left = [a[rank][key][2][i] for rank in ranks for key in pairs]
            right = [b[rank][key][2][i] for rank in ranks for key in pairs]
            group[phase.removesuffix('_s')] = dict(
                socket_ms_per_call=distributions(left), shca_ms_per_call=distributions(right),
                shca_minus_socket_ms_per_call=distributions([y - x for x, y in zip(left, right)]))
        result[rank_group] = group
    return dict(terminal_audit_SHA256=AUDIT_SHA, steps=GATES, planned_pairs=len(pins),
                same_output_pairs=len(match), missing_or_mismatched_output_pairs=len(pins)-len(match),
                same_output_zero_decode_pairs=len(match)-len(pairs), measured_pairs=len(pairs),
                rank_request_cells=len(pairs)*16, forwards_per_arm=16*sum(a[0][k][1] for k in pairs),
                cost=result, fixed_order=True, exclusive_GPU_time=False, service_qualified=False)


if __name__ == '__main__':
    import sys

    print(json.dumps(audit(Path(sys.argv[1])), allow_nan=False))
