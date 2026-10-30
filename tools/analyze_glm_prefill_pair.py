"""Read frozen r241 terminal reports and emit metadata-only paired measurements.

No jobs, HTTP, imports of runtime code, file writes, or automatic adoption.
The archive root must mirror /public/home/aclsylqidf/hygon-inference-refactor-20261014.
Run: python -B tools/analyze_glm_prefill_pair.py --archive-root PATH
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics

PLAN = (1, 2) + (1, 2, 2, 1) * 4
MANIFEST = 'ee1128cece29aeef1637b1ea115397cef3d34828b016f95fc6f561b6f8df2632'
NATIVE = '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
REQUEST = '5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'
REMOTE = pathlib.PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'GLM-prefill-paired-r241'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Timing must be finite and positive')
    return value


def paired_metric(values):
    require(len(values) == 18, 'Require both warmups and sixteen measured requests')
    values = [positive(v) for v in values]
    blocks = []
    for block in range(4):
        a, b, c, d = values[2 + 4 * block:6 + 4 * block]
        # A endpoints surround B. This cancels a linear trend in log time,
        # conditional on that trend model; it does not remove nonlinear drift.
        log_ratio = (math.log(a) + math.log(d) - math.log(b) - math.log(c)) / 2
        blocks.append({'block': block, 'W1_s': [a, d], 'W2_s': [b, c],
                       'W1_over_W2_geometric_ratio': math.exp(log_ratio),
                       'W1_repeat_relative_spread': abs(a - d) / math.sqrt(a * d),
                       'W2_repeat_relative_spread': abs(b - c) / math.sqrt(b * c)})
    ratios = [b['W1_over_W2_geometric_ratio'] for b in blocks]
    ratio = math.exp(statistics.mean(math.log(r) for r in ratios))
    return {'blocks': blocks, 'geometric_W1_over_W2_ratio': ratio,
            'candidate_time_reduction_percent': 100 * (1 - 1 / ratio),
            'block_ratio_min': min(ratios), 'block_ratio_max': max(ratios),
            'blocks_faster_W2': sum(r > 1 for r in ratios),
            'warmups_excluded': 2, 'independent_sample_unit': 'ABBA block; four blocks',
            'confidence_interval': None, 'statistical_significance_claim': False,
            'limitation': 'Fixed order, four blocks; spreads are descriptive, not an independent noise estimate or randomized confidence interval.'}


def summarize(cases, rows):
    require(len(cases) == len(rows) == 18, 'Incomplete pair')
    require([r['pair_index'] for r in rows] == list(range(18)), 'Pair order differs')
    require([r['prefill_control_window'] for r in rows] == list(PLAN), 'Window plan differs')
    require([r['index'] for r in cases] == list(range(18)), 'Client order differs')
    prefill, decode, wall, http, lengths = [], [], [], [], []
    for index, (case, row) in enumerate(zip(cases, rows)):
        require(case['passed'] is True and case['window'] == PLAN[index] and
                case['warmup'] is (index < 2) and row['paired_warmup'] is (index < 2), 'Bad arm/warmup binding')
        require(case['request_id'] == row['request_id'], 'Client/owner request differs')
        require(case['usage']['prompt_tokens'] == row['prompt_tokens'] == 8900 and
                row['requested_output_budget'] == 4096, 'Input/budget changed')
        count = row['generated_tokens']
        require(type(count) is int and 1 < count <= 4096 and count == case['usage']['completion_tokens'], 'Invalid output length')
        require(row['generated_natural_EOS'] is True and row['reason'] == 'eos_token' and
                row['original_reset_all32_completed'] is True, 'EOS/reset not proven')
        calls = row['native_calls']
        pf = [c for c in calls if c['prefill'] is True]
        dc = [c for c in calls if c['prefill'] is False]
        require(len(pf) == 18 and [(c['past'], c['T']) for c in pf] ==
                [(i * 512, min(512, 8900 - i * 512)) for i in range(18)], 'Wrong full prefill geometry')
        require(len(dc) == count - 1 and [(c['past'], c['T']) for c in dc] ==
                [(8900 + i, 1) for i in range(count - 1)], 'Wrong decode geometry')
        p = sum(positive(c['Native_CAPI_s']) for c in pf)
        d = sum(positive(c['Native_CAPI_s']) for c in dc)
        w = positive(row['wall_s'])
        require(p + d <= w + max(.01, w * 1e-5), 'CAPI spans exceed same-owner wall')
        prefill.append(p); decode.append(d); wall.append(w)
        http.append(positive(case['HTTP_wall_s'])); lengths.append(count)
    same_length = len(set(lengths[2:])) == 1
    return {'HTTP_wall': paired_metric(http), 'emitter_backend_wall': paired_metric(wall),
            'tail_prefill_CAPI_enclosure': paired_metric(prefill),
            'tail_decode_CAPI_enclosure': paired_metric(decode),
            'measured_generated_tokens': lengths[2:], 'matched_measured_output_lengths': same_length,
            'wall_schedule_attribution': 'CONFOUNDED_OUTPUT_LENGTH' if not same_length else 'Requires timing/noise assessment; equal length alone does not establish causal attribution',
            'automatic_adoption': False, 'Model_or_online_qualification': False,
            'limits': ['Tail CAPI includes upstream waits/host work; not GPU busy.',
                       'Do not sum stages or convert enclosed time to removable hardware cost.',
                       'No token timeline: user decode TPS, TTFT and ITL remain uncomputed.',
                       'Correctness and all32 source/lifecycle checks precede measurement; root must independently verify cleanup and keeper termination.',
                       'No physical-peak, general benchmark quality, or automatic default change.']}


def analyze(root):
    root = pathlib.Path(root).resolve()
    evidence = {}

    def read(relative, expected=None):
        path = (root / relative).resolve()
        require(path.is_relative_to(root), 'Archive path escapes root')
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, 'Evidence SHA mismatch: ' + str(relative))
        evidence[str(relative).replace('\\', '/')] = digest
        return json.loads(raw)

    watch = read(LABEL + '-watch.json')
    require(all(watch.get(k) is True for k in ['completed', 'post_empty', 'Native_lifecycle_qualified',
                                              'paired_measurement_conditions_qualified']), 'Terminal qualified controller required')
    require(not watch.get('forced_owned_step_fail_stop') and watch['restarts'] == 0, 'Forced stop/restart')
    require(watch['source_manifest_SHA256'] == MANIFEST and watch['Native_SHA256'] == NATIVE, 'Unrecognized source/Native')
    handle, run = watch['actual_handle'], watch['run']
    require(watch['terminal'][0] == handle and watch['terminal'][2:4] == ['COMPLETED', '0:0'], 'GPU not terminal0')
    require(len(watch['post']) == 4 and {r['node'] for r in watch['post']} == set(watch['nodes']) and
            all(r['KFD_empty'] is True and r['port43843_bindable'] is True for r in watch['post']), 'Cleanup proof incomplete')
    require(len(watch['network_and_normal']) == 4 and {r['stage'] for r in watch['network_and_normal']} == set(range(4)) and
            all(r['normal_markers'] == 1 and r['IBext_v8_mentions'] >= 1 and r['Socket_network_mentions'] == 0
                for r in watch['network_and_normal']), 'Network/normal proof incomplete')
    owners = watch['owner_results']
    require(len(owners) == 32 and {(o['stage'], o['rank']) for o in owners} ==
            {(s, r) for s in range(4) for r in range(8)}, 'All32 unique owners required')
    signature = None; emitter = None
    for owner in owners:
        relative = pathlib.PurePosixPath(owner['path']).relative_to(REMOTE)
        q = read(relative, owner['SHA256'])
        require(q['paired_run'] == run and q['paired_actual_step'] == handle and q['library_sha256'] == NATIVE, 'Owner identity differs')
        require(q['stage'] == owner['stage'] and q['rank'] == owner['rank'], 'Owner coordinate differs')
        require(all(q.get(k) is True for k in ['completed', 'normal_shutdown', 'normal_reset_close',
                                             'worker_shutdown_completed', 'native_service_loop_completed', 'post_close_Torch_liveness']), 'Owner lifecycle incomplete')
        require(not q.get('automatic_CPU_client_group_forced_stop'), 'CPU forced stop')
        binding = q['paired_source_binding_r241']
        require(binding['source_manifest_SHA256'] == MANIFEST and binding['all_source_files_bound_before_Torch'] is True, 'Owner source not bound')
        rows = q['service_requests']
        require(len(rows) == 18 and [r['pair_index'] for r in rows] == list(range(18)) and
                [r['prefill_control_window'] for r in rows] == list(PLAN), 'Owner pair incomplete')
        require(len({r['actual_prompt_token_ids_SHA256'] for r in rows}) == 1 and
                all(r['prompt_tokens'] == 8900 and r['requested_output_budget'] == 4096 and
                    r['generated_natural_EOS'] is True and r['original_reset_all32_completed'] is True for r in rows), 'Owner prompt/EOS/reset differs')
        sig = [(r['actual_prompt_token_ids_SHA256'], r['generated_token_ids_SHA256'], r['generated_tokens'], r['reason']) for r in rows]
        require(signature is None or sig == signature, 'Cross-owner outputs differ')
        signature = sig
        if (owner['stage'], owner['rank']) == (3, 0):
            require(q['normal_HTTP_shutdown'] is True, 'HTTP not shut down'); emitter = rows
    client_name = pathlib.Path('results') / (LABEL + '-client-' + run)
    client = read(client_name / 'result.json', watch['paired_client_result_SHA256'])
    wrapper = read(str(client_name) + '-wrapper.json', watch['paired_wrapper_SHA256'])
    require(all(client.get(k) is True for k in ['completed', 'passed', 'no_retries']) and
            client['attempts'] == client['completed_HTTP_200'] == 18 and client['run'] == run and
            client['actual_GPU_handle'] == handle and client['request_file_SHA256'] == REQUEST, 'Client not qualified')
    require(all(wrapper.get(k) is True for k in ['completed', 'passed', 'single_HTTP_child_reaped']), 'CPU wrapper not qualified')
    result = summarize(client['cases'], emitter)
    result.update(actual_handle=handle, source_manifest_SHA256=MANIFEST, evidence_SHA256=evidence,
                  scope='Offline r241 controller/owner/client metadata; no new execution or automatic promotion')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.archive_root), indent=2, allow_nan=False))
