"""Read-only audit of the scoped six-request GLM W2/W4 run, never launches jobs.

Archive root mirrors the remote project root. Raw owners/wires stay outside Git;
the output records hashes and numeric summaries, not prompts or generated text.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

PLAN = (2, 4, 2, 4, 4, 2)
LABEL = 'GLM-prefill-W2-W4-Model-r308'
OWNER = '3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
CONTROLLER = '0205eded1df492d99535a98eaae1289b4cf19f2500df2b2cc8039a3010c79846'
NATIVE = '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
REQUEST = '5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'
REMOTE = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Timing must be finite and positive')
    return value


def paired_metric(values):
    require(len(values) == 6, 'Need two warmups and four measured requests')
    values = [positive(v) for v in values]
    a, b, c, d = values[2:]
    ratio = math.exp((math.log(a) + math.log(d) - math.log(b) - math.log(c)) / 2)
    return {'W2_s': [a, d], 'W4_s': [b, c], 'geometric_W2_over_W4_ratio': ratio,
            'candidate_time_reduction_percent': 100 * (1 - 1 / ratio),
            'W2_repeat_relative_spread': abs(a-d)/math.sqrt(a*d),
            'W4_repeat_relative_spread': abs(b-c)/math.sqrt(b*c),
            'warmups_excluded': 2, 'ABBA_blocks': 1,
            'statistical_significance_claim': False, 'confidence_interval': None}


def summarize(cases, rows):
    require(len(cases) == len(rows) == 6, 'Incomplete scoped pair')
    require([r['pair_index'] for r in rows] == list(range(6)), 'Owner order differs')
    require([c['index'] for c in cases] == list(range(6)), 'Client order differs')
    require([r['prefill_control_window'] for r in rows] == list(PLAN), 'Window plan differs')
    prefill, decode, wall, http, lengths, paths = [], [], [], [], [], []
    for i, (case, row) in enumerate(zip(cases, rows)):
        require(case['passed'] is True and case['natural_EOS'] is True and
                case['window'] == PLAN[i] and case['warmup'] is (i < 2) and
                row['paired_warmup'] is (i < 2), 'Arm or warmup binding differs')
        require(case['request_id'] == row['request_id'], 'Client/owner identity differs')
        require(case['usage']['prompt_tokens'] == row['prompt_tokens'] == 8900 and
                row['requested_output_budget'] == 4096, 'Input/output budget differs')
        n = row['generated_tokens']
        require(type(n) is int and 1 < n <= 4096 and n == case['usage']['completion_tokens'],
                'Output length invalid')
        require(row['generated_natural_EOS'] is True and row['reason'] == 'eos_token' and
                row['original_reset_all32_completed'] is True, 'EOS/reset incomplete')
        pf = [v for v in row['native_calls'] if v['prefill'] is True]
        dc = [v for v in row['native_calls'] if v['prefill'] is False]
        require(len(pf)+len(dc) == len(row['native_calls']), 'Unclassified Native call')
        require([(v['past'], v['T']) for v in pf] ==
                [(j*512, min(512, 8900-j*512)) for j in range(18)], 'Prefill geometry differs')
        require([(v['past'], v['T']) for v in dc] ==
                [(8900+j, 1) for j in range(n-1)], 'Decode geometry differs')
        p = sum(positive(v['Native_CAPI_s']) for v in pf)
        d = sum(positive(v['Native_CAPI_s']) for v in dc)
        w = positive(row['wall_s'])
        require(p+d <= w+max(.01, w*1e-5), 'CAPI enclosure exceeds owner wall')
        prefill.append(p); decode.append(d); wall.append(w)
        http.append(positive(case['HTTP_wall_s'])); lengths.append(n)
        paths.append(row['generated_token_ids_SHA256'])
    same_length, same_path = len(set(lengths[2:])) == 1, len(set(paths[2:])) == 1
    return {'HTTP_wall': paired_metric(http), 'emitter_backend_wall': paired_metric(wall),
            'tail_prefill_CAPI_enclosure': paired_metric(prefill),
            'tail_decode_CAPI_enclosure': paired_metric(decode),
            'measured_generated_tokens': lengths[2:], 'matched_output_lengths': same_length,
            'matched_output_paths': same_path,
            'wall_schedule_attribution': 'CONFOUNDED_OUTPUT_PATH_OR_LENGTH' if not (same_length and same_path)
                else 'Same path; one fixed-order block still needs noise assessment',
            'automatic_adoption': False, 'full_Model_or_online_qualification': False,
            'limits': ['One fixed-request ABBA block; no randomized or statistical generalization.',
                       'CAPI encloses host/upstream waits, not GPU busy or removable time.',
                       'Do not sum stages; buffered HTTP does not prove client TTFT/ITL/decode SLO.',
                       'Tool correctness and EOS are scoped checks, not a wide quality benchmark.']}


def analyze(root):
    root = Path(root).resolve(); evidence = {}

    def read(relative, expected=None):
        p = (root / relative).resolve()
        require(p.is_relative_to(root), 'Evidence escapes archive root')
        raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, 'Evidence SHA differs: '+str(relative))
        evidence[str(relative).replace('\\', '/')] = digest
        return json.loads(raw)

    w = read('results/'+LABEL+'-watch.json')
    require(all(w.get(k) is True for k in ['completed','post_empty','scoped_Model_pair_qualified',
            'all32_token_EOS_reset_signatures_equal']), 'Scoped terminal audit incomplete')
    require(w['owner_manifest_SHA256'] == OWNER and w['controller_manifest_SHA256'] == CONTROLLER,
            'Unrecognized source')
    require(w['actual_HTTP'] == w['actual_Native'] == 6 and w['no_retry_reload'] is True,
            'Wrong scope or retry')
    handle = w['actual_GPU']
    require(w['terminal'][0] == handle and w['terminal'][2:4] == ['COMPLETED','0:0'],
            'GPU not completed0')
    require(len(w['post']) == 4 and len({v['node'] for v in w['post']}) == 4 and
            all(v['KFD_empty'] is True and v['port43843_bindable'] is True for v in w['post']),
            'Independent postcheck incomplete')
    require({v['stage'] for v in w['network_and_normal']} == set(range(4)) and
            len(w['network_and_normal']) == 4 and all(v['normal_marker'] == 1 and
            v['IBext_v8'] is True and v['Socket_fallback'] is False for v in w['network_and_normal']),
            'Network or normal-exit proof incomplete')
    owners = w['owner_results']; emitter = None; signature = None
    require(len(owners) == 32 and {(v['stage'],v['rank']) for v in owners} ==
            {(s,r) for s in range(4) for r in range(8)}, 'All32 unique owners required')
    for ref in owners:
        p = PurePosixPath(ref['path']).relative_to(REMOTE); q = read(str(p), ref['SHA256'])
        require(q['paired_run'] == LABEL and q['actual_producer'] == handle and
                q['library_sha256'] == NATIVE and q['stage'] == ref['stage'] and
                q['rank'] == ref['rank'] and q['latest_protocol_binding_r290']['source_manifest_SHA256'] == OWNER,
                'Owner source/coordinate/actual handle differs')
        require(all(q.get(k) is True for k in ['completed','worker_shutdown_completed',
                'post_close_Torch_liveness','normal_reset_close','native_service_loop_completed',
                'normal_shutdown']), 'Owner lifecycle incomplete')
        rows = q['service_requests']
        require(q['full_requests_completed'] == len(rows) == 6 and
                [r['pair_index'] for r in rows] == list(range(6)) and
                [r['prefill_control_window'] for r in rows] == list(PLAN), 'Owner plan differs')
        require(all(r['original_reset_all32_completed'] and r['generated_natural_EOS'] and
                r['reason'] == 'eos_token' and r['prompt_tokens'] == 8900 and
                r['requested_output_budget'] == 4096 for r in rows), 'Owner EOS/input/reset differs')
        sig = [(r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],
                r['generated_tokens'],r['reason']) for r in rows]
        require(len({x[0] for x in sig}) == 1, 'Prompt changes across arms')
        require(signature is None or signature == sig, 'Cross-owner signature differs'); signature = sig
        if (ref['stage'],ref['rank']) == (3,0):
            require(q['normal_HTTP_shutdown'] is True and not q['cleanup']['forced_cleanup'] and
                    not q['cleanup']['errors'], 'Emitter cleanup incomplete'); emitter = rows
    prefix = 'results/GLM-latest-full-owner-r290-'
    sup = read(prefix+'supervisor-'+LABEL+'.json', w['CPU_supervisor_SHA256'])
    require(all(sup.get(k) is True for k in ['completed','passed','CPU_only','no_retry_reload']) and
            sup['actual_GPU_handle'] == handle and sup['actual_HTTP'] == sup['actual_Native'] == 6 and
            set(sup['phases']) == {'paired'}, 'Supervisor scope differs')
    phase = sup['phases']['paired']; require(phase['exit_code'] == 0, 'Client failed')
    client_dir = prefix+'client-'+LABEL+'/paired/'
    c = read(client_dir+'result.json', phase['result_SHA256'])
    require(c == phase['result'] and all(c.get(k) is True for k in
            ['completed','passed','no_retries','matched_pair_conditions_validated']) and
            c['actual_HTTP_requests'] == 6 and c['actual_request_SHA256'] == REQUEST,
            'Client incomplete or source differs')
    request = read('results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json', REQUEST)
    data = json.loads(request['messages'][-1]['content']); values = {v['id']:v['value'] for v in data['records']}
    expected = dict(zip(('a','b'), (values[k] for k in data['query_ids'][:2])))
    ids = set()
    for i, case in enumerate(c['cases']):
        wire = read(client_dir+'response-%02d.wire'%i, case['wire_SHA256'])
        require('error' not in wire and wire['id'] == case['request_id'] and wire['id'] not in ids and
                wire['usage'] == case['usage'] and len(wire['choices']) == 1, 'Wire identity differs')
        ids.add(wire['id']); choice = wire['choices'][0]; calls = choice['message'].get('tool_calls',[])
        require(choice['finish_reason'] == 'tool_calls' and len(calls) == 1 and
                calls[0]['type'] == 'function' and calls[0]['function']['name'] == 'calculate', 'Wrong tool')
        args = json.loads(calls[0]['function']['arguments'])
        require(args == expected and all(type(args[k]) is int for k in ('a','b')), 'Wrong tool arguments')
    result = summarize(c['cases'], emitter)
    result.update(actual_handle=handle, owner_manifest_SHA256=OWNER,
                  controller_manifest_SHA256=CONTROLLER, Native_SHA256=NATIVE,
                  source_identity_declarations_and_all32_lifecycle_consistent=True,
                  source_manifest_members_rehashed_by_this_analyzer=False,
                  tool_reply_semantics_audited=True,
                  CPU_keeper_terminal_requires_independent_accounting=True,
                  evidence_SHA256=evidence)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--archive-root',required=True)
    print(json.dumps(analyze(parser.parse_args().archive_root),indent=2,allow_nan=False))
