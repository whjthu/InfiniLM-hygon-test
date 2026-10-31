"""Source-only bounded real multicall supplement. Never imported by r223.

No launcher or GPU authority is included. Wire traces and tool data stay local.
Two modes, at most two actual HTTP requests each, no retries or fabricated calls.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import secrets
import time

from transport_binding_r278 import load_transport
NATIVE_SHA = 'ec239cf5d60fa4c00d6f2168beb2289b13151063f1f068a83e3cf64d8b9c99e4'
MODEL_MANIFEST_SHA = 'f72bbc3fcb0971e395121ba297eae08080d67eab16c7eb9aefb101eaa1057671'
PROFILE_RECEIPT_SHA = 'd63db286e894e93fcd1eb2b5cf5893280f5b6ddae958ad41725153db78a4de4a'
PATHS = ('left.json', 'right.json')
TOOLS = [{'type': 'function', 'function': {'name': 'read_file',
    'description': 'Read one of the two actual JSON data files.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'enum': list(PATHS)}},
                   'required': ['path'], 'additionalProperties': False}}}]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_calls(message, finish):
    calls = message.get('tool_calls', [])
    if finish != 'tool_calls' or len(calls) != 2:
        raise ValueError('MULTICALL_NOT_OBSERVED: exactly two requested real calls required')
    seen, paths = set(), []
    for call in calls:
        call_id = call.get('id')
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ValueError('Missing or duplicate external opaque ID')
        seen.add(call_id)
        if call.get('type') != 'function' or call['function']['name'] != 'read_file':
            raise ValueError('Unexpected generated function')
        args = json.loads(call['function']['arguments'])
        if not isinstance(args, dict) or set(args) != {'path'} or args['path'] not in PATHS:
            raise ValueError('Unexpected generated path or arguments')
        paths.append(args['path'])
    if set(paths) != set(PATHS):
        raise ValueError('Both actual files must be requested once')
    return list(zip(calls, paths))


def execute_and_reverse(validated, workspace):
    results, expected = [], {}
    for call, name in validated:
        data = json.loads((workspace / name).read_bytes())
        values = {row['id']: row['value'] for row in data['records']}
        total = sum(values[key] for key in data['query_ids'])
        expected[name] = {'nonce': data['nonce'], 'sum': total}
        results.append({'role': 'tool', 'tool_call_id': call['id'], 'content': compact(data)})
    expected['total'] = sum(expected[name]['sum'] for name in PATHS)
    return list(reversed(results)), expected


def check_final(message, finish, expected):
    if finish != 'stop' or message.get('tool_calls'):
        raise ValueError('Final answer must naturally stop without tool calls')
    actual = json.loads(message.get('content') or '')
    if actual != expected:
        raise ValueError('Incorrect file/nonce/sum association or total')
    # JSON booleans compare equal to small ints in Python; reject that ambiguity.
    if type(actual.get('total')) is not int or any(type(actual[name].get('sum')) is not int for name in PATHS):
        raise ValueError('Final sums must be integers')


def bound_report(path, args):
    report = json.loads(path.read_bytes())
    if report['run'] != args.expected_run or report['manifest_SHA256'] != MODEL_MANIFEST_SHA or report['node'] != 0:
        raise ValueError('Same-run owner report identity mismatch')
    return report


def response_binding(folder, serial, usage, messages, thinking):
    wire = json.loads((folder/('request-%03d.wire' % serial)).read_bytes())
    request_id = wire.get('id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('Missing actual HTTP response request ID')
    rc = [hashlib.sha256((v.get('reasoning_content') or v.get('reasoning')).encode()).hexdigest()
          for v in messages if v.get('role') == 'assistant' and (v.get('reasoning_content') or v.get('reasoning'))]
    return {'request_id': request_id, 'usage': usage, 'RC_SHA256_sequence': rc, 'thinking': thinking}


def owner_receipts(path, args, session, bindings, budget, remaining):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        remaining()
        rows = [r for r in bound_report(path, args)['agent_requests'] if r.get('client_session') == session]
        if len(rows) > 2:
            raise ValueError('Unexpected additional Native requests')
        if len(rows) == 2 and all(r.get('normal_reset') and r.get('finish') == 'eos' for r in rows):
            break
        time.sleep(.05)
    else:
        raise ValueError('Two completed same-residency owner receipts unavailable')
    safe = []
    by_id = {row['request_id']: row for row in rows}
    if len(by_id) != 2 or set(by_id) != {b['request_id'] for b in bindings}:
        raise ValueError('Actual HTTP/Native request ID binding mismatch')
    for binding in bindings:
        row = by_id[binding['request_id']]
        expected_index = (2 if binding['thinking'] else 0) + bindings.index(binding)
        if row['admitted_index'] != expected_index:
            raise ValueError('Unexpected admitted request index')
        usage = binding['usage']
        audit = row['history_render_audit']
        if not (row['input_tokens'] == usage['prompt_tokens'] == audit['prompt_tokens']
                and row['output_budget'] == budget and row['long_prefill_progress']['input_tokens_committed'] == usage['prompt_tokens']
                and audit['qualified'] and audit['prequeue_bound'] and not audit['markers_sent_to_Native']
                and audit['official_reencode_equals_actual_prompt_and_IDs'] and audit['generation_prefix_correct']
                and audit['generation_thinking'] == binding['thinking']
                and audit['source_RC_SHA256_sequence'] == binding['RC_SHA256_sequence']
                and audit['RC_one_contribution_per_source_field'] and audit['RC_order_preserved']):
            raise ValueError('Native input/render/commit audit mismatch')
        ids = row['generated_ids']
        if not ids or ids[-1] not in (163585, 163586) or not all(type(i) is int and 0 <= i < 163840 for i in ids):
            raise ValueError('Missing natural EOS or invalid generated IDs')
        safe.append({'input_tokens': row['input_tokens'], 'output_budget': budget,
            'generated_tokens': len(ids), 'normal_reset': True, 'natural_EOS': True,
            'actual_prompt_SHA256': audit['actual_prompt_SHA256'], 'Native_prompt_token_ids_SHA256': audit['Native_prompt_token_ids_SHA256']})
    return safe


def run_mode(p0, args, root, profile_data, thinking, remaining):
    label, budget = ('thinking', 8192) if thinking else ('chat', 4096)
    folder = root / label
    folder.mkdir()
    workspace = folder / 'workspace'
    workspace.mkdir()
    records = profile_data['records']
    middle = len(records) // 2
    for name, rows in zip(PATHS, (records[:middle], records[middle:])):
        if len(rows) < 4:
            raise ValueError('Insufficient real source records')
        data = {'nonce': secrets.token_hex(16), 'query_ids': [rows[i]['id'] for i in (0, len(rows)//3, 2*len(rows)//3, len(rows)-1)], 'records': rows}
        with (workspace / name).open('xb') as stream:
            stream.write((compact(data)+'\n').encode())
    session = 'P1-multicall-' + label + '-r238-' + secrets.token_hex(16)
    client = p0.WireClient(args.base_url, 'Kimi-K3', folder, budget, min(args.timeout, remaining()-30))
    client.session = session
    messages = [{'role': 'system', 'content': 'Use real tools and preserve their actual results. Tool result order may differ from call order; associate each result by its tool_call_id.'},
        {'role': 'user', 'content': 'In ONE response call read_file twice, once for left.json and once for right.json. After both results, find each file query_ids in its records and sum those four values. Output ONLY a JSON object with keys left.json and right.json, each containing its actual nonce and integer sum, and total containing both sums added together. Do not guess file contents.'}]
    usages, events, bindings = [], [], []
    original = copy.deepcopy(messages)
    message, finish, usage, event = client.request(messages, 'required', False, TOOLS, max_tokens=budget, thinking=thinking, parallel=True)
    remaining()
    if messages != original or event['HTTP_status'] != 200:
        raise ValueError('First actual HTTP request failed or history mutated')
    usages.append(usage); events.append(event)
    bindings.append(response_binding(folder, 1, usage, messages, thinking))
    validated = validated_calls(message, finish)
    reversed_results, expected = execute_and_reverse(validated, workspace)
    messages.append(copy.deepcopy(message))
    messages.extend(reversed_results)
    original = copy.deepcopy(messages)
    client.timeout = min(args.timeout, remaining()-30)
    message, finish, usage, event = client.request(messages, 'none', False, TOOLS, max_tokens=budget, thinking=thinking, parallel=True)
    remaining()
    if messages != original or event['HTTP_status'] != 200:
        raise ValueError('Second actual HTTP request failed or complete history mutated')
    usages.append(usage); events.append(event)
    bindings.append(response_binding(folder, 2, usage, messages, thinking))
    if usage['prompt_tokens'] < 8192 or any(u['prompt_tokens'] + budget > 32768 for u in usages):
        raise ValueError('Actual full tool history below 8K or context overflow')
    if thinking and not message.get('reasoning_content'):
        raise ValueError('Requested thinking not observed')
    check_final(message, finish, expected)
    owners = owner_receipts(pathlib.Path(args.service_report), args, session, bindings, budget, remaining)
    return {'mode': label, 'passed': True, 'actual_HTTP_requests': 2, 'output_budget': budget,
        'same_response_two_actual_model_calls': True, 'unique_opaque_IDs': True,
        'tool_results_reversed_without_ID_replacement': True, 'file_nonce_sum_associations_correct': True,
        'wire': events, 'owner_receipts': owners,
        'file_SHA256': {name: sha(workspace/name) for name in PATHS}}


def main():
    p = argparse.ArgumentParser()
    for name in ('base-url', 'output-dir', 'identity', 'service-report', 'frozen-client-dir', 'profiles', 'ready', 'expected-run', 'expected-gpu-handle'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--timeout', type=float, default=3600)
    args = p.parse_args()
    if not 120 < args.deadline_unix-time.time() <= 9000 or not 30 < args.timeout <= 3600:
        raise ValueError('Require one fixed <=9000s client deadline and <=3600s HTTP timeout')
    deadline = time.monotonic() + args.deadline_unix-time.time()
    def remaining():
        value = deadline-time.monotonic()
        if value <= 60:
            raise TimeoutError('Common deadline exhausted; no retries or phase reset')
        return value
    root = pathlib.Path(args.output_dir)
    root.mkdir(exist_ok=False)
    report = {'completed': False, 'passed': False, 'no_retries': True, 'no_cropping': True,
        'external_absolute_timeout_required': True, 'launcher_included': False,
        'max_HTTP_requests': 4, 'deadline_UTC_unix': args.deadline_unix, 'cases': [],
        'qualification': 'CLIENT_WITNESS_ONLY_UNTIL_ALL_32_OWNERS_AND_TERMINAL_CLEANUP_AUDITED'}
    try:
        identity = pathlib.Path(args.identity)
        ident = json.loads(identity.read_bytes())
        if ident['native_SHA256'] != NATIVE_SHA or ident['max_context_tokens'] != 32768 or ident['TP'] != 16 or ident['PP'] != 2:
            raise ValueError('Frozen runtime identity mismatch')
        if ident['manifest_SHA256'] != MODEL_MANIFEST_SHA or ident['actual_step'] != args.expected_gpu_handle or ident['cards'] != 32:
            raise ValueError('Actual handle or frozen model/source manifest mismatch')
        ready = json.loads(pathlib.Path(args.ready).read_bytes())
        if not (ready['ready'] and ready['actual_step'] == args.expected_gpu_handle
                and pathlib.Path(ready['identity_path']).resolve() == identity.resolve()
                and ready['URL_on_service_node'].rstrip('/') == args.base_url.rstrip('/')):
            raise ValueError('HTTP ready/identity/actual-handle binding mismatch')
        if pathlib.Path(args.service_report).name != 'Kimi-latest-service-r275-'+args.expected_run+'-node0.json':
            raise ValueError('Unexpected same-run owner report name')
        bound_report(pathlib.Path(args.service_report), args)
        report['identity_SHA256'] = sha(identity)
        report['actual_GPU_handle'] = args.expected_gpu_handle
        report['same_run'] = args.expected_run
        service = pathlib.Path(args.frozen_client_dir).parent
        p0 = load_transport(service, 'frozen_multicall_p0_r278')
        profiles = pathlib.Path(args.profiles)
        receipt = profiles.parent/'CPU-profile-receipt-r179.json'
        if sha(receipt) != PROFILE_RECEIPT_SHA:
            raise ValueError('CPU profile receipt mismatch')
        rows = json.loads(profiles.read_bytes())
        if rows != json.loads(receipt.read_bytes())['profiles']:
            raise ValueError('Profile manifest mismatch')
        profile = next(row for row in rows if row['label'] == 'HTTP')
        data_file = pathlib.Path(profile['input_file'])
        if sha(data_file) != profile['input_SHA256']:
            raise ValueError('Profile data mismatch')
        data = json.loads(data_file.read_bytes())
        report['source_profile_SHA256'] = sha(data_file)
        for thinking in (False, True):
            remaining()
            report['cases'].append(run_mode(p0, args, root, data, thinking, remaining))
        remaining()
        report['passed'] = len(report['cases']) == 2 and all(row['passed'] for row in report['cases'])
    except Exception as error:
        report['error_class'] = type(error).__name__
        report['error_SHA256'] = hashlib.sha256(str(error).encode()).hexdigest()
    finally:
        report['completed'] = True
        with (root/'result.json').open('x', encoding='utf8') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
