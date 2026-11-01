"""Normal agent acceptance, distinct from the small-budget smoke client.

Profiles are prepared with the official tokenizer before loading weights.
Every workflow retains all real model messages and complete tool results.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import pathlib
import secrets
import threading
import time


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def make_input(record_count, nonce=None):
    """Produce an actual dataset; select records spread across the full file."""
    assert record_count >= 4
    nonce = nonce or secrets.token_hex(8)
    salt = int.from_bytes(hashlib.sha256(nonce.encode()).digest()[:4], 'big') % 100003
    records = [{'id': f'record_{i:06d}', 'value': (i * 7919 + salt) % 100003,
                'status': 'ready', 'description': f'Measurement from independent sample {i}.'}
               for i in range(record_count)]
    indices = [0, record_count // 3, 2 * record_count // 3, record_count - 1]
    return {'nonce': nonce, 'query_ids': [records[i]['id'] for i in indices], 'records': records}


def prepare_profiles(token_counter, output):
    """CPU helper: count real dataset text with the checkpoint tokenizer.

    Raw text counts only size the fixtures. The real workflow separately checks
    prompt usage from actual HTTP responses, without inventing assistant turns.
    """
    output.mkdir(parents=True, exist_ok=False)
    profiles = []
    for label, floor, budget, stream, thinking in [('HTTP', 8192, 4096, False, False), ('SSE', 16384, 4096, True, False), ('thinking', 8192, 8192, False, True)]:
        nonce = secrets.token_hex(8)
        def sample(count):
            text = compact(make_input(count, nonce))
            tokens = token_counter(text)
            assert type(tokens) is int and tokens > 0
            return text, tokens
        low, high = 4, 4
        while sample(high)[1] < floor:
            high *= 2
            assert high <= 8192, 'Tokenizer sizing exceeded the bounded fixture search.'
        while low < high:
            middle = (low + high) // 2
            if sample(middle)[1] >= floor:
                high = middle
            else:
                low = middle + 1
        text, tokens = sample(high)
        assert floor <= tokens < floor + 256
        path = output / (label + '-input.json')
        path.write_text(text + '\n')
        profiles.append({'label': label, 'input_file': str(path), 'minimum_prompt_tokens': floor,
                         'output_budget': budget, 'stream': stream, 'thinking': thinking,
                         'CPU_raw_text_tokens': tokens, 'record_count': high,
                         'input_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'real_model_acceptance': 'NOT_RUN'})
    (output / 'profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
    return profiles


def execute(call, sandbox):
    assert call.get('type') == 'function' and isinstance(call.get('id'), str) and call['id']
    fn = call['function']
    args = json.loads(fn['arguments'])
    assert isinstance(args, dict)
    if fn['name'] == 'read_file':
        assert set(args) == {'path'} and args['path'] in ['input.json', 'result.json', 'final.json']
        result = json.loads((sandbox / args['path']).read_bytes())
    elif fn['name'] == 'calculate':
        assert set(args) == {'a', 'b'} and all(type(args[k]) is int and abs(args[k]) < 10**9 for k in ['a', 'b'])
        result = {'sum': args['a'] + args['b']}
    elif fn['name'] == 'write_file':
        assert set(args) == {'path', 'content'} and args['path'] in ['result.json', 'final.json']
        assert isinstance(args['content'], str) and len(args['content']) < 16384
        value = json.loads(args['content'])
        assert isinstance(value, dict)
        with (sandbox / args['path']).open('xb') as file:
            file.write((compact(value) + '\n').encode())
        result = {'ok': True}
    else:
        raise AssertionError('Undeclared generated tool.')
    return {'role': 'tool', 'tool_call_id': call['id'], 'content': compact(result)}, args, result


def workflow(p0, client, folder, profile):
    folder.mkdir()
    raw = pathlib.Path(profile['input_file']).read_bytes()
    if 'input_SHA256' in profile:
        assert hashlib.sha256(raw).hexdigest() == profile['input_SHA256']
    data = json.loads(raw)
    assert isinstance(data['nonce'], str) and len(data['nonce']) >= 16
    values = {row['id']: row['value'] for row in data['records']}
    a, b, c, d = [values[key] for key in data['query_ids']]
    first, second = a + b, c + d
    total = first + second
    (folder / 'input.json').write_bytes(raw)
    messages = [{'role': 'system', 'content': 'Use tools and preserve their actual results. Follow the requested tool for each step.'},
                {'role': 'user', 'content': 'Perform these eight tool steps in order: read input.json; calculate the sum of values of query_ids[0] and query_ids[1]; calculate the sum for query_ids[2] and query_ids[3]; write result.json with nonce, first_sum, second_sum; read result.json; calculate first_sum+second_sum; write final.json with nonce and total; read final.json. Then report nonce and total. Find values in the records by ID. Never invent tool results.'}]
    plan = ['read_file', 'calculate', 'calculate', 'write_file', 'read_file', 'calculate', 'write_file', 'read_file']
    actions, requests, generated = [], [], []
    budget = profile['output_budget']
    assert budget in [4096, 8192]
    for step, name in enumerate(plan):
        message, finish, _, wire = client.request(messages, {'type': 'function', 'function': {'name': name}}, profile['stream'], p0.TOOLS,
                                                  max_tokens=budget, thinking=profile.get('thinking', False))
        requests.append(wire)
        assert wire['HTTP_status'] == 200 and finish == 'tool_calls', 'A real step did not finish a valid tool call.'
        if step >= 1:
            assert wire['usage']['prompt_tokens'] >= profile['minimum_prompt_tokens'], 'Long real tool history was not observed in actual prompt usage.'
        assert wire['usage']['prompt_tokens'] + budget <= 32768
        calls = message.get('tool_calls', [])
        assert len(calls) == 1 and calls[0]['function']['name'] == name
        call = calls[0]
        assert call['id'] not in [item['generated_call']['id'] for item in actions]
        result, args, actual = execute(call, folder)
        expected = {1: {'a': a, 'b': b}, 2: {'a': c, 'b': d}, 5: {'a': first, 'b': second}}
        if step in expected:
            assert args == expected[step]
        if step == 3:
            assert args['path'] == 'result.json' and json.loads(args['content']) == {'nonce': data['nonce'], 'first_sum': first, 'second_sum': second}
        if step in [4, 7]:
            assert args['path'] == ('result.json' if step == 4 else 'final.json')
        if step == 6:
            assert args['path'] == 'final.json' and json.loads(args['content']) == {'nonce': data['nonce'], 'total': total}
        actions.append({'generated_call': call, 'arguments': args, 'actual_result_sha256': hashlib.sha256(result['content'].encode()).hexdigest()})
        generated.append(message)
        messages.extend([message, result])
    final, finish, _, wire = client.request(messages, 'none', profile['stream'], p0.TOOLS,
                                            max_tokens=budget, thinking=profile.get('thinking', False))
    requests.append(wire)
    assert wire['HTTP_status'] == 200 and finish == 'stop' and not final.get('tool_calls')
    assert wire['usage']['prompt_tokens'] >= profile['minimum_prompt_tokens']
    assert wire['usage']['prompt_tokens'] + budget <= 32768
    assert data['nonce'] in (final.get('content') or '') and str(total) in (final.get('content') or '')
    assert json.loads((folder / 'final.json').read_bytes()) == {'nonce': data['nonce'], 'total': total}
    generated.append(final)
    if profile.get('thinking'):
        assert any(row.get('reasoning_content') for row in generated)
    return {'passed': True, 'input_file_SHA256': hashlib.sha256(raw).hexdigest(), 'nonce': data['nonce'],
            'output_budget': budget, 'minimum_input_tokens': profile['minimum_prompt_tokens'],
            'actual_prompt_lengths': [row['usage']['prompt_tokens'] for row in requests],
            'real_tool_steps': len(actions), 'actions': actions, 'wire': requests, 'generated_messages': generated,
            'final_file_SHA256': hashlib.sha256((folder / 'final.json').read_bytes()).hexdigest()}


def run_checks(p0, url, model, output, identity, profiles, timeout):
    assert len(profiles) == 3 and [p['label'] for p in profiles] == ['HTTP', 'SSE', 'thinking']
    assert profiles[0]['minimum_prompt_tokens'] >= 8192 and profiles[1]['minimum_prompt_tokens'] >= 16384
    assert profiles[2]['minimum_prompt_tokens'] >= 8192 and profiles[2].get('thinking') and profiles[2]['output_budget'] == 8192
    assert profiles[0]['output_budget'] == profiles[1]['output_budget'] == 4096
    output.mkdir(parents=True, exist_ok=False)
    intervals, lock, gate = [], threading.Lock(), threading.Barrier(2, timeout=30)
    entered = {name: threading.Event() for name in ['HTTP', 'SSE']}
    class Client(p0.WireClient):
        def __init__(self, name, budget):
            path = output / name
            path.mkdir()
            super().__init__(url, model, path, budget, timeout)
            self.session = 'P1-' + name
            self.label = name
        def request(self, *args, **kwargs):
            if self.label in entered and self.serial == 0:
                gate.wait()
            row = {'client': self.label, 'serial': self.serial + 1, 'start': time.monotonic()}
            if self.label in entered:
                entered[self.label].set()
            try:
                return super().request(*args, **kwargs)
            finally:
                row['end'] = time.monotonic()
                with lock:
                    intervals.append(row)
    def case(name, fn):
        row = {'name': name, 'passed': False}
        try:
            row.update(fn())
        except BaseException as exc:
            row['error'] = repr(exc)
        return row
    clients = {p['label']: Client(p['label'], p['output_budget']) for p in profiles}
    clients['negative'] = Client('negative', 4096)
    cases = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(case, p['label'], lambda p=p: workflow(p0, clients[p['label']], clients[p['label']].path / 'workspace', p)) for p in profiles[:2]]
        assert all(event.wait(35) for event in entered.values())
        for kind in ['invalid_history', 'truncated']:
            cases.append(case(kind, lambda kind=kind: p0.negative(clients['negative'], kind)))
        cases.extend(job.result() for job in jobs)
    cases.append(case('thinking', lambda: workflow(p0, clients['thinking'], clients['thinking'].path / 'workspace', profiles[2])))
    def recovery():
        message, finish, _, wire = clients['negative'].request([{'role': 'user', 'content': 'What is 1+1? Give a short answer.'}], 'none', False, None, max_tokens=4096)
        assert wire['HTTP_status'] == 200 and finish == 'stop' and '2' in (message.get('content') or '')
        return {'passed': True, 'wire': wire}
    cases.append(case('recovery', recovery))
    flows = [row for row in cases if row['name'] in ['HTTP', 'SSE', 'thinking']]
    checks = {'all_three_eight_step_workflows': len(flows) == 3 and all(row['passed'] and row.get('real_tool_steps') == 8 for row in flows)}
    if checks['all_three_eight_step_workflows']:
        nonces = [row['nonce'] for row in flows]
        checks['distinct_nonces'] = len(set(nonces)) == 3
        checks['no_foreign_nonce_in_any_generated_message'] = all(
            all(other not in compact(row['generated_messages']) for other in nonces if other != row['nonce']) for row in flows)
        ids = [action['generated_call']['id'] for row in flows for action in row['actions']]
        checks['unique_tool_call_ids'] = len(ids) == len(set(ids)) == 24
    overlap = lambda a, b: max(a['start'], b['start']) < min(a['end'], b['end'])
    positives = [row for row in intervals if row['client'] in entered]
    checks['HTTP_SSE_client_request_overlap'] = any(overlap(a, b) for a in positives for b in positives if a['client'] == 'HTTP' and b['client'] == 'SSE')
    negatives = [row for row in intervals if row['client'] == 'negative' and row['serial'] <= 2]
    checks['both_negative_requests_during_normal_workflows'] = len(negatives) == 2 and all(any(overlap(a, b) for b in positives) for a in negatives)
    report = {'completed': True, 'passed': all(row['passed'] for row in cases) and all(checks.values()),
              'scope': 'REGULAR_AGENT_32K_NOT_SMOKE', 'total_context_required': 32768,
              'native_batch_concurrency': 'NOT_CLAIMED', 'identity': str(identity),
              'identity_SHA256': hashlib.sha256(identity.read_bytes()).hexdigest(),
              'cases': cases, 'isolation_checks': checks, 'request_intervals': intervals,
              'actual_HTTP_requests': sum(client.serial for client in clients.values())}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ['p0-client', 'p0-sha256', 'base-url', 'model', 'output-dir', 'identity', 'profiles']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    source = pathlib.Path(args.p0_client)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.p0_sha256
    spec = importlib.util.spec_from_file_location('reviewed_p0', source)
    p0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0)
    report = run_checks(p0, args.base_url, args.model, pathlib.Path(args.output_dir), pathlib.Path(args.identity),
                        json.loads(pathlib.Path(args.profiles).read_bytes()), args.timeout)
    print(compact({'passed': report['passed'], 'isolation_checks': report['isolation_checks'], 'actual_HTTP_requests': report['actual_HTTP_requests']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
