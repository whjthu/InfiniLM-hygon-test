"""Real long tool arguments, with checkpoint-tokenizer counts and file readback."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import secrets


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def load(path, digest, name):
    path = pathlib.Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(p0, count_tokens, url, model, output, identity, timeout=3600):
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'workspace'
    workspace.mkdir()
    client = p0.WireClient(url, model, output, 4096, timeout)
    client.session = 'P1-long-output'
    nonce, total = secrets.token_hex(8), 10000 + secrets.randbelow(90000)
    tools = [p0.TOOLS[0], p0.TOOLS[2]]
    report = {'completed': False, 'passed': False, 'scope': 'REAL_LONG_TOOL_ARGUMENTS_NOT_SMOKE',
              'identity': str(identity), 'identity_SHA256': hashlib.sha256(identity.read_bytes()).hexdigest(),
              'output_budget': 4096, 'minimum_generated_tokens': 1024, 'minimum_argument_tokens': 1024,
              'rows_required': 160, 'nonce': nonce, 'wire': []}
    messages = [{'role': 'system', 'content': 'Use the requested tools. Do not invent tool results.'},
                {'role': 'user', 'content': f'Write report.json as a JSON object with nonce "{nonce}" and rows, an array of exactly 160 objects. For each integer i from 0 through 159, include one object with integer fields i and total, where total is {total}. Write all 160 objects explicitly, without ellipses, loops, omitted rows or code. Then read report.json back and confirm the nonce and number of rows.'}]
    try:
        choice = lambda name: {'type': 'function', 'function': {'name': name}}
        message, finish, usage, wire = client.request(messages, choice('write_file'), True, tools, max_tokens=4096)
        report['wire'].append(wire)
        assert wire['HTTP_status'] == 200 and finish == 'tool_calls'
        calls = message.get('tool_calls', [])
        assert len(calls) == 1 and calls[0]['function']['name'] == 'write_file'
        call = calls[0]
        assert call.get('type') == 'function' and isinstance(call.get('id'), str) and call['id']
        args = json.loads(call['function']['arguments'])
        assert set(args) == {'path', 'content'} and args['path'] == 'report.json'
        assert isinstance(args['content'], str) and len(args['content']) <= 65536
        content = json.loads(args['content'])
        assert set(content) == {'nonce', 'rows'} and content['nonce'] == nonce
        rows = content['rows']
        assert isinstance(rows, list) and len(rows) == 160
        assert all(set(row) == {'i', 'total'} and type(row['i']) is int and row['i'] == i and type(row['total']) is int and row['total'] == total for i, row in enumerate(rows))
        actual_tokens = count_tokens(args['content'])
        assert type(actual_tokens) is int and actual_tokens >= 1024
        assert usage['completion_tokens'] >= 1024
        with (workspace / 'report.json').open('x', encoding='utf-8', newline='\n') as file:
            file.write(args['content'])
        report.update(generated_write_call=call, actual_argument_tokens=actual_tokens,
                      actual_generated_tokens=usage['completion_tokens'], file_SHA256=hashlib.sha256((workspace / 'report.json').read_bytes()).hexdigest())
        messages.extend([message, {'role': 'tool', 'tool_call_id': call['id'], 'content': '{"ok":true}'}])
        read, finish, _, wire = client.request(messages, choice('read_file'), False, tools, max_tokens=4096)
        report['wire'].append(wire)
        assert wire['HTTP_status'] == 200 and finish == 'tool_calls'
        calls = read.get('tool_calls', [])
        assert len(calls) == 1 and calls[0]['function']['name'] == 'read_file'
        read_call = calls[0]
        assert read_call.get('type') == 'function' and isinstance(read_call.get('id'), str) and read_call['id'] and read_call['id'] != call['id']
        assert json.loads(read_call['function']['arguments']) == {'path': 'report.json'}
        actual = (workspace / 'report.json').read_text(encoding='utf-8')
        messages.extend([read, {'role': 'tool', 'tool_call_id': read_call['id'], 'content': actual}])
        final, finish, _, wire = client.request(messages, 'none', False, tools, max_tokens=4096)
        report['wire'].append(wire)
        assert wire['HTTP_status'] == 200 and finish == 'stop' and not final.get('tool_calls')
        assert nonce in (final.get('content') or '') and '160' in (final.get('content') or '')
        report.update(passed=True, real_file_readback=True, final=final)
    except BaseException as exc:
        report['error'] = repr(exc)
    report.update(completed=True, actual_HTTP_requests=client.serial)
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ['p0-client', 'p0-sha256', 'tokenizer-adapter', 'tokenizer-sha256', 'base-url', 'model', 'output-dir', 'identity']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    p0 = load(args.p0_client, args.p0_sha256, 'reviewed_p0')
    tokenizer = load(args.tokenizer_adapter, args.tokenizer_sha256, 'official_tokenizer_adapter')
    report = run_checks(p0, tokenizer.count_tokens, args.base_url, args.model, pathlib.Path(args.output_dir), pathlib.Path(args.identity), args.timeout)
    report.update(p0_client_SHA256=args.p0_sha256, tokenizer_adapter_SHA256=args.tokenizer_sha256,
                  client_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (pathlib.Path(args.output_dir) / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(compact({key: report.get(key) for key in ['passed', 'actual_generated_tokens', 'actual_argument_tokens', 'actual_HTTP_requests', 'error']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
