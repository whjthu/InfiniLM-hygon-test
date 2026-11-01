import argparse, datetime, hashlib, json, pathlib, secrets, time
import urllib.request, urllib.error

def sha(b):
    return hashlib.sha256(b).hexdigest()

def compact(v):
    return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read file.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'calculate', 'description': 'Add integers.', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'write_file', 'description': 'Write file.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content'], 'additionalProperties': False}}}]

def sse_message(raw):
    message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
    calls = {}
    finish = None
    usage = None
    done = False
    for line in raw.decode().splitlines():
        if not line or line.startswith(':'):
            continue
        if not line.startswith('data: '):
            raise AssertionError('Invalid SSE frame')
        value = line[6:]
        if value == '[DONE]':
            done = True
            break
        q = json.loads(value)
        if 'error' in q:
            raise RuntimeError(compact(q))
        if q.get('usage') is not None:
            usage = q['usage']
        for c in q.get('choices', []):
            assert c.get('index', 0) == 0
            delta = c.get('delta', {})
            for key in ['content', 'reasoning_content']:
                if delta.get(key):
                    message[key] += delta[key]
            for part in delta.get('tool_calls', []):
                entry = calls.setdefault(part.get('index', 0), {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    entry['id'] += part['id']
                if part.get('type'):
                    assert part['type'] == 'function'
                for key in ['name', 'arguments']:
                    if part.get('function', {}).get(key):
                        entry['function'][key] += part['function'][key]
            if c.get('finish_reason') is not None:
                finish = c['finish_reason']
    assert done and finish is not None, 'Incomplete SSE; no tools may execute'
    if calls:
        message['tool_calls'] = [calls[k] for k in sorted(calls)]
    if not message['reasoning_content']:
        message.pop('reasoning_content')
    return (message, finish, usage)

class WireClient:

    def __init__(self, url, model, path, max_tokens=4096, timeout=7200):
        self.url = url.rstrip('/') + '/chat/completions'
        self.model = model
        self.path = path
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.serial = 0

    def request(self, messages, choice='none', stream=False, tools=None, max_tokens=None, thinking=False, parallel=False):
        self.serial += 1
        stem = self.path / ('request-%03d' % self.serial)
        body = {'model': self.model, 'messages': messages, 'temperature': 0, 'max_tokens': self.max_tokens if max_tokens is None else max_tokens, 'stream': stream, 'tool_choice': choice, 'parallel_tool_calls': parallel, 'chat_template_kwargs': {'enable_thinking': thinking}}
        if getattr(self, 'session', None):
            body['user'] = self.session
        if tools is not None:
            body['tools'] = tools
        if stream:
            body['stream_options'] = {'include_usage': True}
        stem.with_suffix('.request.json').write_bytes((compact(body) + '\n').encode())
        started = time.monotonic()
        req = urllib.request.Request(self.url, data=compact(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = response.status
                raw = response.read()
                headers = dict(response.headers)
        except urllib.error.HTTPError as e:
            status = e.code
            raw = e.read()
            headers = dict(e.headers)
        stem.with_suffix('.wire').write_bytes(raw)
        ev = {'request': stem.with_suffix('.request.json').name, 'wire': stem.with_suffix('.wire').name, 'wire_SHA256': sha(raw), 'HTTP_status': status, 'elapsed_s': time.monotonic() - started, 'stream': stream}
        if status != 200:
            return ({'error': json.loads(raw) if raw.startswith(b'{') else raw.decode(errors='replace')}, None, None, ev)
        if stream:
            content_type = next((value for key, value in headers.items() if key.lower() == 'content-type'), '')
            assert 'text/event-stream' in content_type.lower(), 'Unexpected SSE Content-Type: ' + content_type
            message, finish, usage = sse_message(raw)
        else:
            q = json.loads(raw)
            assert 'error' not in q
            message = q['choices'][0]['message']
            finish = q['choices'][0]['finish_reason']
            usage = q.get('usage')
        assert isinstance(usage, dict), 'Missing actual token usage'
        assert all((type(usage[k]) is int and usage[k] >= 0 for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']))
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        ev['finish_reason'] = finish
        ev['usage'] = usage
        stem.with_suffix('.decoded.json').write_bytes((compact({'message': message, 'finish_reason': finish, 'usage': usage}) + '\n').encode())
        return (message, finish, usage, ev)

def negative(client, kind):
    if kind == 'invalid_history':
        messages = [{'role': 'tool', 'tool_call_id': 'no-pending-id', 'content': '{}'}]
        tools = TOOLS
        limit = 24
        choice = 'auto'
        expected = 400
    elif kind == 'invalid_schema':
        messages = [{'role': 'user', 'content': 'Hi'}]
        tools = [{'type': 'function', 'function': {'name': 'bad', 'parameters': {'type': 'not-a-type'}}}]
        limit = 24
        choice = 'auto'
        expected = 400
    else:
        messages = [{'role': 'user', 'content': 'Use calculate to add 12 and 34.'}]
        tools = [TOOLS[1]]
        limit = 1
        choice = 'required'
        expected = None
    message, finish, usage, e = client.request(messages, choice, False, tools, limit)
    if expected is not None:
        assert e['HTTP_status'] == expected
    else:
        assert e['HTTP_status'] != 200 or (finish != 'tool_calls' and (not message.get('tool_calls')))
    return {'passed': True, 'wire': e, 'response': message, 'tools_executed': 0}
