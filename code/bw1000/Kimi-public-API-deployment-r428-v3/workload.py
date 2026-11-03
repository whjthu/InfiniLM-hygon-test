"""Two immutable actual CPU-encoded complete histories; no generated tools run."""
from pathlib import Path
import hashlib
import json

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sha = lambda b: hashlib.sha256(b).hexdigest()


def host(path):
    p = Path(path)
    return R / p.relative_to('/refactor')  # Strict known prefix; no arbitrary replacement.


def opaque_history(body):
    calls, pairs, pending = {}, [], set()
    for message in body['messages']:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls', []):
                key = call['id'];assert isinstance(key, str) and key and key not in calls
                calls[key] = call;pending.add(key)
        elif message['role'] == 'tool':
            key = message['tool_call_id'];assert key in pending;pending.remove(key)
            pairs.append(dict(id=key, function=calls[key]['function']['name'],
                result_SHA256=sha(message['content'].encode())))
    assert not pending
    return pairs


def body_and_gold(fixture):
    original = host(fixture['source_path']).read_bytes()
    assert sha(original) == fixture['source_SHA256']
    raw = host(fixture['private_encoded_case']).read_bytes()
    assert sha(raw) == fixture['private_encoded_SHA256']
    encoded = json.loads(raw);body = encoded['body']
    assert len(encoded['prompt_token_ids']) == fixture['current_prompt_tokens']
    assert sha(json.dumps(encoded['prompt_token_ids'], separators=(',', ':')).encode()) == fixture['current_prompt_IDs_SHA256']
    source = json.loads(original)
    source = source['body'] if fixture['source_kind'] == 'fixed-wrapper' else source
    assert body['messages'] == source['messages'] and body['tools'] == source['tools']
    assert body['max_tokens'] == fixture['output_budget'] and body['stream'] == fixture['stream']
    assert len(encoded['prompt_token_ids']) + body['max_tokens'] <= 32768
    data = next(json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool'
        and isinstance(json.loads(m['content']), dict) and 'records' in json.loads(m['content']))
    values = {r['id']: r['value'] for r in data['records']}
    selected = [values[key] for key in data['query_ids']]
    assert all(type(v) is int for v in selected)
    expected = fixture['gold']
    if expected['function'] == 'calculate': assert expected['arguments'] == dict(a=selected[0], b=selected[1])
    else: assert expected['content'] == dict(nonce=data['nonce'], total=sum(selected))
    request = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    assert sha(request) == fixture['actual_request_SHA256']
    return request, body, expected


def validate_reply(wire, fixture, expected, history):
    assert 'error' not in wire and isinstance(wire['id'], str) and wire['id']
    choice = wire['choices'][0];assert choice['finish_reason'] == 'tool_calls'
    calls = choice['message'].get('tool_calls', []);assert len(calls) == 1
    call = calls[0];assert call['type'] == 'function'
    assert isinstance(call['id'], str) and call['id'] and call['id'] not in {r['id'] for r in history}
    assert call['function']['name'] == expected['function']
    args = json.loads(call['function']['arguments'])
    if expected['function'] == 'calculate':
        assert args == expected['arguments'] and all(type(args[k]) is int for k in ['a', 'b'])
    else:
        assert args['path'] == expected['path']
        content = json.loads(args['content'])
        assert content == expected['content'] and type(content['total']) is int
    usage = wire['usage']
    assert all(type(usage[k]) is int for k in ['prompt_tokens', 'completion_tokens', 'total_tokens'])
    assert usage['prompt_tokens'] == fixture['current_prompt_tokens']
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    assert 1 <= usage['completion_tokens'] < fixture['output_budget']
    return usage, call
