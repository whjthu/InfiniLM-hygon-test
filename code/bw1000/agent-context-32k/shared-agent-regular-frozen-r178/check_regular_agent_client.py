"""CPU fixtures only; reported token counts are fake and never model evidence."""
import importlib.util
import json
import pathlib
import tempfile
import time
import types
import uuid

from client_regular_agent import compact, make_input, run_checks

ROOT = pathlib.Path(__file__).resolve().parent
source = pathlib.Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/deep-agent-service-p0-r143e/client_p0_r143.py')
spec = importlib.util.spec_from_file_location('cpu_p0_fixture', source)
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


class FakeClient:
    short_context = False
    duplicate_ids = False
    def __init__(self, url, model, path, max_tokens, timeout):
        self.path, self.serial = path, 0
    def request(self, messages, choice='none', stream=False, tools=None, max_tokens=None, thinking=False, parallel=False):
        self.serial += 1
        time.sleep(0.015)
        event = {'HTTP_status': 200, 'stream': stream}
        history = [json.loads(row['content']) for row in messages if row['role'] == 'tool']
        length = 16384 if self.path.name == 'SSE' else 8192
        if not history or self.short_context:
            length = 64
        usage = {'prompt_tokens': length, 'completion_tokens': 10, 'total_tokens': length + 10}
        event['usage'] = usage
        if messages[0]['role'] == 'tool':
            return {'error': 'CPU invalid history'}, None, None, dict(event, HTTP_status=400)
        if max_tokens == 1:
            return {'error': 'CPU truncated'}, None, None, dict(event, HTTP_status=502)
        source = next((row for row in history if 'records' in row), None)
        prior = [call for row in messages for call in row.get('tool_calls', [])]
        if isinstance(choice, dict):
            name = choice['function']['name']
            if name == 'read_file':
                args = {'path': 'input.json' if not prior else 'result.json' if len(prior) == 4 else 'final.json'}
            else:
                values = {row['id']: row['value'] for row in source['records']}
                a, b, c, d = [values[key] for key in source['query_ids']]
                sums = [a + b, c + d]
                if name == 'calculate':
                    count = sum(call['function']['name'] == name for call in prior)
                    left, right = [(a, b), (c, d), tuple(sums)][count]
                    args = {'a': left, 'b': right}
                else:
                    final = any(call['function']['name'] == name for call in prior)
                    value = {'nonce': source['nonce'], 'total': sum(sums)} if final else {'nonce': source['nonce'], 'first_sum': sums[0], 'second_sum': sums[1]}
                    args = {'path': 'final.json' if final else 'result.json', 'content': compact(value)}
            call_id = 'call_' + str(self.serial) if self.duplicate_ids else 'call_' + uuid.uuid4().hex
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': compact(args)}}]}
            finish = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': compact(history[-1]) if history else '2'}
            finish = 'stop'
        if thinking:
            message['reasoning_content'] = 'CPU fixture reasoning'
        return message, finish, usage, event


fixture = types.SimpleNamespace(WireClient=FakeClient, TOOLS=p0.TOOLS, negative=p0.negative)
with tempfile.TemporaryDirectory(prefix='regular-agent-cpu-') as folder:
    root = pathlib.Path(folder)
    identity = root / 'identity.json'
    identity.write_text('{"CPU_fixture":true,"weights_loaded":false}')
    profiles = []
    for name, floor, budget, stream, thinking in [('HTTP', 8192, 4096, False, False), ('SSE', 16384, 4096, True, False), ('thinking', 8192, 8192, False, True)]:
        path = root / (name + '.json')
        path.write_text(compact(make_input(12)))
        profiles.append({'label': name, 'input_file': str(path), 'minimum_prompt_tokens': floor, 'output_budget': budget, 'stream': stream, 'thinking': thinking})
    result = run_checks(fixture, 'fake://CPU', 'CPU_FIXTURE', root / 'good', identity, profiles, 30)
    assert result['passed'] and result['actual_HTTP_requests'] == 30
    FakeClient.short_context = True
    short = run_checks(fixture, 'fake://CPU', 'CPU_FIXTURE', root / 'short', identity, profiles, 30)
    assert not short['passed'] and not short['isolation_checks']['all_three_eight_step_workflows']
    FakeClient.short_context = False
    FakeClient.duplicate_ids = True
    duplicate = run_checks(fixture, 'fake://CPU', 'CPU_FIXTURE', root / 'duplicate', identity, profiles, 30)
    assert not duplicate['passed'] and not duplicate['isolation_checks']['unique_tool_call_ids']
evidence = {'status': 'PASSED', 'CPU_fixtures_only': True, 'token_counts_are_fake': True, 'weights_loaded': False,
            'checks': ['three_eight_step_workflows', '30_requests', 'reject_short_actual_prompt_counts', 'reject_cross_session_duplicate_ids', 'reasoning_preserved', 'full_file_results_retained'],
            'real_model_acceptance': 'NOT_RUN'}
(ROOT / 'regular-client-cpu-check.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
