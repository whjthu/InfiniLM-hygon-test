"""CPU fixtures only; fake replies/token counts never qualify a real model."""
import importlib.util
import json
import pathlib
import re
import tempfile
import types
import uuid

from client_long_output import compact, run_checks

ROOT = pathlib.Path(__file__).resolve().parent
source = pathlib.Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/deep-agent-service-p0-r143e/client_p0_r143.py')
spec = importlib.util.spec_from_file_location('cpu_p0_fixture', source)
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


class FakeClient:
    generated_tokens = 2048
    rows = 160
    def __init__(self, url, model, path, budget, timeout):
        assert budget == 4096
        self.serial = 0
    def request(self, messages, choice='none', stream=False, tools=None, max_tokens=None):
        assert max_tokens == 4096
        self.serial += 1
        task = messages[1]['content']
        nonce = re.search(r'nonce "([a-f0-9]+)"', task)[1]
        total = int(re.search(r'where total is (\d+)', task)[1])
        if isinstance(choice, dict):
            name = choice['function']['name']
            args = {'path': 'report.json'}
            if name == 'write_file':
                args['content'] = compact({'nonce': nonce, 'rows': [{'i': i, 'total': total} for i in range(self.rows)]})
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_' + uuid.uuid4().hex, 'type': 'function', 'function': {'name': name, 'arguments': compact(args)}}]}
            finish = 'tool_calls'
        else:
            message, finish = {'role': 'assistant', 'content': nonce + ' 160 rows'}, 'stop'
        usage = {'prompt_tokens': 2048, 'completion_tokens': self.generated_tokens, 'total_tokens': 2048 + self.generated_tokens}
        return message, finish, usage, {'HTTP_status': 200, 'usage': usage}


fixture = types.SimpleNamespace(WireClient=FakeClient, TOOLS=p0.TOOLS)
with tempfile.TemporaryDirectory(prefix='long-tool-output-cpu-') as folder:
    root = pathlib.Path(folder)
    identity = root / 'identity.json'
    identity.write_text('{"CPU_fixture":true,"weights_loaded":false}')
    run = lambda label, counter: run_checks(fixture, counter, 'fake://CPU', 'CPU_FIXTURE', root / label, identity)
    assert run('good', lambda text: 2048)['passed']
    assert not run('short-argument', lambda text: 512)['passed']
    FakeClient.generated_tokens = 512
    assert not run('short-generation', lambda text: 2048)['passed']
    FakeClient.generated_tokens = 2048
    FakeClient.rows = 159
    assert not run('omitted-row', lambda text: 2048)['passed']
evidence = {'status': 'PASSED', 'CPU_fixtures_only': True, 'token_counts_are_fake': True, 'weights_loaded': False,
            'checks': ['real_file_write_readback_logic', 'all_requests_reserve4096', 'reject_short_argument', 'reject_short_generation', 'reject_omitted_rows'], 'real_model_acceptance': 'NOT_RUN'}
(ROOT / 'long-output-client-cpu-check.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
