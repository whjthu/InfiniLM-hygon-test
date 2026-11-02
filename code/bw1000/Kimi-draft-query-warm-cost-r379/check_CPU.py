"""Exercise the actual r333 dispatcher with recording math, not a GPU result."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import threading

D = pathlib.Path(__file__).resolve().parent
G = D.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--runtime-dir', type=pathlib.Path)
args = parser.parse_args()
S = args.runtime_dir or G/'code/bw1000/Kimi-long-draft-RPC-CPU-r333'
sys.path[:0] = [str(D), str(S)]
from warm_plan import process_plan, run_contexts, LENGTH
from long_rpc import DraftSession
from GPU_driver import supervise


class Context:
    def __init__(self):
        self.owner = threading.get_ident()
        self.epoch = 0
        self.phase = 'closed'
        self.length = 0
        self.raw = self.projected = None

    def _check(self, epoch, request):
        assert threading.get_ident() == self.owner and epoch == self.epoch and request == 1
        assert self.phase in ['prompt', 'decode']

    def begin(self, epoch, request, prompt_tokens):
        assert self.phase in ['closed', 'retired'] and epoch > self.epoch and request == 1
        self.epoch, self.phase, self.length = epoch, 'prompt', 0
        assert prompt_tokens == LENGTH
        self.prompt_tokens = prompt_tokens

    def append_prompt(self, epoch, request, past, part, fence):
        self._check(epoch, request)
        assert self.phase == 'prompt' and past == self.length
        fence()
        self.length += part
        self.raw = self.projected = self.length

    def seal_prompt(self, epoch, request):
        self._check(epoch, request)
        assert self.length == LENGTH
        self.phase = 'decode'

    def query(self, epoch, request, head, anchor, K, return_debug):
        self._check(epoch, request)
        assert self.length == LENGTH and self.phase == 'decode' and return_debug and K == 7
        return dict(mode='draft', values=(list(range(K)), {'recording_math_only': True}))

    def retire(self, epoch, request):
        self._check(epoch, request)
        self.phase = 'retired'
        self.raw = self.projected = None
        return True

    def close(self):
        self.phase = 'closed'
        self.raw = self.projected = None


class Codec:
    raw_dim, vocab = 1, 163840
    def __init__(self): self.fences, self.fail_query = 0, False
    def decode(self, body, rows): assert len(body) == rows*2; return rows
    def fence(self): self.fences += 1
    def proposals(self, values, debug, K):
        if self.fail_query: raise RuntimeError('Recorded query failure')
        return values


assert [row['warm_query'] for row in process_plan()] == [False, True, True, False]
checks = []
for plan in process_plan():
    c, codec = Context(), Codec()
    session = DraftSession(c, None, codec, enabled=True)
    history = []
    result = run_contexts(session, plan['warm_query'], lambda role, n: bytes([1 if role == 'priming' else 2])*n*2,
                          save=lambda value: history.append(json.loads(json.dumps(value))))
    assert result['completed'] and result['protocol_commands'] == (137 if plan['warm_query'] else 69)
    assert sum(phase['prefix_append_commands'] for phase in result['phases']) == (128 if plan['warm_query'] else 64)
    assert codec.fences == (132 if plan['warm_query'] else 66)
    assert history[-1]['normal_session_close'] and c.raw is c.projected is None
    checks.append('actual_dispatcher_process_'+str(plan['index']))
for failure in ['bad_frame', 'query_failure']:
    c, codec = Context(), Codec()
    codec.fail_query = failure == 'query_failure'
    session = DraftSession(c, None, codec, enabled=True)
    saved = []
    try:
        run_contexts(session, False, lambda role, n: b'' if failure == 'bad_frame' else b'\x01\x00'*n,
                     save=lambda value: saved.append(json.loads(json.dumps(value))))
    except (AssertionError, RuntimeError):
        assert not saved[-1]['completed']
        if failure == 'query_failure':
            assert saved[-1]['complete_query_error'] and saved[-1]['phases'][-1]['error']
        session.close()
        assert session.closed and c.raw is c.projected is None
        checks.append('no_retry_and_saved_failure_'+failure)
    else:
        raise AssertionError(failure)
def child(item):
    phases = [dict(role=role, complete_dispatch_wall_s=.1 if item['warm_query'] else .4,
                   handler_wall_s=.1, proposals=list(range(7)))
              for role in (['priming', 'measured'] if item['warm_query'] else ['measured'])]
    return dict(completed=True, passed=True, child_index=item['index'], warm_query=item['warm_query'],
                normal_session_close=True, normal_Runtime_close=True, PID=100+item['index'], PCI='same-test-PCI',
                feature_fixture={'measured': {'SHA256': 'same-frozen-input'}}, whole_child_wall_s=10.,
                query_plan=dict(phases=phases, protocol_commands=137 if item['warm_query'] else 69,
                                priming_query_count=int(item['warm_query']), measured_query_count=1))
r = supervise(child)
assert r['passed'] and r['measured_query']['ON_over_OFF'] == .25
assert r['all_child_wall_sum_s'] == 40 and r['no_priming_cost_removed_from_startup']
checks.append('actual_supervisor_four_independent_processes_complete_costs')
for failure in ['failed_child', 'exception', 'other_PCI', 'missing_priming']:
    calls, saved = [], []
    def bad(item):
        calls.append(item['index'])
        if failure == 'exception' and item['index'] == 1:
            raise TimeoutError('Original whole-step budget')
        row = child(item)
        if item['index'] == 1:
            if failure == 'failed_child': row['passed'] = False
            elif failure == 'other_PCI': row['PCI'] = 'different-test-PCI'
            elif failure == 'missing_priming': row['query_plan']['priming_query_count'] = 0
        return row
    try:
        supervise(bad, lambda value: saved.append(json.loads(json.dumps(value))))
    except (AssertionError, TimeoutError):
        assert calls == ([0, 1, 2, 3] if failure == 'other_PCI' else [0, 1])
        assert saved and not saved[-1]['completed']
        checks.append('supervisor_stop_without_retry_'+failure)
    else:
        raise AssertionError(failure)
print(json.dumps(dict(passed=True, CPU_only=True, GPU_initialized=False, checkpoint_loaded=False,
    checks=checks, actual_dispatcher_SHA256=hashlib.sha256((S/'long_rpc.py').read_bytes()).hexdigest(),
    scope='Original DraftSession + recording feature/math providers; no trained query, no priming benefit proved')))
