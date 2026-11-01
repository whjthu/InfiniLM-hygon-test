"""Execute the extracted private r307 owner loop with CPU callbacks only."""
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import textwrap
import types
import weakref


class Tensor:
    def __init__(self, data): self.data = list(data)
    def item(self): return self.data[0]
    def __getitem__(self, i): return self.data[i]
    def __setitem__(self, i, v): self.data[i] = v
    def cpu(self): return self
    def tolist(self): return self.data
    def argmax(self, *_): return Tensor([7])
    def all(self): return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    source = a.repo / 'code/bw1000/public-native-bridge/GLM-prefill-W2-W4-owner-r307'
    owner = (source / 'owner_r290.py').read_text()
    block = textwrap.dedent(owner[owner.index('            request_completed_normally=False'):
                                 owner.index("            row={'sequence':seq")])
    compiled = compile(block, '<actual-private-r307-request-loop>', 'exec')
    spec = importlib.util.spec_from_file_location('private307_executor', source / 'glm_prefill_execute_r236.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(owner)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'windows' for t in n.targets)]
    assert len(assignments) == 1
    assert eval(compile(ast.Expression(assignments[0].value), '<immutable-plan>', 'eval'), {}) == (2, 4, 2, 4, 4, 2)
    assert "assert agreement.cpu().tolist()==[CARDS*w for w in windows]" in owner
    def case(n, stage, rank, window=4, limit=3, cancel_at=None, fail_at=None, eos=()):
        events = []
        live = weakref.WeakSet()
        counter = 0
        req = types.SimpleNamespace(cancel_event=types.SimpleNamespace(is_set=lambda: False),
              mark_abort=lambda _: None, emit_token=lambda token, visible: events.append(('emit', token)))
        def full(*_args, **_kwargs):
            assert not live, 'Two live logits buffers on one owner'
            v = Tensor([0]); live.add(v); return v
        def forward(_ctx, t, _rows, desc, *_args):
            nonlocal counter
            counter += 1
            events.append(('forward', int(desc[1]), t))
            if counter == fail_at:
                raise RuntimeError('injected Native failure')
            return 0
        def control(t):
            if len(t.data) == 1:
                events.append(('vote', counter))
                t.data[0] = float(cancel_at is not None and counter >= cancel_at)
            else:
                events.append(('select', counter)); t.data[:] = [7., 1.]
        torch = types.SimpleNamespace(tensor=lambda data, **_: Tensor(data),
                arange=lambda *_, **__: Tensor([]), full=full,
                zeros=lambda *_, **__: Tensor([0, 0]), isfinite=lambda _: Tensor([True]),
                int64=None, bfloat16=None, float32=None,
                cuda=types.SimpleNamespace(synchronize=lambda _: events.append(('sync',)),
                                          mem_get_info=lambda _: (2**30, 2**31)))
        chunks = lambda ids: [(i, ids[i:i + 512]) for i in range(0, len(ids), 512)]
        env = dict(execute_plan=module.execute_plan, torch=torch, ctypes=ctypes,
                   time=types.SimpleNamespace(monotonic=lambda: 0., perf_counter=lambda: 0.),
                   lib=types.SimpleNamespace(modern_glm_model_forward=forward,
                        modern_glm_model_reset=lambda _: events.append(('reset',)) or 0),
                   ctx=None, stage=stage, rank=rank, PP=4, is_emitter=stage == 3 and rank == 0,
                   req=req, FinishReason=types.SimpleNamespace(TIMEOUT='timeout'), deadline=5.,
                   limit=limit, prompt=[1] * n, past=0, generated=[], reason='length', calls=[],
                   report={}, CONTEXT=32768, vocab=16, prefill_chunks=chunks, prefill_window=window,
                   control=control, check=lambda _: None, ptr=lambda x: x, stream=lambda: None,
                   barrier=lambda: events.append(('barrier',)), eos=set(eos), backend_ticks=[])
        error = None
        try: exec(compiled, env)
        except RuntimeError as exc: error = str(exc)
        if fail_at is not None:
            assert error == 'injected Native failure'
            assert events[-1][0] == 'forward'
            assert not any(e[0] in ('reset', 'barrier', 'select', 'emit') for e in events)
            assert not env['report']['normal_shutdown']
        else:
            assert error is None and env['report']['normal_reset']
            assert events[-2:] == [('reset',), ('barrier',)]
            if cancel_at is not None:
                assert env['reason'] == 'canceled' and not env['generated']
                assert not any(e[0] in ('select', 'emit') for e in events)
            else:
                assert len(env['generated']) == (1 if 7 in eos else limit)
                prefill_count = (n + 511) // 512
                expected = [(i, min(512, n - i)) for i in range(0, n, 512)]
                expected += [(n + i, 1) for i in range(len(env['generated']) - 1)]
                assert [(e[1], e[2]) for e in events if e[0] == 'forward'] == expected
                assert [e[1] for e in events if e[0] == 'select'] == list(range(prefill_count, prefill_count + len(env['generated'])))
        return events
    counts = {'normal_all32': 0, 'cancel_all32': 0, 'failure_all32': 0, 'EOS_all32': 0}
    for stage in range(4):
        for rank in range(8):
            for n in [1, 513, 1025, 2049, 8900, 24576]:
                case(n, stage, rank); counts['normal_all32'] += 1
            for at in [0, 1, 4, 5]:
                events = case(2049, stage, rank, cancel_at=at)
                assert sum(e[0] == 'forward' for e in events) == min(5, ((at + 3) // 4) * 4)
                counts['cancel_all32'] += 1
            for at in range(1, 6):
                case(2049, stage, rank, fail_at=at); counts['failure_all32'] += 1
            case(2049, stage, rank, eos=(7,)); counts['EOS_all32'] += 1
    print(json.dumps({'passed': True, 'CPU_only': True, 'GPU': 0, 'Native_calls': 0,
                      'owner_source_SHA256': hashlib.sha256((source / 'owner_r290.py').read_bytes()).hexdigest(),
                      'counts': counts, 'one_live_logits_buffer_per_simulated_owner': True,
                      'decode_W1_and_final_only_sample': True,
                      'qualification': 'Extracted actual owner loop with CPU callbacks only; no actual32owner publication/CCL/device/Model/performance proof'}))


if __name__ == '__main__': main()
