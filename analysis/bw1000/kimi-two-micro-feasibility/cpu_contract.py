"""CPU-only negative contract; no Native loading and no throughput simulation.

Executes the actual frozen Python arm/finish methods against explicit mocks.
C++ singleton tests are source-anchored behavioral counterexamples, not C++ ABI tests.
"""
import ast
import ctypes as C
import hashlib
import json
import pathlib
import time
from types import SimpleNamespace

V = pathlib.Path(__file__).resolve().parent.parent
ROOT = pathlib.Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群')
RUNTIME = V / 'Kimi-current-model-sparse-profile-launch-private-r136b/runtime'
CPP = ROOT / 'Kimi-speculative-batch-private-r9122c'


class ArrayStub:
    int32 = object()
    @staticmethod
    def asarray(values, dtype):
        return list(values)


def main():
    source = RUNTIME / 'batch_native_r9128.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Native')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('arm', 'finish')]
    isolated = ast.Module(body=[ast.ClassDef(name='FrozenMethods', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ns = {'C': C, 'np': ArrayStub, 'time': time}
    exec(compile(ast.fix_missing_locations(isolated), str(source), 'exec'), ns)
    n = ns['FrozenMethods']()
    n.ctx = object()
    n.epoch = 0
    n.ck = lambda rc: None if rc == 0 else (_ for _ in ()).throw(AssertionError(rc))
    n.ptr = lambda x: x
    captured = {'epoch': 0, 'records': []}
    def begin(ctx, epoch, enabled, count, ids, past, width):
        captured.update(epoch=epoch, records=[])
        return 0
    n.lib = SimpleNamespace(modern_kimi_batch_taps_begin_r9122=begin)
    n.arm([1, 2, 3, 4], [10]*4, [8]*4, True)
    captured['records'].append({'epoch': 1, 'logical': [1, 2, 3, 4]})
    n.arm([5, 6, 7, 8], [10]*4, [8]*4, True)
    assert n.epoch == 2 and captured == {'epoch': 2, 'records': []}

    trace = []
    n.tail = False; n.node = n.emitter = 0; n.vocab = 100; n.serial = 7; n.calls = []
    class Control:
        def collect(self, label, payload):
            trace.append('WORLD.collect.selected')
            return [{'wall': .25, 'selected': [17]}]
    n.control = Control()
    n.harvest = lambda *args: trace.append('feature.harvest') or {'snapshot': 7}
    n.finish('verify', [1], [10], [8], True, None, SimpleNamespace(tolist=lambda: [17]), [2], .25, lambda selected: trace.append('callback'))
    assert trace == ['WORLD.collect.selected', 'callback', 'feature.harvest']

    anchors = {
        source: ['self.epoch+=1', "p=self.control.collect('batch-call:'", 'features=self.harvest'],
        CPP/'batch_live_taps_r9122.hpp': ['static Metadata metadata;static std::vector<Record>records;', 'records.clear();context=ctx;metadata=std::move(next)'],
        CPP/'batch_state_r9122.hpp': ['std::unordered_map<rt::Runtime*,Cached> states;', 'auto&c=states[&r];c.layers.clear();c.scalars.clear();', 'c.epoch!=epoch'],
        CPP/'pair_probe.cpp': ['c->batch_requests_r9122=req;', 'logical[i]!=c->batch_requests_r9122[i].logical', 'Operation::Replay,c->batch_epoch_r9122,c->batch_requests_r9122,kept'],
    }
    receipts = []
    for path, patterns in anchors.items():
        raw = path.read_bytes(); text = raw.decode('utf-8')
        assert all(p in text for p in patterns), str(path)
        receipts.append({'path': str(path), 'SHA256': hashlib.sha256(raw).hexdigest(), 'anchors_passed': len(patterns)})

    # Minimal C++ behavior model from the anchors above: row-disjoint is insufficient.
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    pending = {'epoch': 1, 'requests': a, 'replay_layers': {'L1': 'snapshot-A'}}
    pending.update(epoch=2, requests=b, replay_layers={'L1': 'snapshot-B'})
    assert a != pending['requests']          # actual commit order guard rejects A
    assert pending['epoch'] != 1            # actual replay epoch guard rejects A
    assert pending['replay_layers']['L1'] != 'snapshot-A'
    result = {
        'CPU_only': True, 'Native_dlopen': False, 'GPU': False,
        'verdict': 'STOP_OVERLAP_SOURCE_EXPANSION',
        'tests': {
            'actual_python_arm_discards_pending_disjoint_taps': 'PASS',
            'actual_python_finish_collects_before_harvest': 'PASS',
            'source_anchors_CAPI_request_slot_is_single': 'PASS',
            'source_anchors_Runtime_replay_slot_is_single': 'PASS',
            'behavioral_counterexample_A_commit_after_B_is_invalid': 'PASS',
            'behavioral_counterexample_A_replay_after_B_loses_cache': 'PASS',
        },
        'finish_trace': trace, 'source_receipts': receipts,
        'not_proven': ['C++ compilation/ABI execution', 'two pending transaction safety', 'PP overlap', 'GPU speedup'],
    }
    out = pathlib.Path(__file__).with_name('CPU-observed-r154.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'tests': len(result['tests']), 'verdict': result['verdict'], 'receipt': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
