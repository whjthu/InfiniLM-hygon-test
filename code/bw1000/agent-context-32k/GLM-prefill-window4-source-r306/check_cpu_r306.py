"""CPU callback-contract checks against the exact current W1/W2 executor."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute_plan


def run(fn, chunks, *, window, prefill=True, cancel_at=None, fail_at=None, sample_fail=False, vote_fail_at=None):
    events = []
    forwards = 0
    votes = 0
    def forward(past, values, final):
        nonlocal forwards
        forwards += 1
        events.append(('forward', past, tuple(values), final))
        if forwards == fail_at:
            raise RuntimeError('injected forward failure')
    def vote():
        nonlocal votes
        votes += 1
        events.append(('vote', forwards))
        if votes == vote_fail_at:
            raise RuntimeError('injected vote failure')
        return cancel_at is not None and forwards >= cancel_at
    def sample():
        events.append(('sample', forwards))
        if sample_fail:
            raise RuntimeError('injected sample failure')
        return 7
    result, error = None, None
    try:
        result = fn(chunks, chunks[0][0], prefill=prefill, window=window,
                    forward=forward, vote=vote, sample=sample,
                    discard=lambda: events.append(('discard',)))
    except RuntimeError as exc:
        error = str(exc)
    return result, error, events


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    args = p.parse_args()
    root = args.repo.resolve()
    base = root / 'code/bw1000/public-native-bridge/GLM-latest-full-owner-r299/glm_prefill_execute_r236.py'
    candidate = Path(__file__).with_name('glm_prefill_execute_r306.py')
    assert hashlib.sha256(base.read_bytes()).hexdigest() == '788d0ab66d1f84cdafbff684e7878fd28c0b3ecccadda46527e305e88042c0b9'
    old, new = load(base, 'reference236'), load(candidate, 'candidate306')
    # Only the accepted-window guard and the group extent may differ in code.
    oldtree = ast.parse(base.read_bytes())
    newtree = ast.parse(candidate.read_bytes())
    oldfn = next(n for n in oldtree.body if isinstance(n, ast.FunctionDef))
    newfn = next(n for n in newtree.body if isinstance(n, ast.FunctionDef))
    oldfn.body[0] = newfn.body[0]
    oldgroup = oldfn.body[-1].body[0].orelse[1]
    newgroup = newfn.body[-1].body[0].orelse[1]
    oldgroup.iter = newgroup.iter
    oldgroup.body[0] = newgroup.body[0]
    # The original finally contains comments only in addition to discard().
    assert ast.dump(oldfn, include_attributes=False) == ast.dump(newfn, include_attributes=False)
    counts = {'W1_W2_trace_equivalence': 0, 'W4_complete_input_and_boundaries': 0,
              'W4_cancel': 0, 'W4_failure_no_drain': 0, 'decode_W1': 0,
              'W4_vote_or_sample_failure': 0, 'nonzero_prefill_past': 0, 'invalid_pre_forward': 0}
    for n in [1, 512, 513, 1024, 1025, 1536, 2048, 2049, 3584, 4096, 4097, 8900, 17455, 24576, 32768]:
        ids = [i % 100 for i in range(n)]
        chunks = [(i, ids[i:i + 512]) for i in range(0, n, 512)]
        for width in [1, 2]:
            for options in [{}, *[{'cancel_at': i} for i in [0, 1, 2, len(chunks)]],
                            *[{'fail_at': i} for i in [1, len(chunks)]], {'sample_fail': True}]:
                assert run(old, chunks, window=width, **options) == run(new, chunks, window=width, **options)
                counts['W1_W2_trace_equivalence'] += 1
        result, error, events = run(new, chunks, window=4)
        assert result == (n, 7, False) and error is None
        calls = [e for e in events if e[0] == 'forward']
        assert [(e[1], list(e[2])) for e in calls] == chunks
        assert [e[1] for e in events if e[0] == 'vote'] == [0] + list(range(4, len(chunks), 4)) + [len(chunks)]
        assert [e[1] for e in events if e[0] == 'sample'] == [len(chunks)]
        assert events[-1] == ('discard',)
        counts['W4_complete_input_and_boundaries'] += 1
        for at in range(len(chunks) + 1):
            result, error, events = run(new, chunks, window=4, cancel_at=at)
            committed = min(len(chunks), ((at + 3) // 4) * 4)
            assert result == (sum(len(v) for _, v in chunks[:committed]), None, True) and error is None
            assert not any(e[0] == 'sample' for e in events)
            assert events[-1] == ('discard',)
            counts['W4_cancel'] += 1
        for at in range(1, len(chunks) + 1):
            _, error, events = run(new, chunks, window=4, fail_at=at)
            assert error == 'injected forward failure'
            assert events[-2][0] == 'forward' and events[-1] == ('discard',)
            assert not any(e[0] == 'sample' for e in events)
            counts['W4_failure_no_drain'] += 1
        for at in range(1, (len(chunks) + 3) // 4 + 2):
            _, error, events = run(new, chunks, window=4, vote_fail_at=at)
            assert error == 'injected vote failure'
            assert events[-2][0] == 'vote' and events[-1] == ('discard',)
            assert not any(e[0] == 'sample' for e in events)
            counts['W4_vote_or_sample_failure'] += 1
        _, error, events = run(new, chunks, window=4, sample_fail=True)
        assert error == 'injected sample failure' and events[-2][0] == 'sample' and events[-1] == ('discard',)
        counts['W4_vote_or_sample_failure'] += 1
    for past, length in [(17, 513), (8192, 2049), (32255, 513)]:
        chunks = [(past + i, [1] * min(512, length - i)) for i in range(0, length, 512)]
        result, error, _ = run(new, chunks, window=4)
        assert result == (past + length, 7, False) and error is None
        counts['nonzero_prefill_past'] += 1
    for past in [0, 511, 8192, 32767]:
        chunks = [(past, [1])]
        assert run(new, chunks, window=4, prefill=False) == run(old, chunks, window=1, prefill=False)
        counts['decode_W1'] += 1
    invalid = [([(0, [1])], 0, w) for w in [True, False, None, 0, 3, 8, 2., '4']]
    invalid += [([], 0, 4), ([(1, [1])], 0, 4), ([(0, [1] * 513)], 0, 4),
                ([(32768, [1])], 32768, 4), ([(0, [])], 0, 4), ([(0, [1])], True, 4)]
    for chunks, past, window in invalid:
        events = []
        try:
            new(chunks, past, prefill=True, window=window,
                forward=lambda *_: events.append('forward'), vote=lambda: False,
                sample=lambda: 7, discard=lambda: events.append('discard'))
        except ValueError:
            assert 'forward' not in events
        else:
            raise AssertionError('Invalid plan accepted')
        counts['invalid_pre_forward'] += 1
    print(json.dumps({'passed': True, 'CPU_only': True, 'GPU': 0, 'Native': 0,
                      'baseline_SHA256': hashlib.sha256(base.read_bytes()).hexdigest(),
                      'candidate_SHA256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      'counts': counts, 'default_enabled': False,
                      'qualification': 'Callback control/source equivalence only; no CCL liveness, device ownership, numerical, Model or performance qualification'}))


if __name__ == '__main__':
    main()
