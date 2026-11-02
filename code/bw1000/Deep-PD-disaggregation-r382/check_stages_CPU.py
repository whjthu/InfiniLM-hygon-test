"""Compare actual old generation consumer with split P/D using recording state.

The recording state copy is deliberately not a Native cache proof. A real
imported-state qualification is still required before any Model/PD claim.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from native_stages import prefill_until_first, decode_from_import

ORIGINAL = Path(__file__).resolve().parents[1] / 'Deep-HC-T1-Model-r366/owner-v2/native_decode_r206b.py'
SPEC = importlib.util.spec_from_file_location('original_native_decode', ORIGINAL)
OLD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(OLD)
REASONS = SimpleNamespace(CANCELED='cancel', LENGTH='length', EOS_TOKEN='eos')
os.environ['DEEP_R277_GPUSTART_UNIX'] = str(time.time())


class Request:
    def __init__(self, ids, budget):
        self.prompt_token_ids, self.max_new_tokens = tuple(ids), budget
        self.cancel_event, self.abort_reason = threading.Event(), None
        self.tokens, self.events = [], []

    def emit_token(self, token):
        self.tokens.append(token); self.events.append(('emit', token))


class RecordingNative:
    def __init__(self, request, eos_total, cancel_call=None):
        self.req, self.eos_total, self.cancel_call = request, eos_total, cancel_call
        self.state = dict(past=0, sum=0)
        self.calls = []

    def forward(self, ids, positions, rows, past, total, cu):
        assert rows == [0] and past == [self.state['past']] and total == [past[0] + len(ids)]
        assert positions == list(range(past[0], total[0])) and cu == [0, len(ids)]
        self.calls.append(copy.deepcopy((ids, positions, rows, past, total, cu)))
        self.req.events.append(('forward', len(ids), past[0]))
        self.state.update(past=total[0], sum=self.state['sum'] + sum(ids))
        if self.cancel_call == len(self.calls):
            self.req.cancel_event.set()
        token = 1 if self.eos_total is not None and total[0] >= self.eos_total else 33 + self.state['sum'] % 97
        return 0, 0.0, [token]


def check_rc(value):
    if value:
        raise RuntimeError('Recorded Native error')


def finite(value):
    if not math.isfinite(value):
        raise RuntimeError('Recorded nonfinite logit')


def split(req, P, D, *, context=32768, chunk=2):
    carry = prefill_until_first(req, P.forward, check_rc, finite, REASONS, lambda: None,
                                context=context, chunk_size=chunk)
    trace = carry['trace']
    if not carry['handoff_required']:
        return carry['finish_reason'], trace
    assert req.tokens == [carry['first_token']]
    req.events.append(('snapshot_after_first_emit', carry['past']))
    D.state = copy.deepcopy(P.state)  # Recording only; not the actual Native codec.
    reason, tail = decode_from_import(req, carry['past'], carry['first_token'],
                    D.forward, check_rc, finite, REASONS, lambda: None, context=context)
    return reason, trace + tail


def main():
    checks = []
    for budget, eos_total, chunk in [(128, 7, 2), (128, 5, 2), (1, None, 2),
                                      (2, None, 2), (5, None, 3), (128, 6, 4096)]:
        req0, req1 = Request([2, 3, 4, 5, 6], budget), Request([2, 3, 4, 5, 6], budget)
        original, P, D = RecordingNative(req0, eos_total), RecordingNative(req1, eos_total), RecordingNative(req1, eos_total)
        old_reason, old_trace = OLD.generate(req0, original.forward, check_rc, finite, REASONS,
                                             chunk_size=chunk)
        reason, trace = split(req1, P, D, chunk=chunk)
        assert (reason, trace, req1.tokens) == (old_reason, old_trace, req0.tokens)
        assert P.calls + D.calls == original.calls and sum(len(c[0]) for c in P.calls) == 5
        assert all(len(c[0]) == 1 and c[3][0] >= 5 for c in D.calls)
        if D.calls:
            assert D.calls[0] == ([req1.tokens[0]], [5], [0], [5], [6], [0, 1])
            assert next(i for i, e in enumerate(req1.events) if e[0] == 'emit') < next(
                i for i, e in enumerate(req1.events) if e[0] == 'snapshot_after_first_emit')
        checks.append(dict(budget=budget, eos_total=eos_total, chunk=chunk,
                           matches_original_tokens_trace_calls=True,
                           P_input_tokens=5, D_prompt_tokens=0, D_decode_calls=len(D.calls)))
    for cancel_call in [1, 3, 4]:
        r0, r1 = Request([2, 3, 4, 5, 6], 128), Request([2, 3, 4, 5, 6], 128)
        old = RecordingNative(r0, None, cancel_call)
        P, D = RecordingNative(r1, None, cancel_call), RecordingNative(r1, None, 1 if cancel_call == 4 else None)
        a = OLD.generate(r0, old.forward, check_rc, finite, REASONS, chunk_size=2)
        b = split(r1, P, D)
        assert a == b and r0.tokens == r1.tokens
        checks.append(dict(cancel_original_call=cancel_call, original_cancellation_matched=True))
    for budget, context in [(True, 32768), (0, 32768), (128, 132)]:
        r = Request([2, 3, 4, 5, 6], budget); P, D = RecordingNative(r, None), RecordingNative(r, None)
        try:
            split(r, P, D, context=context)
        except ValueError:
            assert not P.calls and not D.calls and not r.tokens
        else:
            raise AssertionError('Invalid budget/context accepted')
        checks.append(dict(rejected_budget=budget, context=context, no_forward=True))
    r = Request([2, 3, 4, 5, 6], 128); P, D = RecordingNative(r, None), RecordingNative(r, None)
    carry = prefill_until_first(r, P.forward, check_rc, finite, REASONS, lambda: None, chunk_size=2)
    # No fake reconstruction from IDs: a missing actual import must fail the
    # first D forward against its still-empty state.
    try:
        decode_from_import(r, carry['past'], carry['first_token'], D.forward,
                           check_rc, finite, REASONS, lambda: None)
    except AssertionError:
        assert not D.calls
    else:
        raise AssertionError('Missing state import silently recomputed prompt')
    checks.append(dict(missing_import_refused=True, D_prompt_replay=False))
    print(json.dumps(dict(passed=True, checks=checks,
                          original_generation_SHA256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                          CPU_recording_state_only=True, Native_cache_qualified=False,
                          GPU=0, no_PD_performance_claim=True)))


if __name__ == '__main__':
    main()
