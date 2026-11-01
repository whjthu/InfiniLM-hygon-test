"""Real new prefill consumer plus unchanged W4/decode functions; recording CAPI.

No Torch/Native/weights, network, or GPU; no Model/throughput qualification.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefill_interleave import prefill_two_rows
from consumer import decode_round


class RecordingNative:
    def __init__(self, eos_after=(3, 4), fail_call=None):
        self.kv = [[], []]
        self.emitted = [[], []]
        self.calls = []
        self.eos_after, self.fail_call = eos_after, fail_call

    def forward(self, row, past, values, prefill):
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise RuntimeError('Injected Native error; must not resume')
        assert past == len(self.kv[row])
        assert len(values) <= 512 and (prefill or len(values) == 1)
        self.kv[row].extend(values)
        self.calls.append((row, past, len(values), prefill))
        # Immutable per-call logits handle: a later row0 call cannot overwrite row1.
        return (row, tuple(self.kv[row])), 0., 0., 0.

    def feedback(self, row, logits):
        assert logits[0] == row and tuple(self.kv[row]) == logits[1]
        eos = len(self.emitted[row]) + 1 == self.eos_after[row]
        token = 10000 if eos else (sum(logits[1]) + row*19) % 1009
        self.emitted[row].append(token)
        return token, eos


def run(enabled, eos_after=(3, 4), budget=8, cancel_at=None, fail_call=None):
    native = RecordingNative(eos_after, fail_call)
    votes = []

    def vote():
        votes.append(len(native.calls))
        return len(votes) == cancel_at

    states, events = prefill_two_rows(
        [[5, 7, 11], [13]*2051], native.forward, native.feedback, vote,
        enabled=enabled, budget=budget)

    def forward(**kw):
        row, past, size = kw['descriptor']
        assert size == 1 and kw['positions'] == (past,)
        return native.forward(row, past, list(kw['input_ids']), False)

    while not all(s.done for s in states):
        decode_round(states, forward, native.feedback, enabled=True)
    return native, states, events, votes


def check():
    off, off_states, _, _ = run(False)
    on, states, events, votes = run(True)
    assert off.emitted == on.emitted and off.kv == on.kv
    assert [s.produced for s in states] == [3, 4]
    assert [s.past for s in states] == [5, 2054]
    assert [e['forwards'] for e in events if e['kind'] == 'complete_long_prefill_group'] == [4, 1]
    # No short forward inside the four-long-forward window.
    assert [(r, prefill) for r, _, _, prefill in on.calls[:8]] == [
        (0, True), (1, True), (1, True), (1, True), (1, True),
        (0, False), (1, True), (0, False)]
    assert votes[:5] == [0, 1, 1, 5, 7]
    eos_native, eos_states, eos_events, _ = run(True, eos_after=(1, 4))
    assert eos_states[0].done and len(eos_native.emitted[0]) == 1
    assert not any(e['kind'].startswith('short_decode') for e in eos_events)
    capped, capped_states, _, _ = run(True, eos_after=(50, 50), budget=2)
    assert [s.produced for s in capped_states] == [2, 2]
    for options, error in [({'cancel_at': 4}, InterruptedError),
                           ({'fail_call': 3}, RuntimeError)]:
        try:
            run(True, **options)
        except error:
            pass
        else:
            raise AssertionError('Expected original stop, no retry')
    for prompts in [[[1], []], [[1], [-1]], [[1], [True]], [[1], [1]*32768]]:
        try:
            prefill_two_rows(prompts, None, None, None)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid metadata accepted')
    return dict(passed=True, GPU=0, actual_Native=False, actual_weights=False,
                actual_new_prefill_consumer=True, unchanged_W4_and_decode_consumers=True,
                complete_prompt_KV_and_output_match_recording_baseline=True,
                long_W4_group_boundary_only=True, EOS_retire=True,
                budget_context_cancel_Native_failure_checked=True,
                Model_gain='UNKNOWN', whole_owner_GPU_entry='NOT_RUN')


if __name__ == '__main__':
    print(json.dumps(check()))
