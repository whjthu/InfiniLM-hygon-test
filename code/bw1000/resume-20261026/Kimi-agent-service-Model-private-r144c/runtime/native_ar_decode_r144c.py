"""Original prefill + width-one verification/commit, no speculative draft."""
import time
from batch_policy_r9128 import greedy


def generate(native, prompt, cap, eos, emit, cancelled, deadline, clock=time.monotonic):
    assert 1 <= len(prompt) <= 512 and 1 <= cap <= 512 and len(prompt) + cap <= 1024
    generated = []
    selected, _, _ = native.call("prefill", [1], [0], [list(prompt)], False)
    token = int(selected[0]);generated.append(token);emit(token)
    past = len(prompt)
    while token not in eos and len(generated) < cap:
        if cancelled(len(generated)):return generated, "canceled"
        if clock() >= deadline:return generated, "timeout"
        selected, _, _ = native.call("verify", [1], [past], [[token]], False)
        decision = greedy(token, [], selected, eos, cap - len(generated))
        native.commit([1], [decision])
        token = int(decision.emitted[0]);generated.append(token);emit(token)
        if not decision.terminal:past += len(decision.commit_inputs)
    return generated, "eos" if token in eos else "length"
