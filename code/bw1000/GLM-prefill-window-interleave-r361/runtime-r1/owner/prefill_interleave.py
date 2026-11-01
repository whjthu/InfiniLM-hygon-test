"""CPU-checked next candidate, GPU-unqualified; unchanged C2 Native row ABI.

Complete short prefill first. At complete long W4 group boundaries only, allow
one short-row decode/feedback. No WORLD agreement or decode inside a W4 group.
The caller retains Native ownership, fail-stop cleanup, and original deadline.
"""
from consumer import Row
from glm_prefill_execute_r236 import execute_plan


def prefill_two_rows(prompts, forward, feedback, cancel_vote, *, enabled=False,
                     budget=4096, capacity=32768, check=lambda: None):
    if type(enabled) is not bool or len(prompts) != 2:
        raise ValueError('Two fixed rows and an explicit boolean policy required')
    if type(budget) is not int or budget < 1 or type(capacity) is not int or capacity != 32768:
        raise ValueError('Bounded original context/output policy required')
    for prompt in prompts:
        if not 0 < len(prompt) < capacity or any(type(x) is not int or x < 0 for x in prompt):
            raise ValueError('Complete unmodified prompt IDs required')
    states, events = [], []

    def advance_short():
        state = states[0]
        if state.done:
            return
        check()
        if state.past >= capacity or state.produced >= state.budget:
            raise ValueError('No remaining short-row context/output capacity')
        # Exactly the existing single-row decode CAPI shape; no cache reset.
        logits, *_ = forward(0, state.past, [state.token], False)
        token, eos = feedback(0, logits)
        if type(token) is not int or token < 0 or type(eos) is not bool:
            raise ValueError('Invalid actual short-row feedback')
        state.past += 1
        state.token, state.produced = token, state.produced + 1
        state.done = eos or state.produced == state.budget
        events.append(dict(kind='short_decode_at_complete_long_W4_boundary',
                           short_past=state.past, short_produced=state.produced))

    for row, prompt in enumerate(prompts):
        retained = [None]
        successful_since_vote = [0]
        last_feedback = [None]

        def pf_forward(past, values, final):
            check()
            logits, *_ = forward(row, past, values, True)
            successful_since_vote[0] += 1
            if final:
                retained[0] = logits

        def pf_sample():
            if retained[0] is None:
                raise ValueError('Missing owned final prompt logits')
            token, eos = feedback(row, retained[0])
            if type(token) is not int or token < 0 or type(eos) is not bool:
                raise ValueError('Invalid actual prompt feedback')
            last_feedback[0] = (token, eos)
            retained[0] = None
            return token

        def boundary_vote():
            # Original cancellation agreement remains first, never within F0..F3.
            if cancel_vote():
                return True
            if enabled and row == 1 and successful_since_vote[0]:
                events.append(dict(kind='complete_long_prefill_group',
                                   forwards=successful_since_vote[0]))
                advance_short()
            successful_since_vote[0] = 0
            return False

        plan = [(past, prompt[past:past+512]) for past in range(0, len(prompt), 512)]
        past, token, canceled = execute_plan(
            plan, 0, prefill=True, window=4, forward=pf_forward,
            vote=boundary_vote, sample=pf_sample,
            discard=lambda: retained.__setitem__(0, None))
        if canceled:
            raise InterruptedError('Original prefill cancellation; no ledger replay')
        if past != len(prompt) or last_feedback[0] is None or last_feedback[0][0] != token:
            raise ValueError('Incomplete actual prompt state')
        states.append(Row(row, past, token, budget, 1, last_feedback[0][1] or budget == 1))
    return states, events
