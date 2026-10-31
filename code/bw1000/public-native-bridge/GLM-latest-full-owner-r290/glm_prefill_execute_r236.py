"""Local source-only plan executor. No Torch/Native/imported service lifecycle.
Callbacks keep each owner synchronous. No collective inside a committed W2
prefill group. Runtime/CCL liveness and Model quality require actual validation.
"""

def execute_plan(plan, past, *, prefill, window, forward, vote, sample, discard):
    if type(window) is not int or window not in (1, 2):
        raise ValueError("window must be exactly 1 or 2")
    if type(prefill) is not bool or not plan:
        raise ValueError("prefill must be boolean and plan nonempty")
    if type(past) is not int or past < 0:
        raise ValueError("past must be a nonnegative integer")
    expected = past
    for logical_past, values in plan:
        if type(logical_past) is not int or logical_past != expected or not 1 <= len(values) <= 512 or expected + len(values) > 32768:
            raise ValueError("invalid complete ordered input plan")
        expected += len(values)
    token = None
    try:
        if window == 1 or not prefill:
            for index, (logical_past, values) in enumerate(plan):
                if vote():
                    return past, None, True
                final = index == len(plan) - 1
                forward(logical_past, values, final)
                past += len(values)
                if final:
                    token = sample()
        else:
            if vote():
                return past, None, True
            for start in range(0, len(plan), 2):
                end = min(start + 2, len(plan))
                for index in range(start, end):
                    logical_past, values = plan[index]
                    final = index == len(plan) - 1
                    forward(logical_past, values, final)
                    past += len(values)
                if vote():
                    return past, None, True
                if final:
                    token = sample()
        return past, token, False
    finally:
        # Drop only local Python references. This is never a Native drain/reset
        # or collective, including when a forward or boundary callback raises.
        discard()
