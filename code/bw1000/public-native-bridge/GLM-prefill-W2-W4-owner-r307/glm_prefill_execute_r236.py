"""Default-OFF W4 source candidate; no runtime or Model qualification.

Each owner still performs synchronous forwards of at most 512 tokens. No
collective is inserted within a committed prefill group; decode remains W1.
Cancellation is checked before and after each group. Native failures propagate
without a collective drain, reset or retry on the failed owner.
"""


def execute_plan(plan, past, *, prefill, window, forward, vote, sample, discard):
    if type(window) is not int or window not in (1, 2, 4):
        raise ValueError('window must be exactly 1, 2 or 4')
    if type(prefill) is not bool or not plan:
        raise ValueError('prefill must be boolean and plan nonempty')
    if type(past) is not int or past < 0:
        raise ValueError('past must be a nonnegative integer')
    expected = past
    for logical_past, values in plan:
        if type(logical_past) is not int or logical_past != expected or not 1 <= len(values) <= 512 or expected + len(values) > 32768:
            raise ValueError('invalid complete ordered input plan')
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
            for start in range(0, len(plan), window):
                end = min(start + window, len(plan))
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
        discard()
