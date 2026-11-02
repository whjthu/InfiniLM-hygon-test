"""Two fixed geometry fixtures, no result-selected routing or parameter grid."""
CLUSTERED = [3, 11, 7, 15, 1, 5, 9, 13, 4, 12, 2, 10, 6, 14, 0, 8]


def row_order(fixture):
    if fixture == 0:
        return list(range(16)) * 32
    if fixture == 1:
        return [row for row in CLUSTERED for _ in range(32)]
    raise ValueError('Only two frozen row orders')


def records():
    for fixture in range(2):
        for arm in (0, 1):
            yield dict(fixture=fixture, warm=True, block=-1, phase=arm, arm=arm)
        for block in range(8):
            for phase, arm in enumerate((0, 1, 1, 0)):
                yield dict(fixture=fixture, warm=False, block=block, phase=phase, arm=arm)


def descriptor(record, pointers, deadline):
    import time
    if not isinstance(deadline, (int, float)) or time.time() >= deadline:
        raise TimeoutError('Original residency deadline; no retry')
    if record not in list(records()):
        raise ValueError('Frozen complete-call plan')
    if len(pointers) != 9 or any(type(p) is not int or p <= 0 for p in pointers):
        raise ValueError('Nine actual guarded device pointers')
    return [2, 0, 512, 896, 3584, 192, 16, *pointers]
