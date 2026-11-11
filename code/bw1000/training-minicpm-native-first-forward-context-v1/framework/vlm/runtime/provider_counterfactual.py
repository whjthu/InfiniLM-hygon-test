"""Bounded instance-only gated-delta provider substitution for numerical diagnostics."""
from contextlib import contextmanager
import functools


@contextmanager
def selected_callables(bindings, replacement=None):
    """Replace one attribute on known instances and restore it on every exit.

    The caller validates model identities and provider sources. This helper never
    edits class attributes or module globals. Counts describe calls, not kernels.
    """
    rows = list(bindings)
    if not rows or len({id(owner) for _, owner in rows}) != len(rows):
        raise ValueError('empty or duplicate provider owners')
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError('duplicate provider names')
    if replacement is not None and not callable(replacement):
        raise ValueError('replacement must be callable')
    originals = []
    for name, owner in rows:
        current = owner.chunk_gated_delta_rule
        if not callable(current) or getattr(current, '_provider_arm_wrapper', False):
            raise ValueError('invalid or nested provider arm')
        originals.append((name, owner, current))
    records = []
    installed = []
    try:
        for name, owner, original in originals:
            target = original if replacement is None else replacement
            row = dict(name=name, calls=0, first_call=None)
            records.append(row)
            @functools.wraps(target)
            def invoke(*args, _target=target, _row=row, **kwargs):
                _row['calls'] += 1
                if _row['first_call'] is None:
                    values = dict(zip(('q', 'k', 'v', 'g', 'beta'), args))
                    values.update(kwargs)
                    _row['first_call'] = {
                        key: dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
                        for key, value in values.items() if hasattr(value, 'shape') and hasattr(value, 'stride')
                    }
                return _target(*args, **kwargs)
            invoke._provider_arm_wrapper = True
            owner.chunk_gated_delta_rule = invoke
            installed.append((owner, original, invoke))
        yield records
    finally:
        changed = []
        for owner, original, invoke in reversed(installed):
            if owner.chunk_gated_delta_rule is not invoke:
                changed.append(type(owner).__name__)
            owner.chunk_gated_delta_rule = original
        if changed:
            raise RuntimeError('provider arm was changed externally; originals restored')
