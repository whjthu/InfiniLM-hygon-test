"""Lower scheduling priority only; injected stdlib callables allow CPU-only boundary checks."""
def apply_nice(get_priority,increment_nice):
    inherited=get_priority()
    if type(inherited) is not int or not -20<=inherited<=19:
        raise ValueError('INHERITED_NICE_OUT_OF_RANGE')
    requested=max(inherited,10)
    increment=requested-inherited
    increment_nice(increment)  # Always nonnegative: never requests a priority increase.
    actual=get_priority()
    if type(actual) is not int or not requested<=actual<=19:
        raise ValueError('ACTUAL_NICE_NOT_LOW_ENOUGH')
    return dict(inherited_nice=inherited,requested_nice=requested,nice_increment=increment,actual_nice=actual,
        scheduling_priority_increase_requested=False)
