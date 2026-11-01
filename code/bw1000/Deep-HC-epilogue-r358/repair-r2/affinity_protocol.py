"""Keep the full allowed CPU pool until the actual Runtime submission chooses a subset."""
import os

def rank_plan(rank):
    assert type(rank) is int and 0 <= rank < 8
    initial = sorted(os.sched_getaffinity(0))
    assert len(initial) >= 32, initial
    return initial, initial[rank * 4:rank * 4 + 4]
