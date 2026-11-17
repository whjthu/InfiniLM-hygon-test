"""Self-contained exact function copy; original unpub parent is not a runtime dependency."""
import random

def exposure_indices(seed=20261118):
    rng=random.Random(seed);indices=[]
    while len(indices)<800:
        block=list(range(45));rng.shuffle(block);indices.extend(block)
    return indices[:800]
