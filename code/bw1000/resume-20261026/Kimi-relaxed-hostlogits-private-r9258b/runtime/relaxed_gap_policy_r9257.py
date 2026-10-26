"""Experimental target-logit-gap acceptance; no lossless-distribution claim.
This module reuses the original accepted-prefix/commit contract. It does not
change the model, native state machine, sampling logits, or RPC implementation.
"""
import math
from batch_policy_r9128 import greedy, finish

def decide(anchor, proposals, target_top1, gaps, epsilon, eos=(), special=(), remaining=None):
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative')
    proposals = tuple(proposals)
    target_top1 = tuple(target_top1)
    if len(target_top1) != len(proposals) + 1:
        raise ValueError('expected K+1 conditional target rows')
    if epsilon == 0:
        return greedy(anchor, proposals, target_top1, eos, remaining), {'enabled': False, 'relaxed': 0, 'fallback': 0}
    if gaps is None or len(gaps) != len(proposals):
        raise ValueError('one conditional target logit gap per proposal required')
    protected = set(eos) | set(special)
    accepted = relaxed = fallback = 0
    relaxed_positions = []
    for i, proposal in enumerate(proposals):
        top1 = target_top1[i]
        if proposal == top1:
            accepted += 1
            continue
        gap = gaps[i]
        valid = not isinstance(gap, bool) and isinstance(gap, (int, float)) and math.isfinite(gap) and gap >= 0
        if not valid:
            fallback += 1
            break
        if proposal in protected or top1 in protected or gap > epsilon:
            break
        accepted += 1
        relaxed += 1
        relaxed_positions.append(i)
    decision = finish(anchor, proposals, accepted, target_top1[accepted], eos, remaining)
    emitted_relaxed = [i for i in relaxed_positions if i < min(accepted, len(decision.emitted))]
    return decision, {'enabled': True, 'relaxed': len(emitted_relaxed), 'relaxed_positions': emitted_relaxed, 'verified_relaxed': relaxed, 'verified_relaxed_positions': relaxed_positions, 'fallback': fallback}
