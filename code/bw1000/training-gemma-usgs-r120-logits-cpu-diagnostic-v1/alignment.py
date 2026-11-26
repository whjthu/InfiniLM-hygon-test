"""Strict causal-label alignment reference for one CPU-attested native selector."""

from hashlib import sha256
import json


IGNORE = -100


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(values):
    return sha256(json.dumps(values,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def positions(selection, width):
    if type(selection) is int:
        require(0 <= selection <= width, "BAD_NATIVE_SUFFIX")
        return list(range(width-selection,width)) if selection else list(range(width))
    require(type(selection) is list and bool(selection), "UNOBSERVED_NATIVE_SELECTOR")
    if all(type(x) is bool for x in selection):
        require(len(selection)==width, "BAD_NATIVE_MASK_EXTENT")
        return [i for i,x in enumerate(selection) if x]
    require(all(type(x) is int and 0<=x<width for x in selection), "BAD_NATIVE_INDEX")
    require(selection==sorted(set(selection)), "DUPLICATE_OR_UNORDERED_NATIVE_INDEX")
    return selection


def alignment(full, passed, indices, *, mode):
    """Compare with full next-token targets, including every supervised anchor."""
    n=len(full)
    require(mode in {"native_mask","native_suffix","no_compression"} and 1<n<=512 and len(passed)==len(indices)>0
            and all(type(x) is int and 0<=x<n for x in indices)
            and indices==sorted(set(indices)),"FULL_AND_PROJECTED_LABEL_EXTENT")
    shifted=full[1:]+[IGNORE]
    live=[i for i,value in enumerate(shifted) if value!=IGNORE]
    require(bool(live) and [i for i in indices if shifted[i]!=IGNORE]==live,"SUPERVISED_CAUSAL_POSITION_LOSS")
    independent=[shifted[i] for i in indices]
    if mode == "native_mask":
        require(indices==live+[n-1] and passed[0]==IGNORE, "MASK_ANCHORS_OR_LEADING_IGNORE")
    else:
        require(indices==list(range(indices[0],n)),"NONCONTIGUOUS_LOCALLY_SHIFTED_SELECTOR")
        require(passed==[full[i] for i in indices],"PROJECTED_UNSHIFTED_LABEL_IDENTITY")
    effective=passed[1:]+[IGNORE]
    require(effective==independent,"PROJECTED_CAUSAL_LABEL_IDENTITY")
    return dict(full_length=n,effective_length=len(indices),supervised=len(live),
                selector_positions_SHA256=digest(indices),reference_targets_SHA256=digest(independent),
                effective_targets_SHA256=digest(effective),alignment_passed=True)


def shifted_one_is_rejected(full, passed, indices, *, mode):
    # Shift the supervised anchors while preserving the terminal ignore anchor for the mask path.
    for direction in (-1,1):
        wrong=([p+direction for p in indices[:-1]]+[indices[-1]]) if mode=="native_mask" else [p+direction for p in indices]
        try:
            alignment(full,passed,wrong,mode=mode)
        except ValueError:
            continue
        raise ValueError("OFF_BY_ONE_CONTROL_PASSED")
    return True
