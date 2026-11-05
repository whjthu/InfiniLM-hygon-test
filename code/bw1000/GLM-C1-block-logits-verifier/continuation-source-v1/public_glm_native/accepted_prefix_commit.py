"""Two-vote C1 prefix commit for the original GLM TP8 PP2 control path."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .owner_compare import CONTEXT, OWNERS


@dataclass(frozen=True)
class StagedPrefix:
    sequence: int
    past: int
    width: int
    accepted: int
    staged_input_sha256: str

    def vote_identity(self):
        if (any(type(value) is not int for value in (self.sequence, self.past, self.width, self.accepted))
                or not 0 <= self.sequence < 2**64 or self.past < 0 or self.width not in (4, 8)
                or self.past + self.width > CONTEXT or not 0 <= self.accepted <= self.width
                or not isinstance(self.staged_input_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.staged_input_sha256) is None):
            raise ValueError("invalid GLM staged prefix")
        return sha256(json.dumps(
            (self.sequence, self.past, self.width, self.accepted, self.staged_input_sha256),
            separators=(",", ":"),
        ).encode("ascii")).digest()


def agree_ready(reduce, identity, ready):
    # At TP8 PP2 both byte moments sum exactly in FP32. Comparing only
    # first moments could accept different values that cancel around a mean.
    expected = [*identity, *(value * value for value in identity)]
    votes = reduce([int(ready), *expected])
    return ready and votes == [OWNERS, *(OWNERS * value for value in expected)]


def commit_after_forward_drain(plan, *, reduce, can_publish, publish):
    """Run on every original owner, on its owner thread, after a drained forward.

    The Native callbacks must bind the same staged forward and its private
    device-bank state. A failed second vote invalidates continuation and
    requires the caller to enter the original fatal failure path.
    """
    identity = bytes(32)
    ready = False
    preflight_error = None
    try:
        if type(plan) is not StagedPrefix:
            raise TypeError('expected a frozen GLM staged prefix')
        identity = plan.vote_identity()
        ready = can_publish(plan) is True
    except Exception as error:
        preflight_error = error
    if not agree_ready(reduce, identity, ready):
        raise RuntimeError("GLM prefix preflight or 16-owner agreement failed") from preflight_error

    commit_error = None
    committed = False
    try:
        committed = publish(plan) is True
    except Exception as error:
        commit_error = error
    if reduce([int(committed)]) != [OWNERS] or not committed:
        raise RuntimeError("GLM prefix commit failed; no continuation is valid") from commit_error
    return plan.past + plan.accepted
