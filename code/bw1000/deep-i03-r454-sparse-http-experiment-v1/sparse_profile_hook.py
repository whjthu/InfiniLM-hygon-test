"""Opt-in sparse R454 observer for three serial occurrences of the original GSM8K request."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

PROMPT_SHA256 = "62465754106f1ac717fb73261997c17c67b27c81aceb9ffe3bde723578b9ce46"
PROMPT_TOKENS = 1242
OUTPUT_BUDGET = 4096
CAPTURE = (8, 87)
ARMS = (False, True, False)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sequence_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def install(policy_class, observer, output):
    """Wrap the two owner methods before startup; all Native calls remain on that owner."""
    output = Path(output)
    require(output.is_dir() and not getattr(policy_class, "_i03_sparse_installed", False),
            "private output or single installation required")
    original_request = policy_class.request
    original_forward = policy_class.forward

    @contextmanager
    def request(self, req, record):
        index = self.index
        require(0 <= index < len(ARMS) and self._mode == 0 and not self.poisoned,
                "foreign request or unavailable fixed owner")
        prompt = list(req.prompt_token_ids)
        require(len(prompt) == PROMPT_TOKENS and req.max_new_tokens == OUTPUT_BUDGET
                and sequence_sha(prompt) == PROMPT_SHA256, "diagnostic exact input differs")
        state = {"index": index, "armed": ARMS[index], "req": req, "windows": []}
        require(getattr(self, "_i03_sparse_state", None) is None, "nested owner request")
        self._i03_sparse_state = state
        try:
            with original_request(self, req, record):
                yield self
        finally:
            del self._i03_sparse_state
            received = [row["decode_index"] for row in state["windows"]]
            record["i03_sparse_observation"] = {
                "armed": state["armed"], "windows": state["windows"],
                "missing_samples": [sample for sample in CAPTURE if sample not in received],
                "measurement_scope": "fixed owner policy wrapper and sparse existing Native events",
            }

    def forward(self, *args):
        state = getattr(self, "_i03_sparse_state", None)
        require(state is not None, "forward outside original owner request")
        req = state["req"]
        index = len(req.generated_token_ids)
        if index not in CAPTURE:
            return original_forward(self, *args)
        require(len(args[0]) == 1 and args[2] == [0] and args[3] == [PROMPT_TOKENS + index - 1],
                "sparse decode geometry differs")
        row = {"decode_index": index, "past": args[3][0],
               "generated_prefix_SHA256": sequence_sha(list(req.generated_token_ids))}
        state["windows"].append(row)
        start = time.monotonic()
        try:
            if not state["armed"]:
                return original_forward(self, *args)
            lib = self.control.lib
            check = self.control.check
            observer.bind_inner(lib)
            geometry = dict(tokens=1, requests=1, rows=[0], past=args[3], decode_call_1based=index)
            prefix = output / ("request-%d-decode-%d" % (state["index"], index))
            def actual():
                value.append(original_forward(self, *args))
                return value[0][0]
            value = []
            try:
                rc, _, _, profile = observer.inner_owner_forward_once(
                    lib, actual, check, 1, observer.native_expected_kinds(self.control.native.cfg),
                    geometry, prefix)
                require(rc == 0 and len(value) == 1 and profile["native_window_closed"]
                        and profile["owner_event_validation_passed"] and profile["inner_event_validation_passed"],
                        "sparse event evidence incomplete")
                row["profile"] = profile
                return value[0]
            except BaseException as error:
                row["error_type"] = type(error).__name__
                if hasattr(error, "profile"):
                    row["profile"] = error.profile
                recovery = {}
                for name in ("deep_owner_arm_r8900", "deep_attention_inner_arm_r9193"):
                    try:
                        recovery[name] = int(getattr(lib, name)(0))
                    except Exception as cleanup:
                        recovery[name] = type(cleanup).__name__
                row["recovery"] = recovery
                self.control.poisoned = True
                raise
        finally:
            row["host_wrapper_s_including_observer"] = time.monotonic() - start

    policy_class.request = request
    policy_class.forward = forward
    policy_class._i03_sparse_installed = True
