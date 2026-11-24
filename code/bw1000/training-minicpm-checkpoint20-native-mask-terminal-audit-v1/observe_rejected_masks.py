"""Compare the archived captured masks under their measured profiler counts."""

from hashlib import sha256
import json
from pathlib import Path
import sys


REQUEST = "3b71a062261547af8b997684603c9e6b"
STEP = "1103606.2412"
POST = "1103606.2413"
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = RESULTS / "training-minicpm-checkpoint20-native-mask-target-v1"
ORIGINAL = RESULTS / "training-minicpm-usgs-exposure25-resume20-target-v2"
AUDIT = RESULTS / "training-minicpm-native-mask-original2412-independent-audit-3b71a0622615.json"
AUDIT_SHA = "64937670fb6198522fa9a328bf73a568b1896a84b3dc3abba15257fec4a81aca"
RAW_PINS = "eb63c0e67886f8035f7733eb96b909dd3168c7bab810f81ab32622749198302c"
FILES = {
    "mask_compare.py": "6d370a8c0870f40d2c5164a01facaec3a785aa4a93f613e4272205846bae08a9",
    "mask_rows.py": "c1a7653985b8cb80438782b466e4c891ca080423b2d6417627bd0066e9303e47",
    "route_control.py": "16ad617be313454684d1879da800837ec223d081fefd264c2b77848ceb4de8ba",
    "read_completed.py": "564c9b92d57f8e5df328f50726289d20658ce8b680126cb4d2f336a7c8f75fd9",
    "original-observer/compare_first.py": "e11c8b7b0955be3ea4b3e35dc8d747cb7edfbf51f62a0416ffad765b406200dd",
    "original-observer/probe.py": "1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4",
}


def exact(path, limit, pin):
    if path.resolve(strict=True) != path:
        raise ValueError("aliased input")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or sha256(raw).hexdigest() != pin:
        raise ValueError("original evidence bytes differ")
    return raw


def compare_recorded_paths(records, history, original_comparator):
    """Reuse its non-profiler checks after recording the exact measured pairs."""
    normalized = {}
    for (rank, arm), record in records.items():
        count = 372 if arm in ("c1", "c2") else 186
        operators = record["operators"]
        if operators["native_dropout"] != {"aten::dropout": 186, "aten::native_dropout": count}:
            raise ValueError(f"measured profiler envelope differs for {arm} rank {rank}")
        # Only the profiler count requirement is relaxed in this post hoc view.
        # The raw input records remain untouched and the original gate stays failed.
        normalized[rank, arm] = dict(record, operators=dict(operators, native_dropout={
            "aten::dropout": 186, "aten::native_dropout": 186}))
    result = original_comparator(normalized, STEP, history)
    return dict(result, observed_capture_native_dropout_profiler_events=372,
                observed_control_native_dropout_profiler_events=186, original_preregistered_gate_passed=False,
                capture_and_control_GPU_kernel_execution_equivalent="UNKNOWN", post_hoc=True)


def main():
    original = json.loads(exact(AUDIT, 262144, AUDIT_SHA))
    rows = original["numbered_slurm"]
    if not (original["request"] == REQUEST and rows["gpu"][0] == STEP and rows["gpu"][2:4] == ["FAILED", "1:0"]
            and rows["post"][0] == POST and rows["post"][2:4] == ["COMPLETED", "0:0"]
            and original["clean_independent_post"] is True and original["diagnosis_complete"] is False
            and original["comparison"]["planned_forwards"] == 32 and original["comparison"]["verified_forwards"] == 16
            and original["native_metadata_set_SHA256"] == RAW_PINS):
        raise ValueError("original independent audit scope differs")
    directory = RUN / "target-worker"
    for file, pin in FILES.items():
        exact(directory / file, 32768, pin)
    sys.path[:0] = [str(directory), str(directory / "original-observer")]
    try:
        from compare_first import original_rows
        from mask_compare import comparison
        from read_completed import read_arms

        records, pins, rejected = read_arms(RUN, STEP)
        keys = sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if len(records) != 32 or len(pins) != 96 or rejected or keys != RAW_PINS:
            raise ValueError("archived record closure differs")
        result = compare_recorded_paths(records, original_rows(ORIGINAL, directory / "original1171.json"), comparison)
    finally:
        del sys.path[:2]
    result.update(request=REQUEST, independent_terminal_SHA256=AUDIT_SHA, original_metadata_set_SHA256=keys,
                  observed_profile_route_uncertainty_retained=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
