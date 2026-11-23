"""Recompute an original Dev24 report with bounded cross-runtime AUC roundoff."""

import hashlib
import json
import math
from pathlib import Path
import re
import sys


AUC_ABSOLUTE_TOLERANCE = 2e-15
AUC_FIELD = re.compile(r"development_AUC_at_(?:5|15|30)(?:_deg)?\Z")


def compare_aggregates(rebuilt, original, path=()):
    """Allow roundoff on AUC or AUC deltas after exact structure checks."""
    if type(rebuilt) is not type(original):
        raise ValueError(f"summary types differ at {path}")
    if isinstance(rebuilt, dict):
        if rebuilt.keys() != original.keys():
            raise ValueError(f"summary fields differ at {path}")
        return [difference for name in rebuilt
                for difference in compare_aggregates(rebuilt[name], original[name], (*path, name))]
    if isinstance(rebuilt, list):
        if len(rebuilt) != len(original):
            raise ValueError(f"summary lengths differ at {path}")
        return [difference for index, (left, right) in enumerate(zip(rebuilt, original, strict=True))
                for difference in compare_aggregates(left, right, (*path, str(index)))]
    if rebuilt == original:
        return []
    if (type(rebuilt) is float and path and AUC_FIELD.fullmatch(path[-1])
            and math.isfinite(rebuilt) and math.isfinite(original)):
        delta = abs(rebuilt - original)
        if delta <= AUC_ABSOLUTE_TOLERANCE:
            return [{"field": ".".join(path), "absolute_difference": delta}]
    raise ValueError(f"summary values differ at {path}")


def audit(reader, expected_reader_sha256, inputs, inputs_sha, heads_sha, progress):
    """Keep the original schedule, row replay and denominator validation."""
    raw = reader.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_reader_sha256:
        raise ValueError("original audit source differs")
    anchor = b'rebuilt != terminal["summary"]'
    if raw.count(anchor) != 1:
        raise ValueError("original audit comparison boundary differs")
    differences = []

    def mismatches(computed, recorded):
        differences.extend(compare_aggregates(computed, recorded))
        return False

    sys.path.insert(0, str(reader.parent.resolve(strict=True)))
    try:
        namespace = {"__name__": "vggt_dev24_original_audit", "mismatches": mismatches}
        patched = raw.replace(anchor, b'mismatches(rebuilt, terminal["summary"])')
        exec(compile(patched, str(reader), "exec"), namespace)
        result = namespace["audit"](inputs, inputs_sha, heads_sha, progress)
    finally:
        sys.path.pop(0)
    result["aggregate_roundoff"] = {
        "auc_absolute_tolerance": AUC_ABSOLUTE_TOLERANCE,
        "differences": differences,
        "maximum_absolute_difference": max((row["absolute_difference"] for row in differences), default=0.0),
    }
    return result


def main():
    reader, source_sha, inputs, input_sha, heads_sha, progress, output = sys.argv[1:]
    result = audit(Path(reader), source_sha, Path(inputs), input_sha, heads_sha, Path(progress))
    with Path(output).open("x", encoding="utf8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
