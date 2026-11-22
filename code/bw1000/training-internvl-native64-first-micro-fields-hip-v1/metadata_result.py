"""Validate a deliberate stop using the eight native first-microbatch receipts."""

from hashlib import sha256
import re

from plugin.fields import REQUEST, plain_json, read_small, validated_fields


def inspect_stop(field_root, log_path, fingerprint_roster, *, returncode, controller_error, cleanup):
    ranks = []
    for rank in range(8):
        path = field_root / f"rank-{rank:02d}.json"
        if not path.exists():
            ranks.append({"rank": rank, "state": "MISSING"})
            continue
        try:
            raw = read_small(path, cap=65536)
            doc = plain_json(raw)
            validated_fields(doc, "HIP", rank, fingerprint_roster[str(rank)][0])
        except (OSError, ValueError, KeyError, TypeError):
            ranks.append({"rank": rank, "state": "INVALID"})
            continue
        ranks.append({"rank": rank, "state": "VERIFIED", "receipt_SHA256": sha256(raw).hexdigest(),
                      "match": doc["input_matches_original"], "field_count": len(doc["fields"])})

    valid = [row for row in ranks if row["state"] == "VERIFIED"]
    matches = sum(row["match"] for row in valid)
    expected = "MATCHED" if matches == 8 else "DIFFERENT"
    marker = re.compile(
        rb"^(?:\[rank[0-7]\]:\s*)?(?:hip_hook\.)?FirstMicroCaptured: "
        + REQUEST.encode("ascii") + rb":(MATCHED|DIFFERENT)_INPUT_DIAGNOSTIC_STOP\s*$", re.M
    )
    boundaries = set()
    log_sha = None
    try:
        raw = read_small(log_path, cap=8 << 20)
        log_sha = sha256(raw).hexdigest()
        boundaries = {match.decode("ascii") for match in marker.findall(raw)}
    except (OSError, ValueError):
        pass
    complete = (
        len(valid) == 8 and boundaries == {expected}
        and type(returncode) is int and returncode > 0 and controller_error is None
        and cleanup.get("owned_processes_remaining") == 0 and not cleanup.get("termination_needed")
    )
    return {
        "request": REQUEST, "native_returncode": returncode, "controller_error": controller_error,
        "planned_ranks": 8, "verified_ranks": len(valid), "matched_ranks": matches,
        "rank_receipts": ranks, "native_log_SHA256": log_sha, "detected_boundaries": sorted(boundaries),
        "diagnostic_complete": complete, "qualified_optimizer_updates": 0,
        "formal17_qualified": False, "numerical_A100_alignment_qualified": False,
    }
