"""Read the one original development result and independently recompute its denominator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PINS = {
    "outcome.json": "7fb7e7fd235d20d279fb371f7d857c3773f0d54582139ca86c76477daeae6ea9",
    "native-result.json": "b91cc581efebc03600a82c7628b0e09c4adb93030c9e06d341d0820fa9b87257",
    "training-authority.json": "d76f3733a2f9a95cb7b243299fb23b056832ab1fcb32b3e61dd5ab1f9c693572",
}


def read(root, name):
    path = root / name
    if path.is_symlink() or path.stat().st_size > 128 * 1024:
        raise ValueError("Unexpected original development artifact")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if name in PINS and digest != PINS[name]:
        raise ValueError("Original evidence changed")
    return json.loads(raw), digest


def audit(root):
    outcome = read(root, "outcome.json")[0]
    native = read(root, "native-result.json")[0]
    authority = read(root, "training-authority.json")[0]
    if (outcome["request"] != "89742fc58fc644499770300a619cf7ec"
            or outcome["publication"] != "16466a368366bf9450fa89690224258333d6064b"
            or outcome["result_SHA256"] != PINS["native-result.json"]):
        raise ValueError("Original lineage differs")
    stages = {}
    for phase, number in (("pre", 323), ("gpu", 324), ("post", 325)):
        accounting = read(root, phase + "-accounting.json")[0]
        row = accounting["row"]
        if (row[0] != "1087656." + str(number) or row[2:4] != ["COMPLETED", "0:0"]
                or row[5] != "f11r4n19" or accounting != outcome["stages"][phase]
                or not accounting["terminal_visible"] or accounting["rc"] != 0):
            raise ValueError("Actual terminal accounting failed")
        stages[phase] = row
    before, before_sha = read(root, "before.json")
    after, after_sha = read(root, "after.json")
    if (any(before[k] != after[k] for k in ("node", "pci", "boot_id"))
            or not after["full_fd_inventory_performed"] or after["kfd_holder_count"] != 0
            or before["monotonic_ns"] >= after["monotonic_ns"]):
        raise ValueError("Independent cleanup failed")
    if (native["training_authority"] != authority or authority["original_recovery_passed"] is not True
            or authority["original_model_training_updates"] != 25
            or authority["trained_head_file_SHA256"] != "522483d98b126de0dcf00825cb32751d8bba7ca98a612d9d098a066b97c951e9"
            or native["actual_DCU"] != 1 or native["optimizer_updates"] != 0
            or native["formal_CO3D_eligible"] is not False or native["formal_performance_eligible"] is not False):
        raise ValueError("Development scope differs")
    rows = native["paired_pose_rows"]
    manifest, manifest_sha = read(root, "data/pair.json")
    if manifest_sha != "6a0490db52ea7120766b3741b9dc788533abb81b8d2006ec7e8a2535fbd49a88":
        raise ValueError("Frozen development inventory changed")
    if len(rows) != 3 or len({(r["sequence"], tuple(r["frame_ids"])) for r in rows}) != 3:
        raise ValueError("Missing or duplicate development pair")
    if {r["pair_sha256"] for r in rows} != {p["manifest_sha256"] for p in manifest["pairs"]}:
        raise ValueError("Development results differ from the planned pair set")
    summary = dict(pairs=3, sequences=len({r["sequence"] for r in rows}))
    for arm in ("baseline", "updated"):
        errors = [r[arm]["pair_error_deg"] for r in rows]
        if any(e is not None and (not math.isfinite(e) or not 0 <= e <= 180) for e in errors):
            raise ValueError("Invalid angular error")
        summary[arm] = dict(failed_pairs=errors.count(None), **{
            "development_AUC_at_" + str(t): sum(max(0, 1 - e / t) for e in errors if e is not None) / 3
            for t in (5, 15, 30)})
    if summary != native["summary"]:
        raise ValueError("Independent AUC calculation differs")
    return dict(independent_root_audit_passed=True, accounting=stages, original_pins=PINS,
                before_SHA256=before_sha, after_SHA256=after_sha, summary=summary,
                development_AUC30_delta=summary["updated"]["development_AUC_at_30"] - summary["baseline"]["development_AUC_at_30"],
                trained_head_promoted=False, domain_effect_qualified=False, formal_CO3D_eligible=False,
                formal_performance_eligible=False, household_blind_accessed=False,
                qualification="Three development pairs in two scenes; no independent generalization, formal CO3D or performance claim. Reported angular metrics were recomputed; this reader does not replay model inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True, allow_nan=False))
