"""Compare the native eight-rank USGS25 continuation and checkpoint-20 recovery."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(root, resumed_root=None):
    evidence, differences = {}, []
    max_loss_gap = 0.0

    def record(arm, rank, name):
        path = Path(arm) / "train/treasury-witness" / f"rank-{rank:02d}" / name
        base = resumed_root if arm == "resumed" and resumed_root is not None else root
        with (base / path).open("rb") as stream:
            raw = stream.read(8*1024**2+1)
        require(0 < len(raw) <= 8*1024**2, f"unexpected receipt size: {path}")
        evidence[path.as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for rank in range(8):
        continuous = record("continuous", rank, "result.json")
        resumed = record("resumed", rank, "result.json")
        skip = record("resumed", rank, "native-data-skip.json")
        original_rng = record("continuous", rank, "checkpoint-20-rng.json")
        restored = record("resumed", rank, "restored-before-RNG.json")
        require(all(x["rank"] == rank and x["world_size"] == 8 and x["completed_global_step"] == 25
                    for x in (continuous, resumed)), f"wrong native rank topology or endpoint: {rank}")
        require(skip == dict(rank=rank, before=100, skipped=80, after=20), f"wrong actual native skip: {rank}")
        require(continuous["saved_checkpoints"] == [10, 20, 25] and resumed["saved_checkpoints"] == [25]
                and resumed["native_resume_window_complete"] is True, f"incomplete native steps: {rank}")
        require(len(continuous["inputs"]) == len(continuous["outputs"]) == 100
                and len(resumed["inputs"]) == len(resumed["outputs"]) == 20
                and len(resumed["RNG_loads"]) == 1, f"incomplete forward or RNG receipts: {rank}")
        require([x["global_step"] for x in continuous["steps"]] == list(range(1,26))
                and [x["global_step"] for x in resumed["steps"]] == list(range(21,26)), f"wrong update order: {rank}")
        require(original_rng["rank"] == rank and resumed["RNG_loads"][0]["saved_equal"] is True
                and resumed["RNG_loads"][0]["global_step"] == 20, f"restored checkpoint RNG not verified: {rank}")
        require(restored["restored_parameter_optimizer_scheduler_equal"] is True,
                f"checkpoint state restoration not verified: {rank}")
        if continuous["steps"][19]["state"] != restored["actual"]:
            differences.append(dict(rank=rank, field="checkpoint_20_state"))
        if original_rng["RNG"] != resumed["RNG_loads"][0]["actual"]:
            differences.append(dict(rank=rank, field="checkpoint_20_rng"))
        final_continuous=record("continuous",rank,"checkpoint-25-rng.json")
        final_resumed=record("resumed",rank,"checkpoint-25-rng.json")
        if final_continuous!=final_resumed:
            differences.append(dict(rank=rank,field="checkpoint_25_rng"))
        for index, (left, right) in enumerate(zip(continuous["inputs"][80:], resumed["inputs"]), 1):
            step = 20 + (index - 1) // 4
            require(left["ordinal"] == index + 80 and right["ordinal"] == index
                    and left["global_step"] == right["global_step"] == step, f"wrong forward cursor: {rank}")
            if left["full_input"] != right["full_input"]:
                differences.append(dict(rank=rank, field="input", continuation_ordinal=index))
        for index, (left, right) in enumerate(zip(continuous["outputs"][80:], resumed["outputs"]), 1):
            require(left["ordinal"] == index + 80 and right["ordinal"] == index, f"wrong loss cursor: {rank}")
            gap = abs(left["loss_scalar"] - right["loss_scalar"])
            require(math.isfinite(gap), f"nonfinite loss gap: {rank}")
            max_loss_gap = max(max_loss_gap, gap)
            if left["loss"] != right["loss"] or gap:
                differences.append(dict(rank=rank, field="loss", continuation_ordinal=index))
        for left, right in zip(continuous["steps"][20:], resumed["steps"]):
            if left["state"] != right["state"]:
                differences.append(dict(rank=rank, field="updated_state", global_step=right["global_step"]))
    return dict(ranks=8, paired_remaining_microbatches=160, paired_updates=5,
                planned_optimizer_invocations=30,unique_source_examples=88,physical_exposures_continuous=800, differences=differences,
                maximum_absolute_loss_gap=max_loss_gap, exact_paired_recovery=not differences,
                evidence_SHA256=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--resumed-root", type=Path)
    opts = parser.parse_args()
    record = compare(opts.root, opts.resumed_root)
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if record["exact_paired_recovery"] else 1)


if __name__ == "__main__":
    main()
