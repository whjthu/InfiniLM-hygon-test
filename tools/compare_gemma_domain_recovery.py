"""Compare original continuous and fresh-process Gemma checkpoint20 receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path


RANKS = 8
SKIP = 80
TAIL = 20
LAST_STEP = 25
MAX_BYTES = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_receipt(root, relative, evidence):
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "receipt escapes its run directory")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(0 < len(raw) <= MAX_BYTES, "invalid receipt length")
    evidence[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def state(record):
    require(set(record) == {"adapter", "optimizer", "scheduler"} and all(record.values()),
            "missing adapter, optimizer or scheduler fingerprint")
    return record


def verify_arm(result, rank, offset, start, checkpoint_steps):
    total = 100 - offset
    require((result["rank"], result["world_size"], result["completed_global_step"])
            == (rank, RANKS, LAST_STEP) and result["saved_checkpoints"] == checkpoint_steps,
            "rank, endpoint or checkpoint schedule differs")
    require(len(result["inputs"]) == len(result["outputs"]) == total, "incomplete forward records")
    require([row["global_step"] for row in result["steps"]] == list(range(start + 1, LAST_STEP + 1)),
            "incomplete optimizer step records")
    for i, (observed, output) in enumerate(zip(result["inputs"], result["outputs"]), 1):
        require(observed["ordinal"] == output["ordinal"] == i and observed["global_step"] == start + (i - 1) // 4,
                "unexpected microbatch or accumulation boundary")
        require(observed["full_input"] and output["loss"] and type(output["loss_scalar"]) in (int, float)
                and math.isfinite(output["loss_scalar"]), "invalid forward fingerprint or loss")


def compare_rank(rank, root, evidence, used_processes):
    def read(arm, name):
        kind = "continuous-witness" if arm == "continuous" else "resume-witness"
        return read_receipt(root, Path(arm) / "train" / kind / f"rank-{rank:02d}" / name, evidence)

    continuous, resumed = (read(arm, "result.json") for arm in ("continuous", "resumed"))
    owners = [read(arm, "owner.json") for arm in ("continuous", "resumed")]
    keys = {(owner["PID"], owner["startticks"]) for owner in owners}
    require(all(owner["rank"] == owner["local_rank"] == rank
                and type(owner["PID"]) is int and owner["PID"] > 0
                and type(owner["startticks"]) is int and owner["startticks"] > 0 for owner in owners)
            and len(keys) == 2 and not keys.intersection(used_processes),
            "paired evidence does not contain independent rank processes")
    used_processes.update(keys)
    verify_arm(continuous, rank, 0, 0, [10, 20, 25])
    verify_arm(resumed, rank, SKIP, 20, [25])
    require(resumed["native_resume_window_complete"] is True and
            read("resumed", "native-data-skip.json") == dict(rank=rank, before=100, skipped=SKIP, after=TAIL),
            "native data skip was missing or incorrect")
    require(len(resumed["RNG_loads"]) == 1, "expected one native RNG restoration")
    restored = resumed["RNG_loads"][0]
    original_rng = read("continuous", "checkpoint-20-rng.json")
    original_final, resumed_final = (read(arm, "checkpoint-25-rng.json") for arm in ("continuous", "resumed"))
    require(restored["saved_equal"] is True and restored["global_step"] == 20 and restored["actual"]
            and all(rng["rank"] == rank and rng["RNG"] for rng in (original_rng, original_final, resumed_final)),
            "checkpoint RNG record is incomplete")
    checkpoint = read("resumed", "restored-before-RNG.json")
    require(checkpoint["restored_parameter_optimizer_scheduler_equal"] is True,
            "native checkpoint state restoration was not verified")
    differences = []

    def compare(field, left, right, **location):
        if left != right:
            differences.append(dict(rank=rank, field=field, **location))

    compare("checkpoint_20_state", state(continuous["steps"][19]["state"]), state(checkpoint["actual"]))
    compare("checkpoint_20_rng", original_rng["RNG"], restored["actual"])
    compare("checkpoint_25_rng", original_final["RNG"], resumed_final["RNG"])
    gaps = []
    for i, (left, right) in enumerate(zip(continuous["inputs"][SKIP:], resumed["inputs"]), 1):
        compare("input", left["full_input"], right["full_input"], tail_ordinal=i)
    for i, (left, right) in enumerate(zip(continuous["outputs"][SKIP:], resumed["outputs"]), 1):
        gap = abs(left["loss_scalar"] - right["loss_scalar"])
        require(math.isfinite(gap), "invalid paired loss gap")
        gaps.append(gap)
        compare("loss", (left["loss"], left["loss_scalar"]), (right["loss"], right["loss_scalar"]), tail_ordinal=i)
    for left, right in zip(continuous["steps"][20:], resumed["steps"]):
        compare("updated_state", state(left["state"]), state(right["state"]), global_step=left["global_step"])
    return differences, max(gaps)


def compare(root):
    root = root.resolve()
    evidence, differences, completed, gaps = {}, [], [], []
    processes = set()
    for rank in range(RANKS):
        try:
            changes, gap = compare_rank(rank, root, evidence, processes)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            differences.append(dict(rank=rank, field="missing_or_invalid_rank_evidence", reason=type(error).__name__))
        else:
            completed.append(rank)
            gaps.append(gap)
            differences.extend(changes)
    return dict(expected_ranks=RANKS, compared_ranks=completed,
                planned_tail_microbatch_pairs=RANKS * TAIL, compared_tail_microbatch_pairs=len(completed) * TAIL,
                planned_rank_update_pairs=RANKS * 5, compared_rank_update_pairs=len(completed) * 5,
                differences=differences, maximum_absolute_loss_gap=max(gaps, default=None),
                exact_paired_recovery=len(completed) == RANKS and not differences, evidence_SHA256=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    result = compare(parser.parse_args().root)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["exact_paired_recovery"] else 1)


if __name__ == "__main__":
    main()
