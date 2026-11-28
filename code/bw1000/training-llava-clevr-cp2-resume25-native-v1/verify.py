"""Independently reconcile the continuation receipts and native checkpoint files."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import stat

from contract import checked, cursor, encode, need


def read(path):
    need(path.resolve(strict=True) == path and 0 < path.stat().st_size <= 768 << 10,
         "BOUNDED_CANONICAL_NATIVE_RECEIPT")
    return json.loads(path.read_bytes())


def inspect_checkpoint(path, step):
    need(path.resolve(strict=True) == path and len(list(path.iterdir())) == 17, "NATIVE_17_MEMBER_CHECKPOINT")
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "additional_config.json", "args.json", "README.md", "training_args.bin",
                *(f"rng_state_{rank}.pth" for rank in range(8))}
    need({file.name for file in path.iterdir()} == required, "EXACT_NATIVE_CHECKPOINT_MEMBER_NAMES")
    state = read(path / "trainer_state.json")
    need((state["global_step"], state["epoch"], state["max_steps"]) == (step, step / 2, 25),
         "ORIGINAL_COSINE_CADENCE_AND_NATIVE_CHECKPOINT_CURSOR")
    result = {}
    for name in sorted(required):
        target = path / name
        record = target.lstat()
        need(stat.S_ISREG(record.st_mode) and target.resolve(strict=True) == target
             and 0 < record.st_size < 384 << 20, "BOUNDED_REGULAR_NEW_CHECKPOINT_MEMBER")
        fingerprint = sha256()
        with target.open("rb") as stream:
            while chunk := stream.read(4 << 20):
                fingerprint.update(chunk)
        after = target.lstat()
        keys = ("st_dev", "st_ino", "st_size", "st_ctime_ns", "st_mtime_ns")
        need(all(getattr(record, key) == getattr(after, key) for key in keys), "UNCHANGED_NEW_CHECKPOINT_AFTER_READBACK")
        result[name] = dict(bytes=record.st_size, SHA256=fingerprint.hexdigest())
    return result


def scan(root, request, plan):
    outcome = dict(request=request, planned_forwards=736, planned_updates=23, witnessed_forwards=0,
                   rank_optimizer_returns=0, uninterrupted_equivalence=None, passed=False)
    try:
        need(plan["start"] == cursor(dict(global_step=2, epoch=1.0, max_steps=25))
             and [entry["global_step"] for entry in plan["updates"]] == list(range(3, 26)),
             "EXACT_REAL_CPU_PLAN_23_STEP_DENOMINATOR")
        for row in plan["updates"]:
            updates = []
            step = row["global_step"]
            need([(slot["accumulation"], slot["rank"]) for slot in row["rows"]] ==
                 [(micro, rank) for micro in range(1, 5) for rank in range(8)], "EXACT_STEP_32_CPU_PLAN_BOUNDARIES")
            for slot in row["rows"]:
                rank = slot["rank"]
                rank_dir = root / f"rank-{rank:02d}"
                ordinal = (step - 3) * 4 + slot["accumulation"]
                actual = read(rank_dir / f"forward-{ordinal:02d}.json")
                need(actual == dict(request=request, global_step=step, **slot), "EXACT_REAL_FORWARD_CPU_PLAN")
                outcome["witnessed_forwards"] += 1
            for rank in range(8):
                rank_dir = root / f"rank-{rank:02d}"
                record = read(rank_dir / f"update-{step:02d}-returned.json")
                need(record["request"] == request and record["rank"] == rank
                     and (record["finite_losses"], record["adamw_tensors"], record["pre_adamw_step"],
                          record["post_adamw_step"]) == (4, 504, step - 1, step)
                     and record["learning_rates"] and all(math.isfinite(x) and x > 0 for x in record["learning_rates"]),
                     "ACTUAL_FINITE_POSITIVE_NATIVE_ADAMW_RETURN")
                updates.append({key: value for key, value in record.items() if key != "rank"})
                outcome["rank_optimizer_returns"] += 1
            need(updates == [updates[0]] * 8, "EXACT_STEP_EIGHT_RANK_AGREEMENT")
            if step == 3:
                first = updates[0]
                need(first["pre_SHA256"] != first["post_SHA256"] and first["parameter_delta_l2"] > 0,
                     "ACTUAL_STEP3_PARAMETER_CHANGE")
                outcome["first_positive_update"] = first
        for rank in range(8):
            rank_dir = root / f"rank-{rank:02d}"
            for name in ("restore", "rng-restored", "complete"):
                current = read(rank_dir / f"{name}.json")
                need(current["rank"] == rank and current["request"] == request, "EXACT_NATIVE_RANK_RECEIPT")
                if name == "restore":
                    need(current["native_named_adamw_states"] == 504 and current["original_cp2_step"] == 2
                         and current["no_intra_epoch_skip"] is True, "EXACT_ORIGINAL_RANK_STATE_RESTORATION")
                elif name == "rng-restored":
                    need(current["device_rng"] is current["restored_CP2"] is True, "ACTUAL_EIGHT_DEVICE_RNG_RESTORES")
                else:
                    need((current["actual_forwards"], current["actual_updates"], current["checkpoints"])
                         == (92, 23, [10, 20, 25]), "ACTUAL_COMPLETE_RANK_CONTINUATION")
        need((outcome["witnessed_forwards"], outcome["rank_optimizer_returns"]) == (736, 184),
             "ACTUAL_FULL_CONTINUATION_DENOMINATORS")
        outcome["checkpoints"] = {str(step): inspect_checkpoint(root / "train" / f"checkpoint-{step}", step)
                                  for step in (10, 20, 25)}
        outcome["passed"] = True
    except (KeyError, TypeError, OSError, ValueError) as error:
        outcome["error_type"] = type(error).__name__
        outcome["error_SHA256"] = sha256(str(error).encode("utf-8", "replace")).hexdigest()
    return outcome


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--plan-sha256", required=True)
    cli.add_argument("--request", required=True)
    args = cli.parse_args()
    need(re.fullmatch(r"[a-f0-9]{32}", args.request) and re.fullmatch(r"[a-f0-9]{64}", args.plan_sha256),
         "EXACT_INDEPENDENT_READER_ARGUMENTS")
    plan = json.loads(checked(args.plan, args.plan_sha256, cap=768 << 10))
    outcome = scan(args.run.resolve(strict=True), args.request, plan)
    with (args.run / "independent-resume25.json").open("xb") as stream:
        stream.write(encode(outcome))
    print(json.dumps({key: value for key, value in outcome.items() if key != "checkpoints"}, sort_keys=True))
    return 0 if outcome["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
