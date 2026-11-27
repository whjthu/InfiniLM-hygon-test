"""Reconcile every planned LLaVA rank/forward/update and seal checkpoint member hashes."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from train import need


def record(path, request, rank):
    need(path.resolve(strict=True) == path and 0 < path.stat().st_size <= 64 * 1024,
         "EXACT_PRIVATE_RECEIPT_FILE")
    raw = path.read_bytes()
    value = json.loads(raw)
    need(value["request"] == request and value["rank"] == rank, "EXACT_PRIVATE_RANK_IDENTITY")
    return value


def reconcile(root, request):
    slots, updates, initial, missing = [], [], [], []
    forward_count = update_count = loss_count = 0
    for rank in range(8):
        path = root / f"rank-{rank:02d}"
        names = ["start.json", "complete.json",
                 *(f"forward-{index:02d}.json" for index in range(1, 9)),
                 *(f"loss-{index:02d}.json" for index in range(1, 9)),
                 *(f"update-{index}-{action}.json" for index in (1, 2) for action in ("intent", "returned"))]
        values = {}
        for name in names:
            if (path / name).is_file():
                values[name] = record(path / name, request, rank)
            else:
                missing.append([rank, name])
        if "start.json" in values:
            initial.append(values["start.json"]["initial_SHA256"])
        if "complete.json" in values:
            last = values["complete.json"]
            need((last["actual_forwards"], last["actual_updates"], last["finite_losses"],
                  last["native_save_callback"]) == (8, 2, 8, True), "EXACT_RANK_COMPLETE")
        local = []
        audited = []
        for i in range(1, 9):
            sample = values.get(f"forward-{i:02d}.json")
            if sample:
                need((sample["update"], sample["accumulation"]) == ((i - 1) // 4 + 1, (i - 1) % 4 + 1)
                     and all(re.fullmatch(r"[0-9a-f]{64}", sample[key])
                             for key in ("row_sha256", "image_sha256", "input_sha256")),
                     "EXACT_OBSERVED_RANK_FORWARD")
                local.append(sample["ordinal"])
                forward_count += 1
            loss = values.get(f"loss-{i:02d}.json")
            if loss:
                need(loss["finite"] is True and sample is not None, "EXACT_FINITE_OBSERVED_LOSS")
                loss_count += 1
        for update in (1, 2):
            intent = values.get(f"update-{update}-intent.json")
            delta = values.get(f"update-{update}-returned.json")
            if delta:
                rates = delta["learning_rates"]
                norm = delta["delta_l2"]
                need(intent is not None and intent["initial_parameter_SHA256"] == delta["pre_SHA256"]
                     and intent["learning_rates"] == rates and delta["adamw_counters"] == 504
                     and rates and all(math.isfinite(rate) and (rate == 0 if update == 1 else rate > 0)
                                       for rate in rates) and math.isfinite(norm)
                     and ((delta["pre_SHA256"] == delta["post_SHA256"] and norm == delta["changed_tensors"] == 0)
                          if update == 1 else
                          (delta["pre_SHA256"] != delta["post_SHA256"] and norm > 0
                           and 0 < delta["changed_tensors"] <= 504)), "EXACT_ACTUAL_RANK_DELTA")
                audited.append([rates, delta["pre_SHA256"], delta["post_SHA256"], norm, delta["changed_tensors"]])
                update_count += 1
        slots.append(local)
        updates.append(audited)
    result = dict(request=request, planned_ranks=8, planned_forwards=64, planned_rank_updates=16,
                  recorded_forwards=forward_count, recorded_finite_losses=loss_count, recorded_rank_updates=update_count,
                  missing=missing, passed=False)
    if missing:
        return result
    need(len(initial) == 8 and len(set(initial)) == 1 and updates == [updates[0]] * 8
         and updates[0][0][1] == initial[0] and updates[0][1][1] == updates[0][0][2]
         and sorted(slot for row in slots for slot in row) == list(range(1, 65)), "EXACT_EIGHT_RANKS_64_SOURCE_AND_DELTA")
    cp = root / "train/checkpoint-2"
    need(cp.resolve(strict=True) == cp, "CANONICAL_CP2")
    paths = {path.name: path for path in cp.iterdir()}
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json",
                *(f"rng_state_{rank}.pth" for rank in range(8))}
    need(required <= set(paths) and len(paths) <= 32, "EXACT_REQUIRED_NATIVE_CP2_MEMBERS")
    trainer = json.loads(paths["trainer_state.json"].read_bytes())
    need((trainer["global_step"], trainer["max_steps"]) == (2, 25), "ACTUAL_CP2_TRAINER_STATE")
    files = {}
    for name, path in sorted(paths.items()):
        need(path.is_file() and path.resolve(strict=True) == path, "REGULAR_NATIVE_CP2_MEMBER")
        before = path.stat()
        digest = sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024**2):
                digest.update(chunk)
        after = path.stat()
        fields = ("st_size", "st_ino", "st_dev", "st_ctime_ns", "st_mtime_ns")
        need(all(getattr(before, name) == getattr(after, name) for name in fields) and before.st_size > 0,
             "UNCHANGED_DURABLE_NATIVE_CP2_MEMBER")
        files[name] = dict(bytes=before.st_size, SHA256=digest.hexdigest())
    return dict(result, passed=True, checkpoint_files=files, learning_rates=[row[0] for row in updates[0]],
                initial_SHA256=initial[0], final_SHA256=updates[0][-1][2], final_delta_l2=updates[0][-1][3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("request")
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{32}", args.request) is not None, "EXACT_REQUEST_FORMAT")
    try:
        result = reconcile(args.root.resolve(strict=True), args.request)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = dict(request=args.request, planned_ranks=8, planned_forwards=64, planned_rank_updates=16, passed=False,
                      error_class=type(error).__name__, error_message_SHA256=sha256(str(error).encode()).hexdigest())
    with (args.root / "independent-first2.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps({key: result[key] for key in ("request", "planned_forwards", "planned_rank_updates", "passed")}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
