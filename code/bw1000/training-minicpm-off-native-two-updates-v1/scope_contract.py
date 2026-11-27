"""Pure file and update gates for one native OFF training attempt."""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

INPUT_SHA = "9a1338a008b91e2524873e12f49517889a23af4a9869232d5fa1c6103a0e60d4"
ORIGINAL_REQUEST = "1fde0c1744ce40898ed9d7a2724f0fac"
INPUT_ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/training-minicpm-off-native6-cpu-1fde0c1744ce/train"
ORIGINAL_WORKER_SHA = "839ac443deb900103175835b5db9ba08702ca854c52659bd68b890cb5545b6b1"
ORIGINAL_ARCHIVE_SHA = "0f35f9af0a091016a9f3857ba28cbb19ead563ae42645a9b91ed539385d52369"


def need(test, code):
    if not test:
        raise ValueError(code)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(record):
    return (json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def exact_4479(independent, worker):
    need(sha(worker) == ORIGINAL_WORKER_SHA, "OFF4479_SOURCE_PINS")
    observed = json.loads(worker)
    row = independent["accounting"]
    need(row[:7] == ["1103606.4479", "training-minicpm-off-native6-cpu-1fde0c1744ce",
                     "COMPLETED", "0:0", "34", "f12r2n12", "cpu=8,mem=12G,node=1"]
         and independent["request"] == ORIGINAL_REQUEST and independent["passed"] is True
         and independent["planned"] == independent["completed"] == 6
         and independent["actual_optimizer_updates"] == 0 and independent["model_loaded"] is False, "OFF4479_INDEPENDENT")
    need(observed["source_dataset_SHA256"] == INPUT_SHA and observed["passed"] is True
         and observed["planned_rows"] == observed["completed_rows"] == observed["EOS_supervised"] == observed["full_untruncated"] == 6
         and observed["encode_calls"] == 12 and observed["collator_calls"] == 6 and observed["unique_media"] == 4
         and observed["GPU_allocation"] == observed["actual_optimizer_updates"] == 0
         and observed["dev_or_blind_read"] is observed["model_loaded"] is False, "OFF4479_WORKER")


def view_bytes(rows, mounted_root=INPUT_ROOT):
    need(len(rows) == 6 and [r["id"] for r in rows] == [f"off-train-{i:02d}" for i in range(1, 7)]
         and len({x["images"][0] for x in rows}) == 4, "OFF_VIEW_SOURCE")
    need(PurePosixPath(mounted_root).is_absolute(), "OFF_VIEW_MOUNT_ROOT")
    lines = []
    for i, row in enumerate(rows, 1):
        image = PurePosixPath(row["images"][0].replace("\\", "/")).name
        need(re.fullmatch(r"off-image-0[1-4]\.jpg", image), "OFF_VIEW_MEDIA")
        lines.append(encoded(dict(row, images=[mounted_root + "/" + image])))
    order = [i % 6 for i in range(800)]
    view = b"".join(lines[i] for i in order)
    ledger = dict(source_manifest_SHA256=INPUT_SHA, source_rows=6, physical_rows=800,
                  original_index_by_physical_row=order, view_SHA256=sha(view),
                  original_derived_row_SHA256=[sha(x) for x in lines], exposure_counts=[order.count(i) for i in range(6)],
                  repeated_source_rows=True, dev_or_blind_read=False)
    return view, ledger


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def parameter_identity(record):
    need(re.fullmatch(r"[0-9a-f]{64}", record["SHA256"]) and type(record["tensors"]) is int and record["tensors"] > 0
         and record["elements"] == record["floating_elements"] == 10822656 and record["finite"] is True
         and number(record["floating_l2"]) and record["floating_l2"] >= 0, "OFF_PARAMETER_IDENTITY")


def audit_update(record, step):
    need(type(step) is int and step in (1, 2) and record["step"] == step, "OFF_UPDATE_ORDINAL")
    before, after, rates, delta = (record[x] for x in ("before", "after", "lr", "delta"))
    parameter_identity(before)
    parameter_identity(after)
    need(before["tensors"] == after["tensors"] == delta["parameter_tensors"], "OFF_PARAMETER_COUNT")
    need(isinstance(rates, list) and rates and all(number(x) and (x == 0 if step == 1 else x > 0) for x in rates), "OFF_ACTUAL_LR")
    moved = step == 2
    need((before["SHA256"] != after["SHA256"]) == moved
         and all(number(delta[k]) and (delta[k] > 0 if moved else delta[k] == 0) for k in ("l2", "max_abs"))
         and all(type(delta[k]) is int and (delta[k] > 0 if moved else delta[k] == 0)
                 for k in ("changed_elements", "changed_tensors"))
         and delta["changed_elements"] <= before["elements"] and delta["changed_tensors"] <= before["tensors"], "OFF_PARAMETER_MOVEMENT")


def audit_ranks(records, request):
    need(len(records) == 8 and [r["rank"] for r in records] == list(range(8)), "OFF_RANK_DENOMINATOR")
    all_updates, saved = [], []
    for row in records:
        need(row["request"] == request and row["world_size"] == 8 and row["completed_global_step"] == 2
             and row["saved_checkpoints"] == [2] and len(row["inputs"]) == len(row["outputs"]) == 8
             and [x["ordinal"] for x in row["inputs"]] == [x["ordinal"] for x in row["outputs"]] == list(range(1, 9))
             and [x["global_step"] for x in row["inputs"]] == [0]*4 + [1]*4
             and all(x["tokens"]["shifted_supervised_tokens"] > 0 for x in row["inputs"])
             and all(number(x["loss_scalar"]) for x in row["outputs"]) and len(row["steps"]) == 2, "OFF_RANK_FORWARD_OR_STEPS")
        updates = []
        for step, result in enumerate(row["steps"], 1):
            update = result["actual_optimizer"]
            need(result["global_step"] == step == result["optimizer_step_min"] == result["optimizer_step_max"]
                 and result["optimizer_slot_count"] == update["before"]["tensors"], "OFF_ADAMW_COUNTER")
            audit_update(update, step)
            updates.append(update)
        need(updates[0]["after"] == updates[1]["before"], "OFF_SAME_PARAMETER_CONTINUITY")
        all_updates.append(updates)
        parameter_identity(row["checkpoint_adapter"])
        saved.append(row["checkpoint_adapter"])
    need(all(r == all_updates[0] for r in all_updates) and all(r == saved[0] for r in saved), "OFF_EIGHT_RANK_UPDATE_AGREEMENT")
    return dict(request=request, planned_ranks=8, verified_ranks=8, planned_native_forwards=64, native_forwards=64,
                planned_optimizer_updates=2, actual_optimizer_updates=2,
                same_update_on_all_ranks=True, distinct_source_rows=6, capacity_exposures=800,
                update_boundaries=all_updates[0], adapter=saved[0])
