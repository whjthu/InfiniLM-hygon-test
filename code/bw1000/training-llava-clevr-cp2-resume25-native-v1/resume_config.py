"""Private scope binding and exact immutable resume argv."""

import json
import os
from pathlib import Path
import re

from contract import CP2_RUN, INPUTS, TENSORS, checked, checkpoint, digest, encode, need


CP2 = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / CP2_RUN / "train/checkpoint-2"
GUEST = Path("/llava-resume-authority")
CALLBACK = "llava_clevr_cp2_resume25_native_v1"
CP_BINDING = None
UPDATE_PLAN = None
TRAINER = Path("/usr/local/lib/python3.10/dist-packages/transformers/trainer.py")
SWIFT_MIXIN = Path("/swift-runtime/site/swift/trainers/mixin.py")
SWIFT_SHARD = Path("/swift-runtime/site/swift/dataloader/shard.py")
PROVIDERS = {
    TRAINER: "17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a",
    SWIFT_MIXIN: "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9",
    SWIFT_SHARD: "ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c",
}


def source_provider(path, role):
    allowed = {SWIFT_SHARD} if role == "set_epoch" else {TRAINER, SWIFT_MIXIN}
    need(path in allowed, "EXACT_ACTIVE_RESUME_LOADER_PROVIDER")
    checked(path, PROVIDERS[path], cap=512 << 10)


def bind(root, request):
    global CP_BINDING, UPDATE_PLAN
    pin = os.environ["LLAVA_RESUME_AUTH_SHA256"]
    need(re.fullmatch(r"[0-9a-f]{64}", pin) is not None, "EXACT_ROOT_RESUME_AUTH_SHA")
    auth = json.loads(checked(GUEST / "AUTH.json", pin, cap=16384))
    need(auth["request"] == request and auth["GPU_GO"] is True and auth["zero_GRES_CPU_terminal_verified"] is True
         and auth["original_CP2_members_verified"] == 17 and auth["original_INPUTS_SHA256"] == INPUTS
         and auth["original_TENSORS_SHA256"] == TENSORS and auth["original_checkpoint"] == str(CP2)
         and re.fullmatch(r"1103606\.[0-9]+", auth["CPU_step"]) and auth["CPU_step"] != "1103606.4983"
         and os.statvfs(CP2).f_flag & os.ST_RDONLY, "EXACT_ORIGINAL_CP2_AND_ACTUAL_CPU_PLAN_AUTHORITY")
    CP_BINDING = checkpoint(GUEST / "4983.json", CP2, metadata_only=True)
    raw = checked(GUEST / "CP2_NATIVE_RESUME_PLAN.private.json", auth["plan_SHA256"], cap=768 << 10)
    plan = json.loads(raw)
    need(encode(plan) == raw and plan["start"] == CP_BINDING and
         [row["global_step"] for row in plan["updates"]] == list(range(3, 26)), "EXACT_REAL_RANK_SEQUENCE_PLAN")
    for i, step in enumerate(plan["updates"], 3):
        rows = step["rows"]
        need(len(rows) == 32 and [(row["accumulation"], row["rank"]) for row in rows] ==
             [(m, rank) for m in range(1, 5) for rank in range(8)] and
             len({row["ordinal"] for row in rows}) == 32
             and all(row["epoch"] == (i - 1) // 2 and 1 <= row["ordinal"] <= 64 and
                     row["offset"] == 4 * ((i - 1) % 2) + row["accumulation"] - 1
                     and all(re.fullmatch(r"[a-f0-9]{64}", row[key]) for key in
                             ("row_sha256", "image_sha256", "input_sha256")) for row in rows),
             "EXACT_BOUND_23_UPDATE_RANK_ORDER")
    UPDATE_PLAN = {entry["global_step"]: entry["rows"] for entry in plan["updates"]}
    return auth


def resume_argv(original, callback):
    need("--resume_from_checkpoint" not in original and "--resume_only_model" not in original
         and "--ignore_data_skip" not in original and
         original[original.index("--callbacks") + 1] == callback == CALLBACK
         and original[original.index("--max_steps") + 1] == "25",
         "EXACT_NATIVE_SWIFT_CP2_RESUME_FLAGS")
    return [*original, "--resume_from_checkpoint", str(CP2), "--resume_only_model", "false", "--ignore_data_skip", "false"]
