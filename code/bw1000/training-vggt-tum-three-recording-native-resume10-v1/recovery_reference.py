"""Bind the original completed VGGT run and its untouched checkpoint 10."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat


ORIGINAL_LABEL = "training-vggt-tum-three-recording-native25-resume20-v1"
ORIGINAL_REQUEST = "7b07ac6c8bd3e86e63e7528fd1c885f0"
ORIGINAL_PUBLICATION = "12d7d21f06bfea4e0afc6a0b0af385ad36c214b0"
ORIGINAL_INTENT_SHA = "3bacc370eaf04360b51aa34fe11e44770fe1cc43189054854120cfd0791fd7a7"
CONTINUOUS_SHA = "096d217a5ce227fc3093d6fb808f8f07bf0703c822cbe5e7fb0d666bff13efb4"
ORIGINAL_PUBLIC_SHA = "b1fa0d5ffd8f072f18f8306487905e451699fd7681a5cf5255dd7171fb7aa04c"
RESUME_AFTER, FINAL_STEP = 10, 25
CP10_BYTES = 2594191507
CP10_SHA = "8925acf4cbab54fe6023480b39f78482681946404fee49cae5a9b31a1f588fba"
CP10_BOUNDARY_SHA = "23a55e890ce777bdaa8fc6b4932d1db14b2e70511efac116470b29b13896953e"
CHECKPOINT = Path("arms/continuous/train/checkpoint-10.pt")
CONTINUOUS = Path("arms/continuous/receipt.json")


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def signature(stat_result):
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
            stat_result.st_mtime_ns, stat_result.st_ctime_ns)


def extent(path, cap):
    before = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
         "bounded original file is missing or aliased")
    return before


def read_pinned_json(path, expected, cap=65536):
    before = extent(path, cap)
    with path.open("rb") as stream:
        need(signature(os.fstat(stream.fileno())) == signature(before), "original JSON open identity changed")
        raw = stream.read(cap + 1)
        need(signature(os.fstat(stream.fileno())) == signature(before), "original JSON changed during read")
    need(signature(path.lstat()) == signature(before) and len(raw) == before.st_size
         and sha256(raw).hexdigest() == expected, "original JSON evidence changed")
    return json.loads(raw)


def validate_reference(public, intent, baseline):
    need(public["request"] == ORIGINAL_REQUEST and public["publication"] == ORIGINAL_PUBLICATION
         and public["state"] == "ACTUAL_ONE_DCU_VGGT_TUM_25_PLUS_5_EXACT_RECOVERY_POST_CLEAN"
         and public["accounting"]["gpu"][0] == "1103606.927"
         and public["accounting"]["gpu"][2:4] == ["COMPLETED", "0:0"]
         and public["actual_independent_post"][0] == "1103606.929"
         and public["actual_independent_post"][2:4] == ["COMPLETED", "0:0"],
         "original continuous accounting changed")
    evidence = public["receipt"]
    need(evidence["request"] == ORIGINAL_REQUEST and evidence["qualified"] is True
         and evidence["post_clean"] is True
         and evidence["actual_updates"] == {"continuous": FINAL_STEP, "resume20": 5}
         and evidence["artifact_SHA256"]["intent.json"] == ORIGINAL_INTENT_SHA
         and evidence["artifact_SHA256"][CONTINUOUS.as_posix()] == CONTINUOUS_SHA,
         "original terminal does not bind the continuous receipt")
    need((intent["request"], intent["publication"], intent["parent"], intent["node"]) ==
         (ORIGINAL_REQUEST, ORIGINAL_PUBLICATION, "1103606", "f13r4n09")
         and intent["deadline"] == intent["epoch"] + 1800
         and intent["epoch"] == baseline["epoch"] and baseline["deadline"] == intent["deadline"] - 100,
         "original intent and worker epoch do not bind the reference")
    cp10 = baseline["checkpoints"][str(RESUME_AFTER)]
    need(baseline["arm"] == "continuous" and baseline["executed_optimizer_steps"] == FINAL_STEP
         and baseline["resume_source_SHA256"] is None and cp10["global_step"] == RESUME_AFTER
         and cp10["next_global_step"] == RESUME_AFTER + 1 and cp10["next_batch_cursor"] == RESUME_AFTER % 9
         and cp10["file_SHA256"] == CP10_SHA and cp10["full_boundary_SHA256"] == CP10_BOUNDARY_SHA,
         "not the actual original checkpoint 10")
    return cp10


def bind_reference(root, public_file):
    public = read_pinned_json(public_file, ORIGINAL_PUBLIC_SHA)
    intent = read_pinned_json(root / "intent.json", ORIGINAL_INTENT_SHA)
    baseline = read_pinned_json(root / CONTINUOUS, CONTINUOUS_SHA)
    cp10 = validate_reference(public, intent, baseline)
    before = extent(root / CHECKPOINT, 8 << 30)
    need(before.st_size == CP10_BYTES, "original checkpoint 10 extent differs")
    return baseline, dict(reference_request=ORIGINAL_REQUEST, reference_gpu_step="1103606.927",
                          reference_public_SHA256=ORIGINAL_PUBLIC_SHA, reference_intent_SHA256=ORIGINAL_INTENT_SHA,
                          reference_continuous_receipt_SHA256=CONTINUOUS_SHA, checkpoint10_SHA256=cp10["file_SHA256"],
                          checkpoint10_bytes=before.st_size)


def checkpoint_digest(path, expected_size=None):
    before = extent(path, 8 << 30)
    need(expected_size is None or before.st_size == expected_size, "checkpoint extent changed")
    digest = sha256()
    total = 0
    with path.open("rb") as stream:
        need(signature(os.fstat(stream.fileno())) == signature(before), "original checkpoint open identity changed")
        while block := stream.read(4 << 20):
            total += len(block)
            need(total <= before.st_size, "original checkpoint grew")
            digest.update(block)
        need(signature(os.fstat(stream.fileno())) == signature(before), "original checkpoint changed during read")
    need(total == before.st_size and signature(path.lstat()) == signature(before), "checkpoint was unstable")
    return digest.hexdigest()


def check_checkpoint(root, binding):
    path = root / CHECKPOINT
    need(checkpoint_digest(path, binding["checkpoint10_bytes"]) == binding["checkpoint10_SHA256"],
         "original checkpoint 10 bytes differ")
    return path
