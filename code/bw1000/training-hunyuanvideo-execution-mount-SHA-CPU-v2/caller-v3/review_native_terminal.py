"""Audit the fixed native T09 run after terminal Slurm accounting is visible."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from check_hunyuan_sft_replay_inputs import WEIGHT_SHA, WEIGHT_SIZE
import re
import stat

from framework_binding import COMMIT, stamp

PARENT = "1103606"
NODE = "f13r3n13"
PHASES = ("before", "inputs", "forward", "preclip", "postclip", "updated")
CHECKPOINTS = (10, 20, 25)
COLUMNS = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList"]
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def file_evidence(path, limit, *, content=False):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError(f"Unexpected file type or extent: {path.name}")
    digest = hashlib.sha256()
    data = bytearray()
    remaining = before.st_size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024**2, remaining))
            if not chunk:
                raise ValueError(f"Short read: {path.name}")
            digest.update(chunk)
            if content:
                data.extend(chunk)
            remaining -= len(chunk)
        if stream.read(1) or stamp(before) != stamp(path.lstat()):
            raise ValueError(f"File changed during audit: {path.name}")
    return dict(bytes=before.st_size, SHA256=digest.hexdigest()), bytes(data)


def exact_step(accounting, scope, phase):
    handle = accounting[f"{phase}_step"]
    if not isinstance(handle, str) or not re.fullmatch(r"1103606\.[0-9]+", handle):
        raise ValueError("Wrong Slurm parent or step identity")
    rows = [r for r in accounting["accounting"] if r[0] == handle]
    if (len(rows) != 1 or len(rows[0]) != len(COLUMNS)
            or rows[0][1] != f"{scope}-{phase}" or rows[0][5] != NODE
            or rows[0][2].split()[0] not in TERMINAL):
        raise ValueError(f"{phase} accounting is not terminal for this run")
    return rows[0]


def decode(path, maximum):
    identity, raw = file_evidence(path, maximum, content=True)
    return identity, json.loads(raw)


def post_clean(workspace, post_step, device):
    _, before = decode(workspace / "pre" / f"{NODE}.json", 65536)
    identity, after = decode(workspace / "post" / f"{NODE}.json", 65536)
    cards = before.get("pci")
    if (type(cards) is not list or len(cards) != 8 or len(set(cards)) != 8
            or before.get("node") != NODE or after.get("node") != NODE
            or after.get("step") != post_step or after.get("pci") != cards
            or before.get("boot_id") != device.get("boot_id") or after.get("boot_id") != device.get("boot_id")
            or device["device"]["pci"] not in cards
            or any(type(row.get("kfd_holder_count")) is not int or row["kfd_holder_count"] != 0
                   for row in (before, after))):
        raise ValueError("Independent host post-device evidence differs")
    return identity


def inspect(workspace, accounting):
    if accounting.get("accounting_columns") != COLUMNS:
        raise ValueError("Unexpected Slurm accounting columns")
    gpu = exact_step(accounting, workspace.name, "gpu")
    post = exact_step(accounting, workspace.name, "post")
    if gpu[0] == post[0] or int(post[0].split(".")[1]) <= int(gpu[0].split(".")[1]):
        raise ValueError("Post must be a separate later step")
    result = dict(case="T09.sft", gpu_step=gpu[0], post_step=post[0], gpu_slurm_state=gpu[2],
                  post_slurm_state=post[2], planned_steps=25, completed_steps=0,
                  planned_capture_phases=list(PHASES), verified_capture_phases=[],
                  planned_checkpoints=list(CHECKPOINTS), verified_checkpoint_files={},
                  post_device_verified=False, native_evidence_complete=False, numerical_thresholds_qualified=False,
                  recovery_qualified=False, performance_qualified=False)
    try:
        _, device = decode(workspace / "assigned-device.json", 65536)
        if (device["step"] != gpu[0] or device["node"] != NODE
                or device["framework"]["framework_commit"] != COMMIT):
            raise ValueError("Assigned device and framework evidence differ from the terminal step")
        try:
            result["post_device_receipt"] = post_clean(workspace, post[0], device)
            result["post_device_verified"] = True
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["post_device_error_type"] = type(error).__name__
        transformer = device["transformer"]
        bytes_by_verification = {"ORIGINAL_FINGERPRINT": 0, "EXECUTION_MOUNT_FULL_SHA256": WEIGHT_SIZE}
        if (transformer.get("state") != "VERIFIED_RECEIVER_IDENTITY"
                or transformer.get("weight_SHA256") != WEIGHT_SHA or transformer.get("final_bytes") != WEIGHT_SIZE
                or transformer.get("verification") not in bytes_by_verification
                or transformer.get("payload_bytes_read") != bytes_by_verification[transformer["verification"]]):
            raise ValueError("Missing or inconsistent execution-mount weight evidence")
        result["transformer_admission"] = transformer
        root = workspace / "outputs/hunyuanvideo-1.5/train/fixed-a100-replay"
        step_pin, raw = file_evidence(root / "steps.ndjson", 65536, content=True)
        result["step_log"] = step_pin
        for line in raw.splitlines():
            row = json.loads(line)
            step = row["global_step"]
            if type(step) is not int or step != result["completed_steps"] + 1 or step > 25:
                raise ValueError("Unexpected native step sequence")
            if any(not math.isfinite(row[name]) for name in ("loss", "grad_norm", "step_seconds")):
                raise ValueError("Nonfinite native step evidence")
            if row["grad_norm"] <= 0 or row["step_seconds"] <= 0:
                raise ValueError("Nonpositive gradient norm or step time")
            result["completed_steps"] = step
        capture = workspace / "capture"
        pin, receipt = decode(capture / "capture-result.json", 16384)
        if (receipt.get("case") != "T09.sft" or receipt.get("actual_global_step") != 1
                or receipt.get("fixed_a100_input_replay") is not True
                or receipt.get("performance_eligible") is not False
                or tuple(row["phase"] for row in receipt.get("artifacts", [])) != PHASES):
            raise ValueError("Unexpected first-update capture receipt")
        result["capture_receipt"] = pin
        total = 0
        for row in receipt["artifacts"]:
            name = row["phase"] + ".pt"
            if row["file"] != name:
                raise ValueError("Unexpected capture path")
            identity, _ = file_evidence(capture / name, 128 * 1024**2)
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError("Capture file identity differs")
            total += identity["bytes"]
            result["verified_capture_phases"].append(row["phase"])
        if total != receipt["artifact_bytes"] or total > 256 * 1024**2:
            raise ValueError("Invalid capture byte total")
        for step in CHECKPOINTS:
            identity, _ = file_evidence(root / f"checkpoint-{step}/state.pt", 64 * 1024**2)
            result["verified_checkpoint_files"][str(step)] = identity
        pin, summary = decode(root / "summary.json", 65536)
        if (summary["optimizer"] != "Muon" or summary["trainable_parameters"] != 1769472
                or summary["lora_targets"] != ["img_attn_q", "img_attn_v"]
                or summary["lora_rank"] != 4 or summary["configured_warmup_steps"] != 5
                or summary["configured_measured_steps"] != 20
                or summary["completed_global_step_at_start"] != 0
                or summary["external_resume"] is not False):
            raise ValueError("Unexpected native training summary")
        result["summary"] = pin
        result["native_evidence_complete"] = (
            gpu[2:4] == ["COMPLETED", "0:0"] and post[2:4] == ["COMPLETED", "0:0"]
            and result["post_device_verified"] and result["completed_steps"] == 25
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["artifact_error_type"] = type(error).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("accounting", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, accounting = decode(args.accounting, 65536)
    result = inspect(args.workspace, accounting)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
