"""Read the original terminal Z-Image windows and verify the checkpoint and clock evidence."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
import sys


LABEL = "training-zimage-coco20-registered-hip-window3-v1"
REQUEST = "ef761e04519b422ba7f463ee6f19aa86"
PARENT = "1103606"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "DEADLINE", "BOOT_FAIL"}
REPEATS = ("repeat01", "repeat02", "repeat03")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, limit=1 << 20):
    require(not path.is_symlink(), "unexpected evidence symlink: " + path.name)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit, "evidence exceeds bound: " + path.name)
    return json.loads(data), sha256(data).hexdigest()


def file_pin(path, limit=512 << 20):
    require(not path.is_symlink(), "unexpected checkpoint symlink")
    fingerprint = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            size += len(chunk)
            require(size <= limit, "checkpoint exceeds original audit bound")
            fingerprint.update(chunk)
    return size, fingerprint.hexdigest()


def accounting():
    args = ["sacct", "-n", "-P", "-j", PARENT, "-o", ",".join(COLUMNS)]
    lines = subprocess.check_output(args, text=True, timeout=10).splitlines()
    found = {}
    for line in lines:
        row = line.split("|")
        if len(row) != len(COLUMNS) or not row[1].startswith(LABEL + "-"):
            continue
        phase = row[1].removeprefix(LABEL + "-")
        require(phase in {"control", "pre", "baseline", "post"} and phase not in found
                and re.fullmatch(PARENT + r"\.[0-9]+", row[0]), "unrecognized phase or step")
        require(row[5] == ("f13r2n08" if phase == "control" else "f13r2n07"), "wrong accounting node")
        gres = [part for part in row[6].split(",") if part.startswith("gres/")]
        require(gres == (["gres/dcu=1"] if phase == "baseline" else []), "changed accounting GRES allocation")
        found[phase] = row
    return found


def check_window(window):
    require(window["clock"] == "monotonic_ns" and window["warmup_updates"] == 5 and window["measured_updates"] == 20
            and window["update_ordinals"] == list(range(6, 26)) and window["checkpoint_ordinals"] == [10, 20, 25]
            and window["includes_input_fetch_and_durable_checkpoint_io"] is True
            and window["profile"] is False and window["first_update_capture"] is False,
            "window workload or clock boundary changed")
    start, end = window["start_ns"], window["end_ns"]
    require(type(start) is int and type(end) is int and end > start >= 0, "bad native timer interval")
    seconds = (end - start) / 1_000_000_000
    require(math.isclose(window["seconds"], seconds, rel_tol=1e-12)
            and math.isclose(window["updates_per_second"], 20 / seconds, rel_tol=1e-12), "bad native timer arithmetic")
    return 20 / seconds


def read(root):
    require(root.name == LABEL and root.resolve() == root, "wrong sealed run directory")
    rows = accounting()
    answer = dict(request=REQUEST, label=LABEL, accounting_columns=COLUMNS, accounting=rows,
                  formal_A100_numerics=False, stage_three_accepted=False, heldout_effect_verified=False,
                  planned_repeats=3, qualified_repeats=0, completed_optimizer_updates=0, repeats={})
    if "control" not in rows or rows["control"][2] not in TERMINAL or any(row[2] not in TERMINAL for row in rows.values()):
        return dict(answer, state="ORIGINAL_ACCOUNTING_NOT_TERMINAL")
    if not (root / "outcome.json").is_file():
        return dict(answer, state="ORIGINAL_CONTROLLER_OUTCOME_MISSING")
    outcome, outcome_sha = load(root / "outcome.json")
    require(outcome["request"] == REQUEST, "wrong original controller result")
    answer.update(controller_outcome_SHA256=outcome_sha, original_controller_passed=outcome["passed"],
                  independent_post_clean=outcome.get("post_clean", False), failure=outcome.get("error_type"))
    if "baseline" not in rows or not (root / "batch-result.json").is_file():
        return dict(answer, state="ORIGINAL_TRAINING_RECEIPTS_MISSING")
    gpu_step = rows["baseline"][0]
    batch, batch_sha = load(root / "batch-result.json")
    require(batch["request"] == REQUEST and batch["planned_repeats"] == list(REPEATS)
            and batch_sha == outcome["batch_SHA256"], "batch result differs from controller")
    pre, pre_sha = load(root / "pre.json")
    answer["pre_SHA256"] = pre_sha
    if (root / "post.json").is_file():
        post, post_sha = load(root / "post.json")
        answer["post_SHA256"] = post_sha
        require(pre["PCIs"] == post["PCIs"] and pre["boot_id_SHA256"] == post["boot_id_SHA256"]
                and post["kfd_process_count"] == post["own_device_fd_process_count"] == 0,
                "original independent device postcheck differs")
    gpu = None
    pids = set()
    rates = []
    for name in REPEATS:
        receipt_path = root / (name + "-worker-result.json")
        if not receipt_path.is_file():
            continue
        worker, worker_sha = load(receipt_path)
        require(worker["request"] == REQUEST and worker["phase"] == name and worker["slurm"] == gpu_step
                and batch["repeats"].get(name) == worker, "worker differs from the exact batch receipt")
        if not worker["passed"]:
            answer["repeats"][name] = dict(passed=False, worker_SHA256=worker_sha, error=worker.get("error_type"),
                                          last_progress_update_count=len(worker["completed_global_steps"]))
            continue
        original, original_sha = load(root / name / "result.json")
        require(worker["result_SHA256"] == original_sha and original["passed"] is True
                and original["completed_global_steps"] == worker["completed_global_steps"] == list(range(1, 26))
                and original["actual_optimizer_updates"] == 25 and original["mode"] == "continuous"
                and original["frozen_parameters_unchanged"] is True and original["initial_tensors"] == 476
                and original["original_fused_base"]["matched_tensors"] == 521 and original["profile_summary"] is None
                and original["first_update_capture_bytes"] == 0 and original["restore_info"] is None,
                "one registered native result has the wrong training state")
        pci = worker["physical_pci"]
        require(pci in pre["PCIs"] and (gpu is None or pci == gpu) and worker["owner_pid"] not in pids,
                "repeated process identity or hardware drift")
        gpu = pci
        pids.add(worker["owner_pid"])
        window = original["engineering_complete_window"]
        require(window == worker["window"], "worker and native clock records differ")
        rate = check_window(window)
        result_checkpoints = original["checkpoints"]
        require([item["global_step"] for item in result_checkpoints] == [10, 20, 25], "wrong checkpoint ordinals")
        checks = []
        for item in result_checkpoints:
            step = item["global_step"]
            size, digest = file_pin(root / name / f"checkpoint-{step}" / "state.private.pt")
            require((size, digest) == (item["bytes"], item["SHA256"]), "original checkpoint identity changed")
            checks.append(dict(global_step=step, bytes=size, SHA256=digest))
        rates.append(rate)
        answer["repeats"][name] = dict(passed=True, clock=window, checkpoints=checks, worker_SHA256=worker_sha,
                                      registered_result_SHA256=original_sha, load_seconds=original["load_seconds"],
                                      peak_allocated_bytes=original["peak_allocated_bytes"], peak_reserved_bytes=original["peak_reserved_bytes"])
    answer.update(qualified_repeats=len(rates), completed_optimizer_updates=25 * len(rates), actual_gpu_step=gpu_step,
                  batch_SHA256=batch_sha, same_PCI_verified=len(rates) == 3, independent_process_count=len(pids))
    if (len(rates) != 3 or not outcome["passed"] or not answer["independent_post_clean"]
            or set(rows) != {"control", "pre", "baseline", "post"} or any(row[2:4] != ["COMPLETED", "0:0"] for row in rows.values())):
        return dict(answer, state="THREE_QUALIFIED_WINDOWS_NOT_AVAILABLE")
    claim, claim_sha = load(root / "CLAIM.json")
    reference = claim["A100_complete_window_reference"]
    a100 = median(20 / seconds for seconds in reference["repeats"])
    require(math.isclose(a100, reference["median_updates_per_second"], rel_tol=1e-12), "A100 reference median differs")
    return dict(answer, state="THREE_MATCHED_BOUNDARY_ENGINEERING_WINDOWS", claim_SHA256=claim_sha,
                framework_commit=claim["common_candidate"]["candidate_commit"], rate_units="updates_per_second",
                hygon_median=median(rates), a100_median=a100, engineering_rate_ratio=median(rates)/a100,
                a100_source_SHA256=reference["source_SHA256"])


if __name__ == "__main__":
    print(json.dumps(read(Path(sys.argv[1])), sort_keys=True, allow_nan=False))
