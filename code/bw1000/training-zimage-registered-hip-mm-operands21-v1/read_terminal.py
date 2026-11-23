"""Reconcile the original operand sampling run and its metadata-only output."""

from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL = "training-zimage-coco20-hip-mm-operands21-v1"
REQUEST = "ee8365605c304601a86ef85c071bf5b3"
RUN = ROOT / LABEL
ORIGINAL = ROOT / "training-zimage-coco20-registered-hip-profile21-v1" / "registered-output/result.json"
ORIGINAL_SHA = "02fb232a8b3aee62850fd8a0d29917fa104596d34cc01c0eae39dd8ec7093e80"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
TARGET_SHAPES = {(288, 3840, 3840): 240, (288, 3840, 10240): 90, (288, 10240, 3840): 90}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OOM", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def document(path, *, expected=None):
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    require(0 < len(raw) <= 1 << 20, "Evidence JSON exceeded its bound")
    digest = sha256(raw).hexdigest()
    require(expected is None or digest == expected, "Evidence content identity differs")
    return json.loads(raw), {"bytes": len(raw), "SHA256": digest}


def observe():
    q = subprocess.check_output(["sacct", "-nP", "-j", "1103606", "--format", ",".join(COLUMNS)],
                                text=True, timeout=7)
    rows = [r.split("|") for r in q.splitlines() if "|" + LABEL + "-" in r or r.startswith("1103606.1771|")]
    terminal = {"observed_epoch": time.time(), "request": REQUEST, "accounting_columns": COLUMNS, "accounting": rows}
    if any(row[2].rstrip("+") not in TERMINAL for row in rows) or not (RUN / "outcome.json").is_file():
        return dict(terminal, controller_finished=False)
    terminal["controller_finished"] = True
    outcome, terminal["outcome_pin"] = document(RUN / "outcome.json")
    terminal["outcome"] = {k: outcome[k] for k in ("passed", "post_clean", "gpu_srun_attempted", "device_passed",
                                                  "error_type", "error") if k in outcome}
    pre, terminal["pre_pin"] = document(RUN / "pre.json")
    post, terminal["post_pin"] = document(RUN / "post.json")
    terminal["independent_device_clear"] = (outcome["post_clean"] is True
        and post["node"] == pre["node"] == "f13r2n07"
        and len(set(pre["PCIs"])) == 8 and set(post["PCIs"]) == set(pre["PCIs"])
        and post["kfd_process_count"] == post["own_device_fd_process_count"] == 0
        and any(row[0] == post["step"] and row[2:4] == ["COMPLETED", "0:0"]
                and "gres/" not in row[6] for row in rows))
    worker, terminal["worker_pin"] = document(RUN / "worker-result.json")
    terminal["worker"] = {k: worker[k] for k in ("passed", "slurm", "completed_global_steps",
                                               "actual_optimizer_updates", "error_type", "error") if k in worker}
    if worker["passed"] is not True:
        return terminal
    require(any(row[0] == worker["slurm"] and row[2:4] == ["COMPLETED", "0:0"]
                and row[5] == "f13r2n07" and "gres/dcu=1" in row[6].split(",") for row in rows),
            "Missing actual successful one-card step")
    require(worker["request"] == REQUEST and worker["completed_global_steps"] == [21]
            and worker["actual_optimizer_updates"] == 2, "Wrong training update extent")
    output = RUN / "registered-output"
    result, terminal["result_pin"] = document(output / "result.json", expected=worker["result_SHA256"])
    prior, terminal["prior_profile_result_pin"] = document(ORIGINAL, expected=ORIGINAL_SHA)
    sample, = result["step_metrics"]
    source, = prior["step_metrics"]
    comparison = {k: sample[k] == source[k] for k in ("global_step", "input_SHA256", "loss", "grad_norm")}
    require(all(math.isfinite(sample[k]) for k in ("loss", "grad_norm")), "Non-finite trained scalar")
    comparison["adapter_SHA256"] = result["checkpoints"][0]["adapter_SHA256"] == prior["checkpoints"][0]["adapter_SHA256"]
    terminal["same_platform_numeric"] = {"exact": comparison, "global_step": sample["global_step"],
                                         "loss": sample["loss"], "grad_norm": sample["grad_norm"]}
    probe, terminal["GPU_profiler_smoke_pin"] = document(output / "operator-preflight-private/profile-summary.json")
    terminal["GPU_profiler_smoke"] = {k: probe[k] for k in ("passed", "events", "CUDA_device_events")}
    bounds = worker["actual_operand_observation"]
    require(bounds["file"] == "backward-mm-operands.json", "Unexpected operand artifact")
    metadata, terminal["operand_receipt_pin"] = document(output / "operator-profile-private" / bounds["file"],
                                                         expected=bounds["SHA256"])
    require(bounds["bytes"] == terminal["operand_receipt_pin"]["bytes"] and
            metadata["records_tensor_values"] is False and metadata["records_addresses"] is False,
            "Unexpected private metadata contract")
    totals = {tuple(row["MNK"]): row for row in metadata["totals"]}
    require(len(totals) == len(metadata["totals"]) and set(totals) == set(TARGET_SHAPES),
            "Original three hot shapes differ")
    for shape, row in totals.items():
        joined = sum(signature["host_mm_calls"] for signature in metadata["signatures"]
                     if tuple(signature["MNK"]) == shape)
        require(row.get("metadata_joined", 0) == joined
                and row.get("linked_host_mm_calls", 0) >= joined
                and row["previous_linked_calls"] == TARGET_SHAPES[shape],
                "Matched signature denominator differs")
    terminal["operand_capture"] = {"totals": metadata["totals"], "signature_count": len(metadata["signatures"]),
                                   "examples": metadata["signatures"][:12],
                                   "all_prior_linked_calls_covered": metadata["all_prior_linked_calls_covered"],
                                   "flat_events": metadata["flat_events"], "tree_events": metadata["tree_events"]}
    return terminal


if __name__ == "__main__":
    print(json.dumps(observe(), sort_keys=True, allow_nan=False))
