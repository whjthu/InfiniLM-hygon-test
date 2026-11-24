"""Independently consume the original T05 zero-update Slurm diagnostic."""

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys


REQUEST = "cbbd25487dca4090a52c923bbfb7f60b"
PARENT, NODE, PORT = "1103606", "f13r1n19", 29953
LABEL = "training-minicpm-checkpoint20-sdpa-repeatability-target-v1"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = ROOT / LABEL
ORIGINAL = ROOT / "training-minicpm-usgs-exposure25-resume20-target-v2"
MANIFEST = "5bf1503b7af5863de7b3c8b5dc330d0e6f5bc1a8d0a20abab7988bf0cf8792b2"
PRE, GPU = "1103606.2302", "1103606.2303"
ARMS = ("a1", "b1", "b2", "a2")
PHASES = ("pre", "gpu", "post")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
PCI = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
SOURCE = {
    "route_control.py": "16ad617be313454684d1879da800837ec223d081fefd264c2b77848ceb4de8ba",
    "read_completed.py": "2c6702af06e8edb98cae9963f0f66abcafa38c544c159efba33f727e79d5933f",
    "original-observer/compare_first.py": "e11c8b7b0955be3ea4b3e35dc8d747cb7edfbf51f62a0416ffad765b406200dd",
    "original-observer/probe.py": "1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path, limit, expected=None):
    require(path.resolve(strict=True) == path and path.is_file(), "aliased evidence path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    pin = sha256(raw).hexdigest()
    require(0 < len(raw) <= limit and (expected is None or pin == expected), "evidence bytes differ")
    return raw, pin


def select_rows(raw):
    rows = {phase: None for phase in PHASES}
    labels = {LABEL + "-" + phase: phase for phase in PHASES}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = line.split("|")
        require(len(row) == 7, "accounting does not contain seven columns")
        phase = labels.get(row[1])
        if phase is None:
            continue
        require(rows[phase] is None and row[5] == NODE and re.fullmatch(PARENT + r"\.[0-9]+", row[0]),
                "duplicate or foreign accounting step")
        entries = [item.split("=", 1) for item in row[6].split(",")]
        resources = dict(entries)
        limits = {"cpu": "64", "gres/dcu": "8", "mem": "224000M", "node": "1"} if phase == "gpu" else {
            "cpu": "2", "mem": "2048M", "node": "1"}
        if resources.get("mem") == "2G":
            resources["mem"] = "2048M"
        require(len(entries) == len(resources) and resources == limits and row[4].isdigit(),
                "step resources differ")
        rows[phase] = row
    require(rows["pre"] and rows["pre"][0] == PRE and rows["pre"][2:4] == ["COMPLETED", "0:0"]
            and rows["gpu"] and rows["gpu"][0] == GPU, "original pre or GPU step differs")
    if rows["post"]:
        require(int(rows["post"][0].rsplit(".", 1)[1]) > int(GPU.rsplit(".", 1)[1]),
                "post is not a separate later step")
    return rows


def state(row):
    return None if row is None else row[2].split()[0].rstrip("+")


def validate_devices(before, after, worker, post, epoch):
    card_ids = before["pci"]
    require(len(card_ids) == len(set(card_ids)) == 8 and all(PCI.fullmatch(x) for x in card_ids),
            "device inventory must contain eight distinct PCI IDs")
    require(worker["request"] == REQUEST and worker["epoch"] == epoch and worker["actual_step"] == GPU
            and sorted(worker["visible_PCI_by_index"]) == card_ids and len(set(worker["CPU_affinity"])) >= 64,
            "worker exact scope or device lease differs")
    lease = worker["leased_snapshot"]
    require(lease["node"] == NODE and lease["step"] == GPU and lease["boot_id"] == worker["boot_id"]
            and lease["pci"] == card_ids and sorted(lease["pci_to_kfd"]) == card_ids
            and lease["kfd_holder_count"] == 0 and lease["full_fd_inventory_performed"] is False
            and lease["holder_inventory_basis"] == "kernel_KFD_proc_only"
            and 0 <= lease["monotonic_ns"] - before["monotonic_ns"] <= 45_000_000_000,
            "original container PCI lease or freshness differs")
    for label, row, snap in (("pre", [PRE], before), ("post", post, after)):
        require(row and snap["node"] == NODE and snap["step"] == row[0] and snap["request"] == REQUEST
                and snap["phase"] == label and snap["master_port"] == PORT and snap["boot_id"] == worker["boot_id"]
                and snap["full_fd_inventory_performed"] is True and snap["kfd_holder_count"] == 0
                and snap["holder_inventory_basis"] == "kernel_KFD_plus_sameUID_FD_complete"
                and snap["pci"] == card_ids and snap["pci_to_kfd"] is None,
                "pre or post physical device scope differs")
    require(lease["monotonic_ns"] < after["monotonic_ns"] and after["owned_registry_complete"] is True
            and after["owned_processes_remaining"] == [], "same boot post has no complete cleanup proof")


def recompute(run, original):
    directory = run / "target-worker"
    for name, pin in SOURCE.items():
        read(directory / name, 32 * 1024, pin)
    sys.path[:0] = [str(directory), str(directory / "original-observer")]
    try:
        from compare_first import original_rows
        from read_completed import read_arms
        from route_control import comparison
        baseline = original_rows(original, directory / "original1171.json")
        records, pins, rejected = read_arms(run, GPU)
        result = comparison(records, GPU, baseline)
    finally:
        del sys.path[:2]
    return result, pins, rejected


def inspect(run, original, rows, compare=recompute):
    status = state(rows["gpu"])
    post = rows["post"]
    report = dict(request=REQUEST, numbered_slurm=rows, actual_gpu_terminal=status in TERMINAL,
                  planned_rank_forwards=32, actual_optimizer_updates=None,
                  original_worker_comparison_agrees=False, clean_independent_post=False,
                  diagnosis_complete=False, formal_recovery_performance_or_effect_qualified=False)
    if status not in TERMINAL or state(post) not in TERMINAL:
        return report
    pins = {}

    def document(name, cap=32768):
        raw, pins[name] = read(run / name, cap)
        return json.loads(raw)

    intent = document("diagnostic-intent.json")
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["parent"] == PARENT and intent["node"] == NODE
            and intent["port"] == PORT and intent["source_manifest_SHA256"] == MANIFEST
            and intent["planned_forwards"] == 32 and intent["planned_optimizer_updates"] == 0,
            "original diagnostic intent differs")
    worker = document("worker-assigned.json")
    before = document("pre-device-inventory.json")
    report["independent_post_errors"] = []
    try:
        require(post[2:4] == ["COMPLETED", "0:0"], "independent post step was unsuccessful")
        after = document("post-device-inventory.json")
        validate_devices(before, after, worker, post, epoch)
        report["clean_independent_post"] = True
    except (OSError, KeyError, TypeError, ValueError) as error:
        report["independent_post_errors"].append(str(error)[:150])
    fresh, raw_pins, rejected = compare(run, original)
    # Each arm preserves its complete owned-child registry. Use the same
    # two-MiB bound that the original HOST accepted for this specific file.
    attempts = document("worker-outcome.json", 2 * 1024**2)
    require(attempts["request"] == REQUEST and attempts["epoch"] == epoch
            and attempts["planned_arms"] == list(ARMS) and attempts["planned_forwards"] == 32
            and attempts["planned_optimizer_updates"] == 0, "worker outcome changed scope")
    observers = attempts["arms"]
    require([row["arm"] for row in observers] == list(ARMS)[:len(observers)], "arm execution order differs")
    report["comparison"] = {k: fresh[k] for k in ("actual_step", "planned_forwards", "verified_forwards",
                                                 "missing_forwards", "comparable_ranks", "complete", "rows", "errors")}
    report["rejected_native_receipts"] = rejected
    report["actual_optimizer_updates"] = 0 if fresh["complete"] else None
    report["native_metadata_members"] = len(raw_pins)
    report["native_metadata_set_SHA256"] = sha256(json.dumps(raw_pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["file_SHA256"] = pins
    original_summary = run / "sdpa-repeatability-comparison.json"
    if original_summary.is_file():
        prior = document("sdpa-repeatability-comparison.json")
        report["original_worker_comparison_agrees"] = prior == dict(
            fresh, native_state_receipts_SHA256=raw_pins, missing_or_invalid_receipt_files=rejected)
    host_result = document("diagnostic-outcome.json")
    require(host_result["request"] == REQUEST and host_result["epoch"] == epoch, "host outcome identity differs")
    report["original_host_outcome"] = {name: host_result.get(name) for name in ("complete", "clean_post", "error")}
    report["diagnosis_complete"] = (rows["gpu"][2:4] == ["COMPLETED", "0:0"]
        and report["clean_independent_post"] and report["original_worker_comparison_agrees"]
        and host_result["complete"] is True and attempts["passed"] is True
        and fresh["complete"] is True and fresh["verified_forwards"] == 32 and len(raw_pins) == 96
        and len(observers) == 4 and all(row["cleanup_known"] is True and row["passed"] is True
                                       and row["cleanup"]["owned_processes_remaining"] == 0 for row in observers))
    return report


def main():
    columns = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT,
                                   "-S", "2026-11-24T00:00:00", "-o", columns], text=True, timeout=9)
    require(0 < len(raw) <= 1048576, "accounting exceeds expected bounds")
    rows = select_rows(raw)
    result = inspect(RUN, ORIGINAL, rows)
    result["accounting_SHA256"] = sha256(raw.encode()).hexdigest()
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
