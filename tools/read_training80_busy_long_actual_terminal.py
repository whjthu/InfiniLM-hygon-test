"""Reconcile the original fixed-hour observer and independently re-read its samples."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INSTALL_REQUEST = "d32e877d95e0420c94b7f0ab209732ce"
RUN_REQUEST = "0e1a294fabfc4854854c09ed6b4d6f83"
ORIGINAL_PINS = "training80-sync-after-rollover-12a31891d2544ae6a1fa7d13829642f1"
STEPS = {"1103606": "1103606.3088", "1194900": "1194900.1182"}
CONTROLLER = {"controller_PID": 3614429, "controller_start_ticks": 905090997}
SOURCE_PINS = {
    "run_training80_busy_long_host.py": "ef1262903a59046c2c57cb1b7a93d39147858551305b88d2f1d2a6f31676a099",
    "collect_training80_busy_pulses.py": "37f1e205a7eac52914978587534f41f9cd60fa6e71650e0ddd628ae80557efaa",
    "collect_training80_busy_long_window.py": "f3e47d96a52dcf7b4596707ffea671c4d8ff7c7511ab860c242dc8bf51c25e22",
    "check_training80_point_clock.py": "0997f63efe318fb310d1e77d21021ebd4dafe753e57670952d319a343263a79f",
    "sample_training80_busy.py": "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324",
    "training_dispatch_allocation.py": "e603bbbb1ef6dd539d667d547675022c2a5236fba2d7bf873dbff44950ed80b6",
    "verify_training80_busy_long_window.py": "d81057156631e0dbbfb7fb5b458b58ae6b957ed52eacd709459d186d764f04d8",
}
HOST_INSTALL_SHA = "fdaf7cd6990707735315d9d1b22e9c389a10855531866f04e7a81eecadadeab6"
READY_SHA = "44e1d5737c57f7f126993971da76d37c6c1d80103ca52339cdb148fb5d2cab63"
TOTAL_DEVICE_SECONDS = 2_304_000
TOTAL_SAMPLES = 1_152_640
THRESHOLD_PERCENT = 95


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin_file(path, limit, expected=None):
    require(path.is_file() and path.resolve(strict=True) == path.absolute(), "EXACT_ORIGINAL_FILE")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    digest = sha256(data).hexdigest()
    require(0 < len(data) <= limit and (expected is None or digest == expected), "BOUNDED_ORIGINAL_BYTES")
    return data, digest


def original_json(path, limit=65_536, expected=None):
    data, pin = pin_file(path, limit, expected)
    value = json.loads(data)
    require(type(value) is dict, "EXPECTED_RECEIPT_OBJECT")
    return value, pin


def decide(measurement, slurm_c0, releases, host_c0):
    expected, qualified, unknown = (measurement[key] for key in (
        "expected_device_seconds", "qualified_device_seconds", "unknown_device_seconds"))
    count, valid, missing = (measurement[key] for key in ("expected_samples", "valid_samples", "missing_samples"))
    require(expected == TOTAL_DEVICE_SECONDS and 0 <= qualified <= expected and qualified + unknown == expected
            and count == TOTAL_SAMPLES and 0 <= valid <= count and valid + missing == count, "ORIGINAL_HOUR_DENOMINATORS")
    average = measurement["full_hour_interval_busy_percent"]
    if measurement["hour_interval_coverage_complete"]:
        require(unknown == missing == 0 and type(average) in (int, float) and 0 <= average <= 100,
                "COMPLETE_INTERVAL_AND_BUSY_RANGE")
    else:
        require(average is None, "INCOMPLETE_HOUR_CANNOT_CLAIM_FULL_MEAN")
    accepted = measurement["hour_interval_coverage_complete"] and slurm_c0 and releases == STEPS and host_c0
    return {"complete_verified_hour": accepted, "threshold_percent": THRESHOLD_PERCENT,
            "accepted_full_hour_busy_percent": average if accepted else None,
            "threshold_met": average >= THRESHOLD_PERCENT if accepted else None}


def inspect(project=PROJECT):
    stage = project / "incoming" / ("training80-long-busy-host-" + INSTALL_REQUEST)
    allocation = project / "incoming" / ORIGINAL_PINS
    output = project / "results" / ("training80-long-busy-" + RUN_REQUEST)
    install, install_sha = original_json(stage / "INSTALL_RESULT.json", 16_384, HOST_INSTALL_SHA)
    require(install["request"] == INSTALL_REQUEST and install["source_SHA256"] == SOURCE_PINS, "ACTUAL_INSTALL_SCOPE")
    for name, digest in SOURCE_PINS.items():
        pin_file(stage / name, 32_768, digest)
    sys.path.insert(0, str(stage))
    import collect_training80_busy_pulses as slurm
    import run_training80_busy_long_host as controller
    import verify_training80_busy_long_window as reader
    ready, ready_sha = original_json(output / "ready.json", 16_384, READY_SHA)
    require(ready["request"] == RUN_REQUEST and ready["actual_READY_steps"] == STEPS and ready["ready_nodes"] == 80
            and all(ready[k] == v for k, v in CONTROLLER.items()), "EXACT_FIRST_READY")
    owners, pins, expected = controller.pinned_paths(allocation / "ledger.json", allocation / "pci.json")
    labels = {parent: f"training80-long-busy-{RUN_REQUEST}-{parent}" for parent in STEPS}
    deadline = time.monotonic_ns() + 35_000_000_000
    rows = {parent: slurm.account(parent, labels[parent], deadline, step) for parent, step in STEPS.items()}
    local_pid_match = controller.proc_start_ticks(CONTROLLER["controller_PID"]) == CONTROLLER["controller_start_ticks"]
    report = {"request": RUN_REQUEST, "accounting": rows, "controller_pid_tick_match_on_read_host": local_pid_match,
              "ready_SHA256": ready_sha, "install_SHA256": install_sha, "replay_allowed": False}
    terminal_path = output / "host-outcome.json"
    if not slurm.terminal(rows) or not terminal_path.is_file():
        return dict(report, state="WAIT_FOR_EXACT_ORIGINAL_TERMINAL", complete_verified_hour=False)
    host, outcome_sha = original_json(terminal_path)
    require(host["request"] == RUN_REQUEST and host["worker_steps"] == STEPS and host["GPU_requested"] == 0
            and all(host[k] == v for k, v in CONTROLLER.items()), "ACTUAL_PRODUCER_TERMINAL_IDENTITY")
    releases = {parent: step for parent, step in STEPS.items()
                if controller.released_step(parent, step, labels[parent], expected[parent], deadline)}
    slurm_c0 = slurm.qualified_accounting(rows, STEPS, expected, lambda nodes: slurm.hostnames(nodes, deadline))
    observed = reader.verify(output / "measurement", owners, pins, RUN_REQUEST, controller.PCI_SHA)
    result = decide(observed, slurm_c0, releases, host["full_hour_observer_C0"])
    return dict(report, state="ACTUAL_TERMINAL_AND_INDEPENDENT_MEASUREMENT",
                original_host_outcome_SHA256=outcome_sha, independent_measurement=observed,
                actual_zero_GRES_C0=slurm_c0, released_steps=releases, **result)


if __name__ == "__main__":
    require(os.name == "posix" and sys.flags.dont_write_bytecode and sys.flags.isolated and sys.flags.no_site,
            "READ_ONLY_ISOLATED_HOST")
    print(json.dumps(inspect(), sort_keys=True, allow_nan=False))
