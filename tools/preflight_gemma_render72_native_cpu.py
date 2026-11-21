"""Check Gemma Render72 stage, exact CPU inputs and Slurm ownership before launch."""

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from types import SimpleNamespace

import audit_gemma_training_native as worker
import control_gemma_render72_native_cpu as controller
import stage_gemma_render72_target_data as stage
from read_training_slurm_membership import snapshot


SOURCE = "tools/audit_gemma_training_native.py"
BUILD_VIEW = "tools/build_gemma_train_exposure.py"
PROCESSOR_MANIFEST = "code/bw1000/training-gemma-native-inputs-HIP-CPU-v1/MODEL_PROCESSOR_FILES.json"
FRAMEWORK_MANIFEST = "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json"
READ_BUDGET_SECONDS = 60


def assets(framework, check_deadline):
    check_deadline()
    controller.require(framework.resolve(strict=True) == framework, "Framework root is aliased")
    for host, _, _ in controller.bind_map(framework)[2:]:
        check_deadline()
        mode = host.lstat().st_mode
        controller.require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Target bind missing or aliased")
    controller.require(Path(controller.APPTAINER).is_file() and os.access(controller.APPTAINER, os.X_OK)
                       and (controller.TARGET / "containers/dtk26-sandbox-r1").is_dir(),
                       "Pinned Apptainer or container unavailable")
    for name in (SOURCE, BUILD_VIEW):
        controller.pinned(controller.REPO / name, controller.PINS[name])
    args = SimpleNamespace(prepared=controller.DATA, view=controller.VIEW,
                           processor_root=controller.PROCESSOR,
                           processor_manifest=controller.REPO / PROCESSOR_MANIFEST,
                           framework_root=framework, framework_manifest=controller.REPO / FRAMEWORK_MANIFEST)
    lines = worker.verify_assets(args, check_deadline)
    controller.require(len(lines) == controller.PLANNED["unique_rows"], "Original training row count differs")
    return dict(training_rows=len(lines), exposure_rows=800, training_media=12, processor_files=14,
                observer_source_verified=True, runtime_paths_present=True, native_encode_calls=0)


def ownership(query, observer_uid):
    raw = controller.pinned(controller.REPO / controller.ROSTER, controller.PINS[controller.ROSTER])
    group_rows = json.loads(raw)["groups"]
    parent, node = controller.PARENT, controller.NODE
    groups = {parent: {name: row["allocations"][parent] for name, row in group_rows.items()}}
    account = snapshot(groups, owner_uid=observer_uid, run_query=lambda *args: query(*args).splitlines())
    before = account["parents"][parent]
    return (before["owned_by_observer"] is True and before["ledger_exact"] is True
            and before["step_locations_exact"] is True and before["state"] == "RUNNING"
            and before["allocated_DCU"] == 320 and node in groups[parent]["T02"]
            and all(node not in row["member_nodes"] for row in before["active_steps"].values()))


def inspect(framework, *, stage_status, final_status, cpu_slot, check_deadline, check_assets=assets):
    check_deadline()
    original = stage_status()
    report = dict(request=controller.REQUEST, parent=controller.PARENT, node=controller.NODE,
                  cpu_resources=dict(cpus=4, memory_MiB=8192, GRES=0),
                  original=original, state=original["state"], actual_Slurm_submissions=0,
                  optimizer_updates=0, training_READY=False)
    if original["state"] != "STAGING_VERIFIED_CPU_PREREQUISITES_PENDING":
        return report

    verified = check_assets(framework, check_deadline)
    check_deadline()
    if not cpu_slot():
        return dict(report, state="CPU_SLOT_UNCONFIRMED_OR_OCCUPIED", assets=verified)

    final = final_status()
    if not final["stage_namespace_present"] or final["cpu_namespace_present"] or final["cpu_accounting_row"]:
        return dict(report, state="RECONCILE_ORIGINAL", final_original=final, assets=verified)
    check_deadline()
    return dict(report, state="TARGET_ASSETS_VERIFIED_CPU_DISPATCH_PENDING", final_original=final,
                assets=verified, framework_root=str(framework))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", required=True)
    args = parser.parse_args()
    controller.require(os.name == "posix", "Target HOST required")
    framework = Path(controller.target_scope_path(args.framework_root))
    start = time.monotonic()

    def expired(*_):
        raise TimeoutError("Gemma read-only CPU preflight deadline")

    def check_deadline():
        if time.monotonic() - start >= READ_BUDGET_SECONDS:
            expired()

    def query(*argv):
        left = READ_BUDGET_SECONDS - (time.monotonic() - start)
        if left <= 1:
            expired()
        return subprocess.check_output(argv, text=True, timeout=min(left - 1, 8)).strip()

    def stage_observation():
        return stage.observe(stage.RESULTS, query, os.getuid())

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, READ_BUDGET_SECONDS)
    try:
        inputs, _ = stage.published_package()
        report = inspect(framework, stage_status=lambda: stage.inspect(stage.RESULTS, stage_observation, inputs),
                         final_status=stage_observation, cpu_slot=lambda: ownership(query, os.getuid()),
                         check_deadline=check_deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(report["state"] != "TARGET_ASSETS_VERIFIED_CPU_DISPATCH_PENDING")


if __name__ == "__main__":
    raise SystemExit(main())
