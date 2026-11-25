"""Ingress for the ordinary frozen common CLI, with no first-step replay substitution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from assigned_kfd import assigned_kfd
from contract import (Budget, CACHE, FRAMEWORK, FRAMEWORK_MANIFEST, LABEL, LAUNCH_SHA,
                      MODEL, NATIVE_ALIAS, NODE, PARENT, REQUEST, RUN, WORK, load, need, read,
                      save, validate_assets, validate_preflight, verify_package)
from framework_binding import verify_framework
from runtime_pci import runtime_pci
from placement import numeric_cgroup
from flash_compat import validate_execution_compatibility

RUN = Path(RUN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    budget = Budget(args.epoch)
    need(budget.left(WORK) > 0, "Original work deadline exhausted")
    verify_package(RUN / "src", args.source_manifest_sha256)
    preflight, pfpin = load(RUN / "preflight.json")
    need(pfpin["SHA256"] == args.preflight_sha256, "Preflight SHA differs")
    validate_preflight(preflight, args.epoch)
    assets = validate_assets(preflight, execution_mount=True)
    framework = verify_framework(Path(FRAMEWORK), Path(FRAMEWORK_MANIFEST))
    _, timing_source=read(FRAMEWORK/'generation/src/gencuda/backends/hunyuanvideo_training_walltime.py',1<<20)
    need(timing_source['SHA256']=='2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae','T09_EXACT_UNPROFILED_CLOCK_BACKEND')
    compatibility = validate_execution_compatibility()
    save(RUN / "compatibility-ingress.json", compatibility)
    node = socket.gethostname().split(".", 1)[0]
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    step = os.environ.get("SLURM_STEP_ID", "")
    need(node == NODE and os.environ.get("SLURM_JOB_ID") == PARENT
         and re.fullmatch(r"0|[1-9][0-9]*", selector) and step.isdecimal(),
         "Expected the exact assigned one-DCU Slurm scope")
    gate, gatepin = load(RUN / "gpu-gate" / (NODE + ".json"))
    need(gate["step"] == PARENT + "." + step and gate["node"] == NODE
         and gate["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
         and gate["kfd_holder_count"] == 0
         and 0 <= time.monotonic_ns() - gate["monotonic_ns"] <= 90_000_000_000,
         "Same-step fresh host KFD gate differs")
    need(numeric_cgroup()==gate["step"] and gate["actual_numeric_cgroup"] is True
         and gate["actual_live_TRES_verified"] is True,"T09_NATIVE_ACTUAL_ONE_DCU_CGROUP")
    pci = runtime_pci()  # Explicit actual HIP mapping, never a CUDA hardware assumption.
    need(pci in gate["pci"], "Runtime PCI differs from actual host inventory")
    device = assigned_kfd(pci)
    save(RUN / "assigned-device.json", {"step": PARENT + "." + step, "request_id": REQUEST,
         "node": node, "selector": selector, "boot_id": gate["boot_id"], "device": device,
         "host_gate": gatepin, "framework": framework, "assets": assets,
         "compatibility": compatibility})
    launch, _ = read(RUN / "src/ORIGINAL_LAUNCH.json")
    need(hashlib.sha256(launch).hexdigest() == LAUNCH_SHA, "Original common launch differs")
    original = json.loads(launch)
    mapping = {original["weights"]: str(MODEL), original["source"]: str(NATIVE_ALIAS),
               original["config"]["training_cache"]: str(CACHE)}
    save(RUN / "assets.json", mapping)
    need(not (RUN / "outputs").exists(), "Native output namespace already exists")
    argv = [sys.executable, "-B", str(FRAMEWORK / "tools/hygon_training_recipe.py"), "T09.sft",
            "--a100-launch", str(RUN / "src/ORIGINAL_LAUNCH.json"),
            "--asset-map", str(RUN / "assets.json"), "--run", "--cards", selector,
            "--source-root", str(NATIVE_ALIAS.parent), "--output-root", str(RUN / "outputs"),
            "--run-name", LABEL, "--lock-root", str(RUN / "locks"), "--lock-host", node]
    save(RUN / "native-ingress.json", {"request_id": REQUEST, "epoch": args.epoch,
         "deadline": args.epoch + 1440, "argv": argv, "preflight": pfpin,
         "source_manifest_SHA256": args.source_manifest_sha256, "planned_updates": 25,
         "planned_resume_updates": 0, "original_first_update_replay": False})
    need(budget.left(WORK) > 0, "Original work deadline exhausted before native exec")
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
