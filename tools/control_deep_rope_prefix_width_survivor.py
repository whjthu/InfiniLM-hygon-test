"""Run the frozen RoPE width pair on one verified node of the surviving allocation."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import control_deep_rope_prefix_width_pair as pair
import control_deep_inverse_rope_compile as original
from read_training_slurm_membership import snapshot
from run_deep_rope_prefix_width_compile_pair import host_step_command, read, require
from training_dispatch_allocation import SURVIVOR_KIND, SURVIVOR_LEDGER, allocation


ROOT = pair.ROOT
REPO = pair.REPO
PARENT, NODE, ZONE = "1103606", "f12r2n13", "O01"
LABEL = pair.PAIR + "-survivor-n13"
CLAIM = REPO / "coordination/tasks/deep-rope-prefix-width-pair-survivor-n13.json"
OLD_CLAIM_SHA = "f197aa27567f91bb08de947fcbc7c367da66f9ea059eddc918f9a963e24123b1"
PCI_SHA = "d736840ad2f4766d3a4ce95e0a76bd6c2d213deeeed074ae44209370958c91b3"


def check_membership(groups):
    membership = snapshot(groups, owner_uid=os.getuid())
    row = membership["parents"][PARENT]["coverage_by_group_and_node"][ZONE][NODE]
    require(membership["membership_qualified"] and membership["distinct_nodes"] == 40
            and membership["allocated_DCU"] == 320 and row["state"] == "NO_NUMBERED_STEP"
            and row["numbered_step_assigned_DCU_bounds"] == [0, 0], "Surviving O01 node unavailable")


def verify_device(group_map, probe_request):
    check_membership(group_map)
    # The published worker checks PCI, all KFD holders, and this user's open device FDs.
    args = [
        "srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", "--mem=256M", "--gres=none", "--nodelist=" + NODE, "--time=1",
        "--job-name=deep-rope-width-admit-" + probe_request, sys.executable, "-B",
        str(REPO / "tools/collect_training80_dispatch_snapshot.py"), "node", probe_request, PARENT, "[]",
    ]
    raw = subprocess.check_output(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12)
    require(len(raw) <= 8192, "Device receipt exceeds its bound")
    row = json.loads(raw)
    pci = row.get("PCIs", [])
    require(row["request_id"] == probe_request and row["node"] == NODE and row["parent"] == PARENT
            and row["observer_uid"] == os.getuid() and row["step"].startswith(PARENT + ".")
            and row["step"].split(".")[1].isdecimal() and row["ports_checked"] == []
            and "probe_error" not in row and len(pci) == len(set(pci)) == 8
            and sha256(json.dumps(sorted(pci), separators=(",", ":")).encode()).hexdigest() == PCI_SHA
            and (row["kfd_process_count"], row["own_device_fd_process_count"]) == (0, 0),
            "O01 device identity or process clearance differs")
    check_membership(group_map)
    return {"node": NODE, "step": row["step"], "pci_SHA256": PCI_SHA,
            "kfd_processes": 0, "owned_device_fds": 0, "result_SHA256": sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--admission-sha256")
    options = parser.parse_args()
    claim = json.loads(read(CLAIM, 8192, options.claim_sha256))
    pair.check_scope(REPO, OLD_CLAIM_SHA)
    require(claim["parent"] == PARENT and claim["node"] == NODE and claim["label"] == LABEL
            and claim["CPU_GO"] is True and claim["GPU_GO"] is claim["replay_allowed"] is False
            and claim["wall_seconds"] == 300 and claim["device_TUs"] == 2,
            "Replacement claim differs")
    for name, digest in claim["new_host_files_SHA256"].items():
        read(REPO / name, 128 * 1024, digest)
    kind, groups, owners = allocation(SURVIVOR_LEDGER.read_bytes())
    require(kind == SURVIVOR_KIND and owners[NODE] == (PARENT, ZONE), "Replacement is outside O01")
    old = ROOT / "results" / (pair.PAIR + "-controller")
    run_path = ROOT / "results" / LABEL
    output = ROOT / "candidates" / pair.PAIR
    require(not old.exists() and not old.is_symlink() and not output.exists() and not output.is_symlink(),
            "Original scope exists; reconcile it before a new step")
    pair.prelaunch()
    original.LABEL = pair.PAIR
    original.REQUEST = claim["request"]
    original.NODE = pair.NODE = NODE
    original.MANIFEST = pair.WORKLOAD
    original.command = lambda epoch: host_step_command(REPO.as_posix(), PARENT, NODE, epoch)
    original.check_result = pair.check_result
    require(original.accounting(8) is None, "An original compile step already exists")
    helper = REPO / original.ACCOUNTING
    env = {"__name__": "frozen_terminal_accounting"}
    exec(compile(read(helper, 8192, pair.HOST_FILES[original.ACCOUNTING.as_posix()]), str(helper), "exec"), env)
    admission_path = REPO / "compute-admission.json"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if options.probe_only:
        device = verify_device(groups, uuid.uuid4().hex)
        original.save_once(admission_path, {
            "request": claim["request"], "controller_boot": boot, "observed_monotonic": time.monotonic(),
            "device_admission": device,
        })
        print(json.dumps({"request": claim["request"], "device_admission": device,
                          "admission_SHA256": sha256(admission_path.read_bytes()).hexdigest()}), flush=True)
        return 0
    admission = json.loads(read(admission_path, 8192, options.admission_sha256))

    def check_reserved_node():
        age = time.monotonic() - admission["observed_monotonic"]
        require(admission["request"] == claim["request"] and admission["controller_boot"] == boot
                and 0 <= age < 55, "Original host admission expired or changed")
        check_membership(groups)

    outcome = original.controller(run_path, output, options.claim_sha256, env["wait_terminal"],
                                  observe_slot=check_reserved_node)
    fields = ("request", "state", "row", "object_SHA256", "receipt_SHA256", "terminal_visible")
    print(json.dumps({key: outcome.get(key) for key in fields}, allow_nan=False), flush=True)
    return int(outcome["state"] != "COMPILE_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
