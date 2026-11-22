"""Preflight, start once, or read the adopted Native v2 scope."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import subprocess

import kernel_855
import worker


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL, REQUEST = worker.LABEL, worker.REQUEST
CLAIM = HERE / "SCOPE.json"
STAGE_MANIFEST = HERE / "STAGE_FILES.json"
PLACEMENT = dict(parent="1103606", node="f12r2n13", zone="O01", cpus=4, memory_MiB=16384, DCU=0)
PLANNED = worker.PLANNED
EXPLICIT_HOST = HERE.parent / "fresh-kv-explicit-ar-native-host-v1/host_once.py"
EXPLICIT_HOST_SHA = "9c13c16e82c5384caed9bf982c7d11725dab22981e52247c24157bc03ae62146"
MEMBERSHIP = "tools/read_training_slurm_membership.py"
ACCOUNTING = "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py"
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
ORIGIN_BUILD = "results/GLM474-depth-stratified-observer-CPU-v1/build-result.json"
ORIGIN_BUILD_SHA = "aa669c0ebc18072b0085c1f98fe34bee898e50af28f6e0542bb8c485ac200b6f"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bytes_sha(raw):
    return sha256(raw).hexdigest()


def validate_package(claim_sha, manifest_sha):
    require(all(re.fullmatch(r"[0-9a-f]{64}", pin) for pin in (claim_sha, manifest_sha)),
            "exact published checkout pins required")
    claim = json.loads(kernel_855.read(CLAIM, claim_sha, limit=16384))
    need = dict(label=LABEL, request=REQUEST, placement=PLACEMENT, planned=PLANNED,
                kernel_prerequisite_request=kernel_855.REQUEST, kernel_object_SHA256=kernel_855.OBJECT_SHA,
                original_CPU308="1103606.308", whole_seconds=1800, work_seconds=1680,
                CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False)
    require(all(type(claim.get(key)) is type(value) and claim[key] == value for key, value in need.items()),
            "adopted Native v2 claim differs")
    files = json.loads(kernel_855.read(STAGE_MANIFEST, manifest_sha, limit=32768))["files_SHA256"]
    require(files[CLAIM.relative_to(REPO).as_posix()] == claim_sha and len(files) == 32,
            "reviewed Native checkout file set differs")
    for name, digest in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unexpected staged member")
        kernel_855.read(REPO.joinpath(*relative.parts), digest)
    return files


def originals(publication, claim_sha):
    kernel_855.read(EXPLICIT_HOST, EXPLICIT_HOST_SHA)
    old = runpy.run_path(str(EXPLICIT_HOST))["controller"]()
    old.update(ROOT=ROOT, REPO=REPO, LABEL=LABEL, REQUEST=REQUEST, CLAIM_PIN=claim_sha,
               WORKER_PUBLICATION=publication, KERNEL_REQUEST=kernel_855.REQUEST, KERNEL_LABEL=kernel_855.LABEL,
               WORKER=(HERE / "worker.py").relative_to(REPO), CHECKOUT=f"/refactor/incoming/{LABEL}-checkout",
               PROOF=f"/refactor/results/{LABEL}-controller/kernel-reconciliation.json")
    read_visible = old["verify_terminal"]

    def verified_when_visible(*args):
        try:
            return read_visible(*args)
        except FileNotFoundError:
            # The independent compute reader owns the missing-login-mount boundary.
            return {}

    old["verify_terminal"] = verified_when_visible
    return old


def accounting_rows(timeout=8):
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", "1087656,1103606", "-S", "2026-11-20T00:00:00",
                                   "-o", "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"],
                                  timeout=min(8, timeout))
    require(0 < len(raw) <= 2**21, "original accounting output absent or too large")
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    require(all(len(row) == len(kernel_855.COLUMNS) for row in rows)
            and {"1087656", "1103606"} <= {row[0] for row in rows}, "full two-parent accounting differs")
    return rows


def execute(mode, publication, claim_sha, manifest_sha, *, read_rows=None, observe=None, expires=None,
            runner=None, wait=None):
    require(re.fullmatch(r"[0-9a-f]{40}", publication) and
            REPO == ROOT / "incoming" / (LABEL + "-reviewed-checkout") and REPO.resolve(strict=True) == REPO,
            "new canonical reviewed checkout and verified publication required")
    files = validate_package(claim_sha, manifest_sha)
    host = originals(publication, claim_sha)
    member = runpy.run_path(str(REPO / MEMBERSHIP))
    schedule = runpy.run_path(str(REPO / ACCOUNTING))
    read_rows = read_rows or accounting_rows

    def current_native(cap):
        rows = [row for row in read_rows(cap) if row[1] == LABEL]
        require(len(rows) <= 1, "ambiguous exact Native label")
        return host["accounting_row"]("\n".join("|".join(row) for row in rows).encode())

    run_path = ROOT / "results" / (LABEL + "-controller")
    output = ROOT / "candidates" / LABEL
    if mode == "reconcile":
        return host["inspect_original"](run_path, output, current_native, schedule["is_terminal"])
    require(all(not path.exists() and not path.is_symlink() for path in (run_path, output)),
            "existing v2 execution scope must be reconciled")
    groups = json.loads((REPO / LEDGER).read_bytes())["groups"]
    parents = {parent: {zone: group["allocations"][parent] for zone, group in groups.items()}
               for parent in ("1087656", "1103606")}
    observe = observe or (lambda: member["snapshot"](parents, owner_uid=os.getuid()))
    # The original build recipe is visible on the login mount. The complete .855
    # object and tool/depfile recheck belongs to the single compute worker.
    original_build = json.loads(kernel_855.read(ROOT / ORIGIN_BUILD, ORIGIN_BUILD_SHA, size=431138))
    require(original_build["step"] == "1103606.308" and original_build["passed"] is True,
            "CPU308 original recipe differs")

    def bind_kernel(_claim=None, *, reconcile=True):
        require(reconcile is True, "kernel verification cannot start any work")
        return kernel_855.login_proof(ROOT, REPO, read_rows)

    if mode == "preflight":
        proof, _ = host["kernel_gate"](bind_kernel, observe, host["time"].time)
        require(current_native(8) is None, "v2 target label already present")
        return dict(state="V2_NATIVE_PRECHECKED", request=REQUEST, kernel_step=proof["row"][0],
                    kernel_object_SHA256=proof["object_SHA256"], reviewed_members=len(files),
                    claim_SHA256=claim_sha, stage_manifest_SHA256=manifest_sha, actual_new_Slurm_steps=0)
    require(mode == "launch", "unknown mode")
    return host["launch"](run_path, output, REPO, reconcile_kernel=bind_kernel, observe=observe,
                          expires=expires or (lambda: host["parent_expiry"](member["fields"])),
                          accounting=current_native, runner=runner or subprocess.run,
                          wait_terminal=wait or schedule["wait_terminal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--stage-manifest-sha256", required=True)
    parser.add_argument("--publication", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "launch", "reconcile"):
        group.add_argument("--" + mode, dest="mode", action="store_const", const=mode)
    args = parser.parse_args()
    report = execute(args.mode, args.publication, args.claim_sha256, args.stage_manifest_sha256)
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(report["state"] not in ("V2_NATIVE_PRECHECKED", "NATIVE_VERIFIED"))


if __name__ == "__main__":
    raise SystemExit(main())
