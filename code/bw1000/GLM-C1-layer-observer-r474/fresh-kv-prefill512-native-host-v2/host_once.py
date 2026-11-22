"""Preflight, submit once, or reconcile the new zero-GRES T512 Native step."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL, REQUEST = worker.LABEL, worker.REQUEST
CLAIM = HERE / "SCOPE.json"
STAGE_MANIFEST = HERE / "STAGE_FILES.json"
ACCOUNTING = "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py"
MEMBERSHIP = "tools/read_training_slurm_membership.py"
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
EXPLICIT_HOST = HERE.parent / "fresh-kv-explicit-ar-native-host-v1" / "host_once.py"
EXPLICIT_HOST_SHA = "9c13c16e82c5384caed9bf982c7d11725dab22981e52247c24157bc03ae62146"
ORIGIN_BUILD = "results/GLM474-depth-stratified-observer-CPU-v1/build-result.json"
ORIGIN_BUILD_SHA = "aa669c0ebc18072b0085c1f98fe34bee898e50af28f6e0542bb8c485ac200b6f"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def validate_package(claim_sha, manifest_sha, kernel):
    need(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (claim_sha, manifest_sha)),
         "exact reviewed claim and stage identities required")
    claim = json.loads(worker.checked(CLAIM, claim_sha, 16384))
    worker.require_request(claim["request"])
    exact = dict(label=LABEL, request=REQUEST, placement=worker.PLACEMENT, planned=worker.PLANNED,
                 original_CPU308="1103606.308", original_918_library_SHA256=worker.ORIGINAL_NATIVE_SHA,
                 original_918_archive_SHA256=worker.ORIGINAL_ARCHIVE_SHA, kernel_prerequisite_request=kernel.REQUEST,
                 failed_1068_evidence_SHA256=worker.PREDECESSOR_FAILURE_SHA,
                 kernel_object_SHA256=kernel.OBJECT_SHA, caller_publication=worker.CALLER_PUBLICATION,
                 caller_manifest_SHA256=worker.CALLER_MANIFEST, new_export=worker.NEW_EXPORT,
                 expected_exports=285, expected_providers=71, whole_seconds=1800, work_seconds=1680, CPU_GO=True,
                 GPU_GO=False, Model_GO=False, replay_allowed=False)
    need(all(type(claim.get(k)) is type(v) and claim[k] == v for k, v in exact.items()), "new T512 reviewed claim differs")
    index = json.loads(worker.checked(STAGE_MANIFEST, manifest_sha, 32768))["files_SHA256"]
    need(len(index) == claim["stage_members"] and index[CLAIM.relative_to(REPO).as_posix()] == claim_sha,
         "reviewed target file inventory differs")
    for member, pin in index.items():
        relative = PurePosixPath(member)
        need(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == member,
             "stage manifest escapes the reviewed checkout")
        worker.checked(REPO.joinpath(*relative.parts), pin)
    worker.caller_sources()
    return index


def original_controller(publication, claim_sha, kernel):
    text = worker.checked(EXPLICIT_HOST, EXPLICIT_HOST_SHA).decode()
    text = worker.once(text, """'linked["strong_exports"] == 284',""", """'linked["strong_exports"] == 285',""")
    original = worker.execute(EXPLICIT_HOST, text)["controller"]()
    original.update(ROOT=ROOT, REPO=REPO, LABEL=LABEL, REQUEST=REQUEST, CLAIM_PIN=claim_sha,
                    WORKER_PUBLICATION=publication, KERNEL_REQUEST=kernel.REQUEST, KERNEL_LABEL=kernel.LABEL,
                    WORKER=(HERE / "worker.py").relative_to(REPO), CHECKOUT=f"/refactor/incoming/{LABEL}-checkout",
                    PROOF=f"/refactor/results/{LABEL}-controller/kernel-reconciliation.json")
    visible = original["verify_terminal"]

    def verified_when_visible(*args):
        try:
            return visible(*args)
        except FileNotFoundError:
            return {}

    original["verify_terminal"] = verified_when_visible
    return original


def original_rows(timeout=8):
    query = (
        "sacct", "-n", "-P", "-j", "1087656,1103606", "-S", "2026-11-20T00:00:00",
        "-o", "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES",
    )
    raw = subprocess.check_output(query, timeout=min(8, timeout))
    need(0 < len(raw) <= 2**21, "independent accounting absent or too large")
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    need(all(len(row) == 7 for row in rows)
         and {"1087656", "1103606"} <= {row[0] for row in rows}, "incomplete two-parent Slurm accounting")
    return rows


def execute(mode, publication, claim_sha, manifest_sha, *, read_rows=None, observe=None, expires=None, runner=None, wait=None):
    need(re.fullmatch(r"[0-9a-f]{40}", publication)
         and REPO == ROOT / "incoming" / (LABEL + "-reviewed-checkout") and REPO.resolve(strict=True) == REPO,
         "canonical T512 reviewed checkout and exact ordinary publication required")
    kernel, _ = worker.baseline()
    files = validate_package(claim_sha, manifest_sha, kernel)
    host = original_controller(publication, claim_sha, kernel)
    member = runpy.run_path(str(REPO / MEMBERSHIP))
    accounting = runpy.run_path(str(REPO / ACCOUNTING))
    read_rows = read_rows or original_rows

    def current(cap):
        matches = [row for row in read_rows(cap) if row[1] == LABEL]
        need(len(matches) <= 1, "ambiguous new T512 accounting label")
        return host["accounting_row"]("\n".join("|".join(row) for row in matches).encode())

    run = ROOT / "results" / (LABEL + "-controller")
    output = ROOT / "candidates" / LABEL
    if mode == "reconcile":
        return host["inspect_original"](run, output, current, accounting["is_terminal"])
    need(not any(path.exists() or path.is_symlink() for path in (run, output)), "existing T512 scope must be reconciled")
    actual = json.loads(worker.checked(ROOT / ORIGIN_BUILD, ORIGIN_BUILD_SHA))
    need(actual["step"] == "1103606.308" and actual["passed"] is True, "actual CPU308 original recipe differs")
    worker.verify_original_archive(read_rows)
    worker.verify_failed_predecessor(read_rows)
    groups = json.loads((REPO / LEDGER).read_bytes())["groups"]
    parents = {parent: {zone: group["allocations"][parent] for zone, group in groups.items()}
               for parent in ("1087656", "1103606")}
    observe = observe or (lambda: member["snapshot"](parents, owner_uid=os.getuid()))

    def bind_kernel(_claim=None, *, reconcile=True):
        need(reconcile is True, "original .855 can only be checked")
        return kernel.login_proof(ROOT, REPO, read_rows)

    if mode == "preflight":
        proof, _ = host["kernel_gate"](bind_kernel, observe, host["time"].time)
        need(current(8) is None, "target T512 label already exists")
        return dict(state="T512_NATIVE_PRECHECKED", request=REQUEST, kernel_step=proof["row"][0],
                    original_native_step="1103606.918", reviewed_members=len(files), claim_SHA256=claim_sha,
                    stage_manifest_SHA256=manifest_sha, actual_new_Slurm_steps=0)
    need(mode == "launch", "unknown HOST mode")
    return host["launch"](run, output, REPO, reconcile_kernel=bind_kernel, observe=observe,
                          expires=expires or (lambda: host["parent_expiry"](member["fields"])), accounting=current,
                          runner=runner or subprocess.run, wait_terminal=wait or accounting["wait_terminal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "claim-sha256", "stage-manifest-sha256"):
        parser.add_argument("--" + arg, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "launch", "reconcile"):
        modes.add_argument("--" + mode, dest="mode", action="store_const", const=mode)
    args = parser.parse_args()
    data = execute(args.mode, args.publication, args.claim_sha256, args.stage_manifest_sha256)
    print(json.dumps(data, allow_nan=False), flush=True)
    return int(data["state"] not in {"T512_NATIVE_PRECHECKED", "NATIVE_VERIFIED"})


if __name__ == "__main__":
    raise SystemExit(main())
