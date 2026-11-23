"""Check eight isolated HIP child PCI views inside one newly admitted step."""

import argparse
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


REQUEST = "8e8e80b5bf56459da1aac27e858bcb2e"
LABEL = "O01-zimage-lt-eight-card-mask-preflight-v1"
PARENT, NODE = "1103606", "f12r2n11"
RESERVED_STEP = "1103606.2075"
PROBE_PCI = "0000:09:00.0"
PCI_SET = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
HIP = Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
HIP_BYTES = 12_349_016
WINDOW = 20


def need(ok, label):
    if not ok:
        raise ValueError(label)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, size):
    with path.open("rb") as handle:
        raw = handle.read(size + 1)
    need(0 < len(raw) <= size, "RECEIPT_SIZE")
    return raw


def value(path, expected, size):
    raw = read(path, size)
    need(digest(raw) == expected, "ADMITTED_SCOPE_SHA256")
    return json.loads(raw)


def write(path, data):
    with path.open("xb") as handle:
        handle.write((json.dumps(data, allow_nan=False, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def admission(scope):
    job, ordinal = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    step = job + "." + ordinal
    need(scope["request"] == REQUEST and scope["label"] == LABEL and scope["fresh_root_admitted"] is True
         and job == scope["parent"] == PARENT and ordinal.isdecimal() and step != RESERVED_STEP
         and socket.gethostname().split(".")[0] == scope["node"] == NODE,
         "NEW_OWNED_STEP_REQUIRED")
    boot = read(Path("/proc/sys/kernel/random/boot_id"), 128)
    need(digest(boot) == scope["boot_id_SHA256"] and scope["allocated_DCU"] == 8
         and len(scope["PCI"]) == 8 and set(scope["PCI"]) == PCI_SET, "PARENT_EIGHT_PCI_ADMISSION")
    need(re.fullmatch(PARENT + r"\.[0-9]+", scope["independent_pre_step"])
         and scope["independent_pre_step"] not in (step, RESERVED_STEP)
         and re.fullmatch("[0-9a-f]{64}", scope["independent_pre_SHA256"]), "INDEPENDENT_PRE_STEP_REQUIRED")
    start, end = scope["epoch"], scope["deadline_epoch"]
    need(all(type(t) in (float, int) and math.isfinite(t) for t in (start, end))
         and end == start + 240 and start <= time.time() < end - 5, "ORIGINAL_EPOCH")
    return step


def visible(expected_count):
    need(HIP.resolve(strict=True) == HIP and HIP.stat().st_size == HIP_BYTES
         and digest(read(HIP, HIP_BYTES)) == HIP_SHA, "EXACT_HIP_PROVIDER")
    hip = C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    count, pci = hip.hipGetDeviceCount, hip.hipDeviceGetPCIBusId
    count.argtypes, count.restype = [C.POINTER(C.c_int)], C.c_int
    pci.argtypes, pci.restype = [C.c_void_p, C.c_int, C.c_int], C.c_int
    actual = C.c_int()
    need(count(C.byref(actual)) == 0 and actual.value == expected_count, "HIP_VISIBLE_COUNT")
    devices = []
    for index in range(expected_count):
        token = C.create_string_buffer(32)
        need(pci(token, len(token), index) == 0, "HIP_PCI_LOOKUP")
        decoded = token.value.decode("ascii").lower()
        devices.append("0000:" + decoded if len(decoded) == 7 else decoded)
    need(len(set(devices)) == expected_count and set(devices) <= PCI_SET, "HIP_PCI_IDENTITY")
    return devices


def selectors(environment):
    """Choose child HIP tokens in the parent HIP-visible order."""
    supplied = [environment[key].split(",") for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                if key in environment]
    if not supplied:
        return list(map(str, range(8)))
    first = supplied[0]
    need(all(row == first for row in supplied) and len(first) == len(set(first)) == 8
         and all(re.fullmatch("(0|[1-9][0-9]{0,2})", selector) for selector in first),
         "INHERITED_HIP_SELECTOR_UNPROVED")
    return first


def child_environment(token, parent):
    # ROCR_VISIBLE_DEVICES and GPU_DEVICE_ORDINAL, when present, are inherited.
    return dict(parent, HIP_VISIBLE_DEVICES=token, CUDA_VISIBLE_DEVICES=token)


def check_child(scope, step, args):
    need(os.getppid() == args.parent_pid and args.pci in PCI_SET
         and os.environ.get("HIP_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES") == args.token,
         "MASK_CHILD_OWNERSHIP")
    [actual] = visible(1)
    need(actual == args.pci, "MASKED_HIP_PCI_MISMATCH")
    write(args.out, {"request": scope["request"], "step": step, "PID": os.getpid(), "parent_PID": os.getppid(),
                     "physical_PCI": actual, "visible_ordinal": 0, "visible_count": 1, "selector": args.token})


def assess(assignments, children):
    need(len(assignments) == len(children) == 8, "EIGHT_CHILD_DENOMINATOR")
    for (token, pci, pid, step, owner), result in zip(assignments, children):
        need(result == {"request": REQUEST, "step": step, "PID": pid, "parent_PID": owner, "physical_PCI": pci,
                        "visible_ordinal": 0, "visible_count": 1, "selector": token}, "CHILD_BINDING_DIFFERS")
    need({child["physical_PCI"] for child in children} == PCI_SET
         and len({child["PID"] for child in children}) == 8, "CHILD_PCI_BIJECTION")
    return {"probe_PCI": PROBE_PCI, "standby_PCIs": sorted(PCI_SET - {PROBE_PCI})}


def prove(scope, step, args, result):
    parent = os.getpid()
    devices = visible(8)
    need(set(devices) == PCI_SET, "OWNED_PARENT_PCI_DIFFERS")
    selected = selectors(os.environ)
    commands, assignments, launched = [], [], []
    end = time.monotonic() + min(WINDOW, scope["deadline_epoch"] - time.time() - 5)
    for index, (token, pci) in enumerate(zip(selected, devices)):
        path = args.out / f"mask-{index}.json"
        command = [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--scope", str(args.scope),
                   "--scope-sha256", args.scope_sha256, "--out", str(path), "--parent-pid", str(parent),
                   "--token", token, "--pci", pci]
        commands.append((command, path, token, pci))
    try:
        for command, path, token, pci in commands:
            need(time.monotonic() < end, "MASK_PROBE_TIME_BOUND")
            with path.with_suffix(".stderr.private").open("xb") as error:
                child = subprocess.Popen(command, env=child_environment(token, os.environ),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error)
            launched.append((child, path, pci))
            assignments.append((token, pci, child.pid, step, parent))
            result["launched_children"].append({"PID": child.pid, "intended_PCI": pci})
        for child, path, pci in launched:
            rc = child.wait(timeout=max(0.0, end - time.monotonic()))
            result["child_outcomes"].append({"PID": child.pid, "intended_PCI": pci, "returncode": rc})
            if rc == 0:
                receipt = json.loads(read(path, 2048))
                need(type(receipt) is dict, "MASK_CHILD_RECEIPT_TYPE")
                result["child_receipts"].append(receipt)
        result["verified_mapping"] = assess(assignments, result["child_receipts"])
        result["mask_bijection_passed"] = True
    finally:
        cleanup_end = time.monotonic() + 2
        for child, _, _ in launched:
            if child.poll() is None:
                child.kill()
        observed = {row["PID"] for row in result["child_outcomes"]}
        for child, _, pci in launched:
            if child.poll() is None:
                try:
                    child.wait(timeout=max(0, cleanup_end - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            if child.pid not in observed:
                result["child_outcomes"].append(
                    {"PID": child.pid, "intended_PCI": pci, "returncode": child.poll()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--token")
    parser.add_argument("--pci")
    args = parser.parse_args()
    scope = value(args.scope, args.scope_sha256, 16384)
    step = admission(scope)
    if args.parent_pid is not None:
        check_child(scope, step, args)
        return 0
    args.out.mkdir(mode=0o700)
    result = {"request": REQUEST, "step": step, "scope_SHA256": args.scope_sha256,
              "independent_pre_step": scope["independent_pre_step"],
              "independent_pre_SHA256": scope["independent_pre_SHA256"], "planned_children": 8,
              "launched_children": [], "child_outcomes": [], "child_receipts": [], "mask_bijection_passed": False,
              "actual_matmul_calls": 0, "actual_BF16_standby_workers": 0, "training_performance": False}
    try:
        prove(scope, step, args, result)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
        if type(error) is ValueError and re.fullmatch("[A-Z0-9_]+", str(error)):
            result["error_gate"] = str(error)
    finally:
        write(args.out / "terminal.json", result)
    return int(not result["mask_bijection_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
