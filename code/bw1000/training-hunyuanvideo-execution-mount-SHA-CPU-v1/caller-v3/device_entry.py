"""Use the Slurm-assigned DCU for the audited native HunyuanVideo replay."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import time

from assigned_kfd import assigned_kfd
from framework_binding import DEPLOYMENT, verify_framework
from run_replay import prepare, remaining

PARENT = "1103606"
NODE = "f13r3n13"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def step_card(env, node):
    job = env.get("SLURM_JOB_ID")
    step = env.get("SLURM_STEP_ID", "")
    selector = env.get("CUDA_VISIBLE_DEVICES", "")
    if (node != NODE or job != PARENT or not re.fullmatch(r"[0-9]+", step)
            or not re.fullmatch(r"0|[1-9][0-9]*", selector)):
        raise ValueError("Expected one assigned DCU in the original T09 node and Slurm allocation")
    return f"{job}.{step}", selector


def check_gate(gate, step, boot_id, pci, node, observed_ns):
    if (gate.get("node") != node or gate.get("step") != step or gate.get("boot_id") != boot_id
            or type(gate.get("kfd_holder_count")) is not int or gate["kfd_holder_count"] != 0):
        raise ValueError("Same-step host device record differs")
    sampled_ns = gate.get("monotonic_ns")
    if type(sampled_ns) is not int or not 0 <= observed_ns - sampled_ns <= 90_000_000_000:
        raise ValueError("Same-step host device record is stale")
    inventory = gate.get("pci")
    if (type(inventory) is not list or len(inventory) != 8 or len(set(inventory)) != 8
            or pci not in inventory):
        raise ValueError("Runtime PCI is outside the same-step eight-DCU host inventory")


def runtime_pci():
    hip = ctypes.CDLL("libamdhip64.so")
    get_count = hip.hipGetDeviceCount
    get_count.argtypes = (ctypes.POINTER(ctypes.c_int),)
    get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if get_count(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError("HIP did not expose exactly one Slurm-assigned DCU")
    get_bus = hip.hipDeviceGetPCIBusId
    get_bus.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int)
    get_bus.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(32)
    if get_bus(buffer, len(buffer), 0) != 0:
        raise RuntimeError("Could not resolve the assigned DCU PCI address")
    pci = buffer.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        pci = "0000:" + pci
    if not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        raise RuntimeError("HIP reported an invalid PCI address")
    return pci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "workspace", "framework", "launch", "audit", "lock-root", "gate", "execution-authority"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    node = socket.gethostname().split(".", 1)[0]
    step, args.card = step_card(os.environ, node)
    args.lock_host = node
    remaining(args.deadline)
    with args.gate.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Oversized host device receipt")
    gate = json.loads(raw)
    observed_ns = time.monotonic_ns()
    manifest = args.results / DEPLOYMENT / "source/FRAMEWORK_FILES.json"
    framework = verify_framework(args.framework, manifest)
    argv = prepare(args)
    pci = runtime_pci()
    check_gate(gate, step, BOOT_ID.read_text().strip(), pci, node, observed_ns)
    device = assigned_kfd(pci)
    with (args.workspace / "assigned-device.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(step=step, node=node, selector=args.card, device=device,
                       boot_id=gate["boot_id"], framework=framework, transformer=args.weight_admission), stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    argv[argv.index("--capture-timeout") + 1] = str(remaining(args.deadline))
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
