"""Verify host PCI ownership and complete the KFD gate inside the eight-DCU step."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


PCI = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
MAX_AGE_NS = 45_000_000_000
NODE = "f12r3n12"
STEP = re.compile(r"1087656\.\d+")
KFD = Path("/sys/class/kfd/kfd")
DRM = Path("/sys/class/drm")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def pci_for_render(render):
    value = Path(os.readlink(DRM / render / "device")).name.lower()
    require(PCI.fullmatch(value), "invalid PCI identity from DRM")
    return value


def drm_inventory():
    cards = sorted(pci_for_render(p.name) for p in DRM.iterdir() if re.fullmatch(r"renderD\d+", p.name))
    require(len(cards) == len(set(cards)) == 8, "expected eight distinct DRM devices")
    return cards


def kfd_inventory():
    cards = {}
    for node in (KFD / "topology/nodes").iterdir():
        gpu_id = int((node / "gpu_id").read_text())
        if gpu_id == 0:
            continue
        props = dict(row.split(maxsplit=1) for row in (node / "properties").read_text().splitlines())
        pci = pci_for_render("renderD" + str(int(props["drm_render_minor"])))
        domain, bus, device = pci.split(":")
        slot, function = device.split(".")
        location = int(bus, 16) << 8 | int(slot, 16) << 3 | int(function, 16)
        require(int(domain, 16) == int(props["domain"]) and location == int(props["location_id"]),
                "DRM and KFD disagree about the PCI identity")
        require(pci not in cards and gpu_id > 0 and gpu_id not in cards.values(), "duplicate PCI or KFD identity")
        cards[pci] = gpu_id
    require(len(cards) == 8, "expected eight distinct KFD devices")
    return dict(sorted(cards.items()))


def gpu_holders():
    kernel = {entry.name for entry in (KFD / "proc").iterdir()}
    require(all(name.isdecimal() for name in kernel), "unexpected KFD process-directory schema")
    same_uid = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target == "/dev/kfd" or target.startswith("/dev/dri/"):
                    same_uid.add(proc.name)
        except FileNotFoundError:
            continue
    return len(kernel | same_uid)


def snapshot(leased):
    node = socket.gethostname().split(".")[0]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(node == NODE and STEP.fullmatch(step), "wrong Slurm step or host")
    pci = drm_inventory()
    mapping = kfd_inventory() if leased else None
    if leased:
        require(sorted(mapping) == pci, "allocated KFD inventory differs from DRM")
    holders = gpu_holders()
    return {
        "kind": "i03_device_snapshot", "mode": "leased" if leased else "cpu",
        "node": node, "step": step, "pci": pci, "pci_to_kfd": mapping,
        "kfd_holder_count": holders,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "monotonic_ns": time.monotonic_ns(),
    }


def validate_before(before, leased):
    require(before["kind"] == leased["kind"] == "i03_device_snapshot"
            and before["mode"] == "cpu" and leased["mode"] == "leased", "wrong snapshot modes")
    require(before["node"] == leased["node"] == NODE and before["boot_id"] == leased["boot_id"],
            "host or boot changed")
    require(STEP.fullmatch(before["step"]) and STEP.fullmatch(leased["step"])
            and int(before["step"].split(".")[1]) < int(leased["step"].split(".")[1]),
            "CPU snapshot does not precede the leased step")
    require(before["accounting_state"] == "COMPLETED" and before["exit_code"] == "0:0",
            "CPU snapshot accounting failed")
    require(0 < leased["monotonic_ns"] - before["monotonic_ns"] <= MAX_AGE_NS,
            "CPU snapshot is stale")
    require(before["kfd_holder_count"] == leased["kfd_holder_count"] == 0, "GPU holders present")
    require(before["pci"] == leased["pci"] == sorted(leased["pci_to_kfd"])
            and len(leased["pci_to_kfd"]) == len(set(leased["pci_to_kfd"].values())) == 8,
            "PCI or KFD device identity differs")


def pinned_bytes(envelope, name, digest):
    raw = base64.b64decode(envelope[name], validate=True)
    require(0 < len(raw) <= 32768 and hashlib.sha256(raw).hexdigest() == digest,
            f"unexpected {name} bytes")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "leased"))
    for name in ("before-sha256", "inside-sha256", "probe-sha256", "deadline-epoch"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.mode == "cpu":
        row = snapshot(False)
        print(json.dumps(row), flush=True)
        return int(row["kfd_holder_count"] != 0)

    require(not os.environ.get("LD_PRELOAD"), "pre-admission process must not preload native libraries")
    deadline = float(args.deadline_epoch)
    require(130 <= deadline - time.time() <= 270, "invalid original dispatch deadline")
    encoded = sys.stdin.buffer.read(132097)
    require(0 < len(encoded) <= 132096, "invalid admission bundle size")
    envelope = json.loads(encoded)
    before_raw = pinned_bytes(envelope, "before", args.before_sha256)
    inside = pinned_bytes(envelope, "inside", args.inside_sha256)
    probe = pinned_bytes(envelope, "probe", args.probe_sha256)
    leased = snapshot(True)
    validate_before(json.loads(before_raw), leased)
    leased["before_sha256"] = args.before_sha256
    print(json.dumps(leased), flush=True)
    command = ["/bin/bash", "--noprofile", "--norc", "-c", inside.decode("utf-8"),
               "deep-key-inventory", args.deadline_epoch, args.probe_sha256]
    return subprocess.run(command, input=probe, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
