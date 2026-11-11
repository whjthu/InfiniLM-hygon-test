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
NODE = "a14r1n20"
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


if __name__ == "__main__":
    print(json.dumps(snapshot(False)), flush=True)
