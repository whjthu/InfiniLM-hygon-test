"""Observe assigned-node PCI and KFD/process data without loading the device SDK."""
import json
import os
from pathlib import Path
import re
import socket
import sys


def inspect(expected):
    devices = Path("/sys/class/drm")
    kfd = Path("/sys/class/kfd/kfd/proc")
    if not kfd.is_dir():
        raise ValueError("KFD process inventory unavailable")
    pci = sorted({p.resolve().name.lower() for p in devices.glob("renderD*/device")})
    living = []
    for row in expected:
        path = Path("/proc", str(row["PID"]), "stat")
        try:
            stat = path.read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            continue
        if "startticks" not in row or int(stat[19]) == row["startticks"]:
            living.append(row["PID"])
    holders = sorted(p.name for p in kfd.iterdir() if p.name.isdigit())
    if (len(pci) != 8 or not all(re.fullmatch("[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}.[0-7]", b) for b in pci)
            or holders or living):
        raise ValueError("current PCI/KFD/PID pre/post requirement failed")
    return {
        "PCI": pci, "KFD_holders": holders, "matching_owned_PIDs": living,
        "actual_CPU": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
        "node": socket.gethostname().split(".")[0],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


if __name__ == "__main__":
    print(json.dumps(inspect(json.loads(sys.argv[1]))))
