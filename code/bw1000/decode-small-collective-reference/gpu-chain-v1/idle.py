"""Read only node-local GPU ownership before and after the single raw step."""
from pathlib import Path
import json
import os
import re
import socket

node = socket.gethostname().split(".")[0]
assert node == "a02r4n12"
pci = sorted(p.resolve(strict=True).name for p in Path("/sys/class/drm").glob("renderD*/device"))
assert len(pci) == len(set(pci)) == 8 and all(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}[.][0-7]", p) for p in pci)
kfd = sorted(int(p.name) for p in Path("/sys/class/kfd/kfd/proc").iterdir() if p.name.isdecimal())
workers = []
for p in Path("/proc").iterdir():
    if not p.name.isdecimal():
        continue
    try:
        args = (p / "cmdline").read_bytes().split(bytes([0]))
    except OSError:
        continue
    if b"/refactor/incoming/Deep-decode-chain-reference/gpu-chain-v1/worker.py" in args:
        workers.append(int(p.name))
step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
print(json.dumps(dict(node=node, actual_step=step, pci=pci, kfd_pids=kfd, worker_pids=workers)))
assert not kfd and not workers
