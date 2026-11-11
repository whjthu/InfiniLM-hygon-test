"""Read per-node DRM, KFD-process and host metadata without initializing a device."""

import json
import os
from pathlib import Path
import re
import socket
import hashlib
import sys
import time

assert os.environ["SLURM_JOB_ID"] == "1087656"
assert os.environ["SLURM_STEP_ID"].isdecimal()
node = socket.gethostname().split(".")[0]
drm = Path("/sys/class/drm")
cards = []
for entry in drm.iterdir():
    if not re.fullmatch(r"renderD\d+",entry.name):
        continue
    device = entry/"device"
    pci = Path(os.readlink(device)).name.lower()
    assert re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",pci)
    cards.append({"pci":pci,"render":entry.name,"vendor":(device/"vendor").read_text().strip(),
                  "device":(device/"device").read_text().strip(),
                  "driver":(device/"driver").resolve().name})
assert len({row["pci"] for row in cards})==len(cards)==8
kfd = Path("/sys/class/kfd/kfd/proc")
processes = sorted(p.name for p in kfd.iterdir())
assert all(name.isdecimal() for name in processes)
meminfo = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
result={"node":node,"step":"1087656."+os.environ["SLURM_STEP_ID"],"rank":int(os.environ["SLURM_PROCID"]),
        "cards":sorted(cards,key=lambda row:row["pci"]),"kfd_process_count":len(processes),
        "cpu_count":os.cpu_count(),"memory_total_kB":int(meminfo["MemTotal"].split()[0]),
        "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "monotonic_ns":time.monotonic_ns(),"wall_epoch":time.time(),
        "scope":"read-only CPU metadata; no device initialization, model execution or busy measurement"}
directory=Path(sys.argv[1])
assert directory.is_dir() and directory.name=="nodes"
temporary=directory/(node+".partial")
final=directory/(node+".json")
assert not final.exists()
raw=(json.dumps(result)+"\n").encode()
with temporary.open("xb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
temporary.rename(final)
print(json.dumps({"node":node,"rank":result["rank"],"SHA256":hashlib.sha256(raw).hexdigest()}),flush=True)
