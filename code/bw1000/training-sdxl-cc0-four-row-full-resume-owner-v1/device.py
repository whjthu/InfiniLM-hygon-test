"""Bind the only visible HIP ordinal to the physical roster before native SDXL."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pci():
    hip = ctypes.CDLL("libamdhip64.so")
    count = ctypes.c_int()
    value = ctypes.create_string_buffer(32)
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
         and hip.hipDeviceGetPCIBusId(value, len(value), 0) == 0, "EXACT_ONE_VISIBLE_HIP_ORDINAL")
    text = value.value.decode("ascii").lower()
    return "0000:" + text if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text) else text


def main(pin, *argv):
    raw = Path("/owner/DEVICE.json").read_bytes()
    need(len(raw) <= 8192 and hashlib.sha256(raw).hexdigest() == pin, "EXACT_OWNER_DEVICE_DOCUMENT")
    owner = json.loads(raw)
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    need(owner["step"] == job + "." + step and job == "1194900" and step.isdecimal()
         and socket.gethostname().split(".")[0] == "f13r4n03"
         and owner["boot"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
         and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_STEP_GPUS", ""))
         and re.search(rf"/job_{job}(?:\.scope)?/step_{step}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE)
         and owner["request"] == os.environ.get("SDXL_SINGLE_DCU_OWNER_REQUEST"), "EXACT_ACTIVE_GPU_STEP")
    selected = pci()
    need(selected in owner["pci"] and len(set(owner["pci"])) == 8, "VISIBLE_GPU_OUTSIDE_OWNER_ROSTER")
    with Path("/durable/device.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(request=owner["request"], step=owner["step"], boot=owner["boot"], pci=selected,
                       ordinal=0, visible_devices=1), stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sys.path.insert(0, "/scope")
    sys.argv = ["/scope/run.py", *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main(*sys.argv[1:])
