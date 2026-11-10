"""Run the frozen training replay on the single assigned HIP device."""

import ctypes as C
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys


def main():
    node = socket.gethostname().split(".")[0]
    if node != "f11r4n11" or os.environ["SLURM_JOB_ID"] != "1087656":
        raise RuntimeError("foreign T06 allocation")
    import torch

    runtime = C.CDLL("libamdhip64.so")
    count = C.c_int()
    if runtime.hipGetDeviceCount(C.byref(count)) or count.value != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("T06 replay requires one visible HIP device")
    pci = C.create_string_buffer(32)
    if runtime.hipDeviceGetPCIBusId(pci, len(pci), 0):
        raise RuntimeError("assigned HIP PCI identity unavailable")
    pci_id = pci.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci_id):
        pci_id = "0000:" + pci_id
    if not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci_id):
        raise RuntimeError("invalid assigned HIP PCI identity")
    state = dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                 node=node, pci=pci_id, hip=torch.version.hip, visible_cards=count.value)
    with Path("/t06-run/assigned.json").open("x", encoding="utf-8") as stream:
        json.dump(state, stream)
    source = "/t06-run/source/tools/hygon_sdxl_first_step.py"
    sys.argv = [source, "--reference", "/reference/reference.pt", "--policy", "/reference/policy.json",
                "--weights", "/model", "--output", "/t06-run/replay"]
    runpy.run_path(source, run_name="__main__")


if __name__ == "__main__":
    main()
