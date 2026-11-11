"""Bind the Slurm-assigned HIP PCI identity, then run the TRELLIS SDPA diagnostic."""

import ctypes
import json
import os
from pathlib import Path
import re
import runpy
import socket

assert socket.gethostname().split(".")[0] == "a14r1n20"
assert os.environ["SLURM_JOB_ID"] == "1087656"
hip=ctypes.CDLL("libamdhip64.so")
count=ctypes.c_int()
pci=ctypes.create_string_buffer(32)
assert hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==1
assert hip.hipDeviceGetPCIBusId(pci,len(pci),0)==0
name=pci.value.decode().lower()
if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",name):
    name="0000:"+name
assert re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",name)
with Path("/replay/assigned.json").open("x",encoding="utf-8") as stream:
    json.dump({"pci":name,"visible_cards":count.value,"step":"1087656."+os.environ["SLURM_STEP_ID"],"node":"a14r1n20"},stream)
runpy.run_path("/replay/tools/diagnose_block0_sdpa.py",run_name="__main__")
