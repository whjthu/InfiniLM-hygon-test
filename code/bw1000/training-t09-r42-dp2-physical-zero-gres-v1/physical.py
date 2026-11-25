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
NODES = {"f13r3n13"}
STEP = re.compile(r"1103606\.\d+")
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
    require(node in NODES and STEP.fullmatch(step), "wrong Slurm step or host")
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



PORT = 29642
CLOSED = {"1103606.1611", "1103606.1612", "1103606.1614"}


def physical_test(args, result):
    require(re.fullmatch(r"[0-9a-f]{32}",args.request) is not None, "T09_NEW_TEST_REQUEST")
    require(0<=time.time()-args.epoch<90,"T09_ORIGINAL_90S_CPU_EPOCH")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,"T09_ISOLATED_HOST_INTERPRETER")
    env=os.environ
    own=env.get("SLURM_JOB_ID","")+"."+env.get("SLURM_STEP_ID","")
    require(STEP.fullmatch(own) and own not in CLOSED and env.get("SLURM_CPUS_PER_TASK")=="1",
            "T09_NEW_NUMERIC_CPU_STEP")
    require(len(os.sched_getaffinity(0)) >= 1 and env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
            and all(env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
            and all(env.get(k)=="-1" for k in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")),
            "T09_ZERO_VISIBLE_DCU")
    groups=Path("/proc/self/cgroup").read_text("ascii").splitlines()
    required={"job_1103606","step_"+own.split(".",1)[1]}
    require(any(required<=set(row.split(":",2)[-1].split("/")) for row in groups),"T09_REAL_STEP_CGROUP")
    seen=[]
    for _ in range(2):
        observation=snapshot(False)
        topology=kfd_inventory()
        require(observation["step"]==own and observation["kfd_holder_count"]==0
                and sorted(topology)==observation["pci"],"T09_FRESH_EIGHT_PCI_ZERO_KFD_OR_SAME_UID_FD_HOLDERS")
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as check:
            check.bind(("127.0.0.1",PORT))
        seen.append(observation)
        time.sleep(.1)
    a,b=seen
    require(a["boot_id"]==b["boot_id"] and a["pci"]==b["pci"]
            and 0<b["monotonic_ns"]-a["monotonic_ns"]<=5_000_000_000
            and time.time()<args.epoch+90,"T09_SAME_BOOT_DOUBLE_SAMPLE")
    result.update(node=b["node"],step=own,pci=b["pci"],boot_id=b["boot_id"],
                  monotonic_ns=b["monotonic_ns"],port=PORT,port_available=True,
                  kfd_holder_count=0,physical_samples=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",type=float,required=True)
    parser.add_argument("--request",required=True)
    parser.add_argument("--mode",choices=("pre","post"),required=True)
    args=parser.parse_args()
    output=dict(request=args.request,epoch=args.epoch,mode=args.mode,passed=False,
                target_GPU=0,optimizer_updates=0,all_KFD_plus_same_UID_FD=True)
    try:
        physical_test(args,output)
        output["passed"]=True
    except (OSError,RuntimeError,ValueError,AttributeError) as e:
        code=str(e)
        output.update(error_class=type(e).__name__,predicate=code if re.fullmatch(r"T09_[A-Z0-9_]+",code) else None)
    print(json.dumps(output,sort_keys=True,separators=(",",":")),flush=True)
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
