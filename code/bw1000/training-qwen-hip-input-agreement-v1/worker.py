"""Exercise Qwen's scalar device MIN on the eight-device training topology."""
from datetime import timedelta
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from snapshot import drm_inventory, kfd_inventory

root = Path(sys.argv[1])
intent = json.loads((root / "intent.json").read_bytes())


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2)


def rank_main(rank):
    import torch
    import torch.distributed as dist

    assert socket.gethostname().split(".")[0] == intent["node"]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    assert step.startswith(intent["parent"] + ".")
    assert torch.__version__.split("+")[0] == "2.10.0"
    assert torch.version.hip == "6.3.26113" and torch.cuda.device_count() == 8
    torch.cuda.set_device(rank)
    hip = ctypes.CDLL("libamdhip64.so")
    bus = ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(bus, len(bus), rank) == 0
    pci = bus.value.decode().lower()
    if pci.count(":") == 1:
        pci = "0000:" + pci
    devices = json.loads((root / "leased.json").read_bytes())
    assert pci in devices
    save(root / "ranks" / f"{rank}-owner.json", dict(
        rank=rank, step=step, pid=os.getpid(), pci=pci, kfd=devices[pci]))

    assert time.time() < intent["work_deadline"]
    dist.init_process_group("nccl", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=8, timeout=timedelta(seconds=30))
    outcomes = []
    try:
        flag = torch.empty((), dtype=torch.int32, device=torch.device("cuda", rank))
        for rejecting_rank in [None, *range(8)]:
            assert time.time() < intent["work_deadline"]
            matched = rank != rejecting_rank
            flag.fill_(int(matched))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            outcomes.append(dict(rejecting_rank=rejecting_rank, local_vote=int(matched),
                                 agreed=bool(flag.item())))
        assert dist.get_rank() == rank and dist.get_world_size() == 8
        assert [row["agreed"] for row in outcomes] == [True] + [False] * 8
        assert dist.get_backend() == "nccl"
        dist.barrier(device_ids=[rank])
        libraries = sorted({line.split(maxsplit=5)[-1] for line in
                            Path("/proc/self/maps").read_text().splitlines()
                            if "/librccl.so" in line})
    finally:
        dist.destroy_process_group()

    save(root / "ranks" / f"{rank}.json", dict(
        node=intent["node"], rank=rank, step=step, pci=pci, kfd=devices[pci],
        torch=torch.__version__, hip=torch.version.hip, backend="nccl",
        torch_source=str(Path(torch.__file__).resolve()),
        torch_source_sha256=hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest(),
        distributed_source_sha256=hashlib.sha256(Path(dist.all_reduce.__code__.co_filename).read_bytes()).hexdigest(),
        mapped_rccl=libraries, observations=outcomes, process_group_destroyed=not dist.is_initialized()))


def launch_ranks():
    devices = kfd_inventory()
    before = json.loads((root / "pre" / (intent["node"] + ".json")).read_bytes())
    assert sorted(devices) == before["pci"] == drm_inventory()
    save(root / "leased.json", devices)
    ranks = []
    try:
        for rank in range(8):
            ranks.append(subprocess.Popen([sys.executable, "-B", __file__, str(root), str(rank)]))
        statuses = [child.wait(timeout=max(0.1, intent["work_deadline"] - time.time()))
                    for child in ranks]
        assert statuses == [0] * 8, statuses
    finally:
        for child in ranks:
            if child.poll() is None:
                child.kill()
        for child in ranks:
            child.wait(timeout=3)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        launch_ranks()
    else:
        rank_main(int(sys.argv[2]))
