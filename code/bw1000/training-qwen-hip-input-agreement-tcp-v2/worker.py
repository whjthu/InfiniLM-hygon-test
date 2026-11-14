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

    assert os.environ["LOCAL_RANK"] == os.environ["RANK"] == str(rank)
    assert os.environ["WORLD_SIZE"] == os.environ["LOCAL_WORLD_SIZE"] == "8"
    assert os.environ["MASTER_ADDR"] and 0 < int(os.environ["MASTER_PORT"]) < 65536
    assert time.time() < intent["work_deadline"]
    stages = root / "ranks"
    save(stages / f"{rank}-before-init.json", dict(monotonic_ns=time.monotonic_ns(), rank=rank,
         master_port=int(os.environ["MASTER_PORT"]), launcher="torchrun", rendezvous="env://"))
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(seconds=30))
    save(stages / f"{rank}-after-init.json", dict(monotonic_ns=time.monotonic_ns(), rank=rank))
    outcomes = []
    try:
        flag = torch.empty((), dtype=torch.int32, device=torch.device("cuda", rank))
        for index, rejecting_rank in enumerate([None, *range(8)]):
            assert time.time() < intent["work_deadline"]
            matched = rank != rejecting_rank
            flag.fill_(int(matched))
            save(stages / f"{rank}-vote-{index}-start.json",
                 dict(monotonic_ns=time.monotonic_ns(), rank=rank, index=index))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            vote = dict(rejecting_rank=rejecting_rank, local_vote=int(matched), agreed=bool(flag.item()))
            outcomes.append(vote)
            save(stages / f"{rank}-vote-{index}-end.json",
                 dict(monotonic_ns=time.monotonic_ns(), rank=rank, index=index, vote=vote))
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
    assert not any(name in os.environ for name in
                   ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))
    budget = int(intent["work_deadline"] - time.time())
    assert budget >= 100
    args = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0",
            __file__, str(root), "--rank-worker"]
    save(root / "launcher.json", dict(module="torch.distributed.run", standalone=True,
                                      max_restarts=0, rendezvous="tcp", budget_seconds=budget))
    result = subprocess.run(args, timeout=budget, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        launch_ranks()
    else:
        assert sys.argv[2:] == ["--rank-worker"]
        rank_main(int(os.environ["RANK"]))
