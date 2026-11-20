"""One fixed eight-rank cross-node measurement."""
import hashlib
import math
import os
from pathlib import Path
import time

BYTES = 25 * 1024**2
PORT_BASE = 29820


def exercise(device, card, node, nodes, arm, mode, deadline, measured):
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    if len(nodes) != 2 or len(set(nodes)) != 2 or node not in nodes or not 0 <= card < 8 or not 0 <= arm < 4:
        raise ValueError("PHYSICAL_AND_LOGICAL_PLACEMENT")
    rank = nodes.index(node)*4 + card%4
    port = PORT_BASE + 2*arm + card//4
    expect = "IBext_v8" if mode == "shca" else "Socket"
    if os.environ.get("NCCL_NET") != expect:
        raise ValueError("EXACT_NETWORK_POLICY")
    payload = torch.empty(BYTES//4,dtype=torch.float32,device=device)
    try:
        dist.init_process_group("nccl",init_method=f"tcp://{nodes[0]}:{port}",rank=rank,world_size=8,
                                timeout=timedelta(seconds=38),device_id=device)
        elapsed=[]
        for step in range(108):
            if time.time()+3 >= deadline:
                raise TimeoutError("ARM_DEADLINE")
            payload.fill_(rank+1)
            torch.cuda.synchronize(device)
            t=time.perf_counter_ns()
            dist.all_reduce(payload,op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            dt=(time.perf_counter_ns()-t)*1e-9
            if step in (0,107) and not bool((payload==36).all().item()):
                raise ValueError("EXACT_PAYLOAD_RESULT")
            if step>=12:
                measured()
                elapsed.append(dt)
        times=torch.tensor(elapsed,device=device,dtype=torch.float32)
        dist.all_reduce(times,op=dist.ReduceOp.MAX)
        values=times.tolist()
        if len(values)!=96 or any(not math.isfinite(x) or x<=0 for x in values):
            raise ValueError("RANK_MAX_TIMING")
        plugin=set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            token=line.rsplit(None,1)[-1]
            if token.startswith("/") and "librccl-net-shca.so." in token:
                plug=Path(token)
                if plug.stat().st_size > 8*1024**2:
                    raise ValueError("PLUGIN_PATH_SIZE")
                plugin.add(hashlib.sha256(plug.read_bytes()).hexdigest())
        expected_plugin={"20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"} if mode=="shca" else set()
        if plugin!=expected_plugin:
            raise ValueError("EXACT_ACTUAL_RCCL_PLUGIN")
        return dict(network_policy=expect,plugin_sha256=sorted(plugin),logical_rank=rank,
                    group=card//4,payload_bytes=BYTES,verified_elements=2*payload.numel(),rank_max_seconds=values)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
