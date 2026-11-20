"""Model-shaped TP16 network comparison on two complete eight-card nodes."""
from datetime import timedelta
import hashlib
import math
import os
from pathlib import Path
import time

PAYLOAD_ELEMENTS = (6144,)
ELEMENT_BYTES = 2
SAMPLE_STEPS = (12, 44, 76)


def exercise(device, card, node, nodes, arm, mode, port_base, deadline, measured, pci):
    import torch
    import torch.distributed as dist

    if len(nodes) != 2 or len(set(nodes)) != 2 or node not in nodes or not 0 <= card < 8 or not 0 <= arm < 4:
        raise ValueError("PHYSICAL_AND_LOGICAL_PLACEMENT")
    if port_base != 30640:
        raise ValueError("WAVE_PORT_RANGE")
    rank = nodes.index(node) * 8 + card
    port = port_base + 2 * arm
    expected = "IBext_v8"
    algorithm = {"ring": "Ring", "tree": "Tree"}[mode]
    if os.environ.get("NCCL_NET") != expected or os.environ.get("NCCL_ALGO") != algorithm:
        raise ValueError("EXACT_NETWORK_POLICY")
    busy_path = Path("/sys/bus/pci/devices") / pci / "gpu_busy_percent"
    sizes = {}
    try:
        dist.init_process_group("nccl", init_method=f"tcp://{nodes[0]}:{port}", rank=rank, world_size=16,
                                timeout=timedelta(seconds=38), device_id=device)
        for elements in PAYLOAD_ELEMENTS:
            column = torch.arange(elements, dtype=torch.int32, device=device) % 15 - 7
            local = (column + rank - 8).bfloat16()
            expected_sum = (column * 16 - 8).bfloat16()
            payload = torch.empty_like(local)
            values = []
            busy = []
            for step in range(108):
                if time.time() + 3 >= deadline:
                    raise TimeoutError("ARM_DEADLINE")
                payload.copy_(local)
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                dist.all_reduce(payload, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(device)
                ended = time.perf_counter_ns()
                if step in (0, 107) and not torch.equal(payload, expected_sum):
                    raise ValueError("EXACT_PAYLOAD_RESULT")
                if step >= 12:
                    measured()
                    values.append((ended - started) * 1e-9)
                if step in SAMPLE_STEPS:
                    at = time.time()
                    busy.append([step, at, int(busy_path.read_text())])
            maxima = torch.tensor(values, device=device, dtype=torch.float32)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            times = maxima.tolist()
            if len(times) != 96 or any(not math.isfinite(x) or x <= 0 for x in times):
                raise ValueError("RANK_MAX_TIMING")
            sizes[str(elements)] = dict(payload_bytes=elements * ELEMENT_BYTES, verified_elements=2 * elements,
                                        rank_max_seconds=times, rank_busy_after_collective=busy)
            del payload, maxima
        loaded = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            token = line.rsplit(None, 1)[-1]
            if token.startswith("/") and "librccl-net-shca.so." in token:
                plug = Path(token)
                if plug.stat().st_size > 8 * 1024**2:
                    raise ValueError("PLUGIN_PATH_SIZE")
                loaded.add(hashlib.sha256(plug.read_bytes()).hexdigest())
        expected_plugin = {"20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"}
        if loaded != expected_plugin:
            raise ValueError("EXACT_ACTUAL_RCCL_PLUGIN")
        return dict(algorithm_policy=algorithm, network_policy=expected, plugin_sha256=sorted(loaded), logical_rank=rank, world_size=16,
                    sizes=sizes)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
