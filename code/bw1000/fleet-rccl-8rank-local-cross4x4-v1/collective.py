"""Matched eight-rank RCCL timing on local and four-plus-four placements."""
import math
import time

BYTES = 25 * 1024**2
PORT_BASE = 29664


def rank_plan(nodes, node, device, arm_index):
    if len(nodes) not in (1, 2) or len(set(nodes)) != len(nodes) or node not in nodes:
        raise ValueError("EXACT_PAIR_COHORT")
    if type(device) is not int or not 0 <= device < 8 or not 0 <= arm_index < 4:
        raise ValueError("EXACT_RANK_AND_ARM")
    if len(nodes) == 1 or arm_index in (0, 3):
        return None, device
    half = device // 4
    return PORT_BASE + 2 * (arm_index - 1) + half, nodes.index(node) * 4 + device % 4


def exercise(device, card, node, nodes, local_store, deadline, pilot, measured):
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    results = []
    timed = 6 if pilot else 192
    warm = 3 if pilot else 12
    payload = torch.empty(BYTES // 4, dtype=torch.float32, device=device)
    for arm in range(4):
        if time.time() + 8 >= deadline:
            raise TimeoutError("RCCL_PAIR_DEADLINE")
        port, rank = rank_plan(nodes, node, card, arm)
        store = (local_store.with_name(f"local-{arm}").as_uri() if port is None
                 else f"tcp://{nodes[0]}:{port}")
        try:
            dist.init_process_group("nccl", init_method=store, rank=rank, world_size=8,
                                    timeout=timedelta(seconds=34), device_id=device)
            observed = []
            for step in range(warm + timed):
                if time.time() >= deadline:
                    raise TimeoutError("RCCL_REDUCTION_DEADLINE")
                payload.fill_(rank+1)
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                dist.all_reduce(payload, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                if step in (0, warm + timed - 1):
                    if not bool((payload == 36.0).all().item()):
                        raise ValueError("RCCL_WHOLE_PAYLOAD_SUM")
                if step >= warm:
                    measured()
                    observed.append(elapsed)
            times = torch.tensor(observed, dtype=torch.float32, device=device)
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
            times = times.tolist()
            if len(times) != timed or not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError("RCCL_RANK_MAX_TIMES")
            results.append(dict(arm=arm, cross=port is not None, logical_rank=rank, payload_bytes=BYTES,
                                payload_dtype="float32", verified_elements=2*payload.numel(),
                                measured_reductions=timed, rank_max_seconds=times))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    return dict(blocks=results, distributed_training=False,
                placement="eight ranks on one node versus four plus four ranks across two nodes")
