"""One same-provider RHS layout change, including its copy in each 16-call burst."""

from collections import Counter
import math
import statistics
import time


M, N, K = 288, 3840, 3840
DIRECT, PACKED = "transposed", "packed"
LAYOUTS = {DIRECT: (1, K), PACKED: (N, 1)}
ORDER = ((DIRECT, PACKED), (PACKED, DIRECT), (DIRECT, PACKED))
CALLS, BURSTS, WARM_BURSTS = 16, 4, 2
PACK_BYTES = 2 * K * N
PACK_LOGICAL_TRAFFIC_BYTES = 2 * PACK_BYTES
SEEDS = (179376, 179377)
MIN_WALL_SPEEDUP = 1.05


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def pairs():
    return [{"pair": i, "order": list(order), "status": "PLANNED", "arms": []}
            for i, order in enumerate(ORDER, 1)]


def inputs(torch, guarded):
    # Reuse the same logical RHS across both independent left operands.
    def small(seed, shape):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(-8, 9, shape, generator=generator, dtype=torch.int32).float().div_(32)

    rhs = small(SEEDS[0], (K, N))
    left = [small(seed, (M, K)) for seed in SEEDS]
    refs = [torch.mm(a, rhs).to(torch.bfloat16).to("cuda") for a in left]
    a = [guarded(torch, (M, K), (K, 1), value.to(torch.bfloat16)) for value in left]
    source = guarded(torch, (K, N), LAYOUTS[DIRECT], rhs.to(torch.bfloat16))
    packed = guarded(torch, (K, N), LAYOUTS[PACKED])
    packed.reference = source.reference
    require(all(torch.isfinite(ref).all().item() for ref in refs), "CPU_ORACLE_NONFINITE")
    return a, source, packed, refs


def pack(source, packed):
    require(tuple(source.view.shape) == tuple(packed.view.shape) == (K, N)
            and tuple(source.view.stride()) == LAYOUTS[DIRECT]
            and tuple(packed.view.stride()) == LAYOUTS[PACKED]
            and source.view.untyped_storage().data_ptr() != packed.view.untyped_storage().data_ptr()
            and source.view.dtype == packed.view.dtype and source.view.device == packed.view.device, "RHS_LAYOUT_OR_ALIAS")
    packed.view.copy_(source.view, non_blocking=False)


def witness_operand(torch, guarded, layout):
    a, source, packed, refs = inputs(torch, guarded)
    if layout == PACKED:
        pack(source, packed)
    else:
        require(layout == DIRECT, "PLACEMENT_NOT_ALLOWED")
    return a[0], source if layout == DIRECT else packed, refs[0]


def check_values(torch, guarded_runtime, values, order, left, source, packed, refs, layout):
    for i, value in zip(order, values, strict=True):
        require(guarded_runtime.same_bits(torch, value, refs[i]), "MM_OUTPUT_DIFFERS")
    guarded_runtime.audit_inputs(torch, [(left[0], source, refs[0]), (left[1], source, refs[1])])
    guarded_runtime.check_guard(packed)
    if layout == PACKED:
        require(guarded_runtime.same_bits(torch, packed.view.cpu(), packed.reference), "PACKED_LOGICAL_RHS_DIFFERS")


def burst(torch, buffers, layout, events, stream, offset):
    left, source, packed, _ = buffers
    require(layout in LAYOUTS, "RHS_LAYOUT_UNKNOWN")
    order = [(i + offset) % len(SEEDS) for i in range(CALLS)]
    start, packed_at, end = events
    started = time.perf_counter_ns()
    start.record(stream)
    if layout == PACKED:
        pack(source, packed)
    packed_at.record(stream)
    right = packed if layout == PACKED else source
    values = [torch.mm(left[i].view, right.view) for i in order]
    end.record(stream)
    end.synchronize()
    wall_s = (time.perf_counter_ns() - started) * 1e-9
    whole_ms = start.elapsed_time(end)
    pack_ms = start.elapsed_time(packed_at)
    gemm_ms = packed_at.elapsed_time(end)
    require(all(math.isfinite(t) and t > 0 for t in (wall_s, whole_ms, gemm_ms))
            and math.isfinite(pack_ms) and pack_ms >= 0 and whole_ms >= gemm_ms, "UNUSABLE_PAIRED_CLOCK")
    return {"total_event_ms": whole_ms, "first_interval_ms": pack_ms, "GEMM_event_ms": gemm_ms,
            "wall_seconds": wall_s, "copies": int(layout == PACKED)}, values, order


def summarize(plan):
    expected = pairs()
    require(len(plan) == len(expected) and all((p["pair"], p["order"]) == (e["pair"], e["order"])
            for p, e in zip(plan, expected)), "PAIR_DENOMINATOR")
    metrics = {"total_event_ms": [], "GEMM_event_ms": [], "wall_seconds": []}
    for pair in plan:
        if pair["status"] != "COMPLETE":
            continue
        arms = pair["arms"]
        require([arm["layout"] for arm in arms] == pair["order"], "PAIR_ORDER")
        for arm in arms:
            require(len(arm["bursts"]) == BURSTS and arm["verified_measured_calls"] == BURSTS * CALLS
                    and arm["verified_warm_calls"] == WARM_BURSTS * CALLS
                    and all(b["copies"] == int(arm["layout"] == PACKED) for b in arm["bursts"]),
                    "CALL_OR_PACK_DENOMINATOR")
        summaries = {arm["layout"]: {key: sum(b[key] for b in arm["bursts"]) for key in metrics}
                     for arm in arms}
        for key, ratios in metrics.items():
            direct, packed = (summaries[layout][key] for layout in (DIRECT, PACKED))
            require(all(math.isfinite(t) and t > 0 for t in (direct, packed)), "PAIR_TIME_INVALID")
            ratios.append(direct / packed)
    medians = {key: statistics.median(value) if len(value) == len(expected) else None for key, value in metrics.items()}
    return {"pairs": len(expected), "statuses": dict(Counter(p["status"] for p in plan)),
            "paired_baseline_over_candidate": metrics, "median": medians,
            "clears_component_wall_gate": medians["wall_seconds"] is not None
             and medians["wall_seconds"] >= MIN_WALL_SPEEDUP and min(metrics["wall_seconds"]) >= 1.0}
