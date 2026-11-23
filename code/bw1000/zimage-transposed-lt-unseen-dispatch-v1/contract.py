"""Frozen diagnostic plan for the actual T07.sft square backward GEMMs."""

from collections import Counter
import math
import statistics


M, N, K = 288, 3840, 3840
BASELINE, ALTERNATIVE = "cublas", "cublaslt"
ORDER = ((BASELINE, ALTERNATIVE), (ALTERNATIVE, BASELINE), (BASELINE, ALTERNATIVE))
LAYOUTS = {"contiguous": (3840, 1), "transposed": (1, 3840)}
SEEDS = (179376, 179377)
BATCHES, CALLS_PER_BATCH = 8, 8
CALLS_PER_ARM = BATCHES * CALLS_PER_BATCH
GUARD = 256  # BF16 elements at each end, 512 bytes.
GUARD_VALUE = -9.75
CPU, CUDA = "DeviceType.CPU", "DeviceType.CUDA"
OPERAND_RECEIPT_SHA256 = "b2fd2a1263cd3bbafe44ff5a21aec4ace0a597bbcfcb65d0eba02534333a77eb"
KERNEL_TAIL = ("_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_"
               "ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_MMAC_BW1_MMAC_KM0_MMAC_KME1_NLCA1_"
               "NLCB1_OLDSR0_PGR1_PLR1_RK1_SU0_SUM0_SUS256_SVW4_TT8_4_TLM0_ULDSWM0_USFGRO0_USSB0_")
ORIGINAL = {
    "contiguous": "Cijk_Ailk" + KERNEL_TAIL + "VAW2_VSn1_VW4_WG16_16_1_WGM8",
    "transposed": "Cijk_Alik" + KERNEL_TAIL + "VAW1_VSn1_VW4_WG16_16_1_WGM8",
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def kernels(profiler, planned_calls):
    """Attribute actual CUDA (HIP) kernels by raw Kineto external correlation."""
    result = profiler.profiler.kineto_results
    need(result is not None, "MISSING_KINETO_RESULT")
    events = result.events()
    need(0 < len(events) <= 1024, "KINETO_ROUTE_BOUND")
    host = [e for e in events if str(e.device_type()) == CPU and e.name() == "aten::mm"]
    ids = [e.correlation_id() for e in host]
    need(len(ids) == len(set(ids)) == planned_calls and all(n > 0 for n in ids)
         and all([list(shape) for shape in e.shapes()[:2]] == [[M, K], [K, N]] for e in host),
         "MM_ROUTE_DENOMINATOR_OR_SHAPE")
    owners = {n: [] for n in ids}
    for event in events:
        if str(event.device_type()) != CUDA:
            continue
        owner = event.linked_correlation_id()
        need(owner in owners and event.device_index() == 0, "UNBOUND_DEVICE_EVENT_IN_MM_ROUTE")
        name = event.name()
        need(0 < len(name) <= 768 and name.isascii(), "UNEXPECTED_KERNEL_NAME")
        owners[owner].append(name)
    signatures = {tuple(sorted(names)) for names in owners.values()}
    need(len(signatures) == 1 and signatures != {()}, "MISSING_OR_INCONSISTENT_MM_DEVICE_ROUTE")
    return next(iter(signatures))


def compare_route(layout, stock, opt_in):
    """Require the original Cijk route and changed contraction, not an extra helper."""
    need(stock == (ORIGINAL[layout],), "STOCK_ROUTE_DIFFERS_FROM_T07")
    if opt_in == stock:
        return False
    need(ORIGINAL[layout] not in opt_in, "ALTERNATIVE_RETAINS_ORIGINAL_GEMM")
    # Require one changed contraction from a positive, recognizable vendor family.
    vendor = [name for name in opt_in if
              (name.startswith("Cijk_") and "_ISA936_" in name)
              or any(token in name.lower() for token in ("tensile", "hipblaslt", "rocblas"))]
    need(len(vendor) == 1, "NO_UNAMBIGUOUS_ALTERNATIVE_VENDOR_CONTRACTION")
    return True


def planned_pairs():
    return [dict(layout=layout, pair=idx, preferences=list(order), status="PLANNED", arms=[])
            for layout in LAYOUTS for idx, order in enumerate(ORDER, 1)]


def summarize(rows):
    planned = planned_pairs()
    need(len(rows) == len(planned) and all(
        (actual["layout"], actual["pair"], actual["preferences"]) ==
        (expected["layout"], expected["pair"], expected["preferences"])
        for actual, expected in zip(rows, planned)), "INVALID_PAIR_DENOMINATOR")
    result = {}
    for layout in LAYOUTS:
        selected = [row for row in rows if row["layout"] == layout]
        ratios = {"event": [], "wall": []}
        for pair in selected:
            if pair["status"] != "COMPLETE":
                continue
            arms = pair["arms"]
            need([arm["preference"] for arm in arms] == pair["preferences"]
                 and all(arm["passed"] and arm["calls"] == arm["bitwise_reference_calls"] == CALLS_PER_ARM
                         for arm in arms),
                 "INCOMPLETE_NUMERICAL_PAIR")
            by_arm = {arm["preference"]: arm for arm in arms}
            for key, field in (("event", "event_ms"), ("wall", "timed_wall_seconds")):
                a, b = by_arm[BASELINE][field], by_arm[ALTERNATIVE][field]
                need(all(math.isfinite(x) and x > 0 for x in (a, b)), "INVALID_PAIR_TIMES")
                ratios[key].append(a / b)
        result[layout] = dict(planned_pairs=len(selected), statuses=dict(Counter(row["status"] for row in selected)),
                              event_throughput_opt_in_over_stock=ratios["event"],
                              wall_throughput_opt_in_over_stock=ratios["wall"],
                              event_throughput_median=statistics.median(ratios["event"]) if len(ratios["event"]) == 3 else None,
                              wall_throughput_median=statistics.median(ratios["wall"]) if len(ratios["wall"]) == 3 else None)
    return result
