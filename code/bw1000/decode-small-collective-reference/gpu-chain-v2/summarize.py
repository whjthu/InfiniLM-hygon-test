"""Summarize an entire raw collective step without mixing rank or iteration clocks."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from driver import DEEP_CHAINS, DEVICES, SAMPLES, schedule, validate_chain_report, validate_report


def summarize(packets, ranks, step, native_sha256, *, deep_chain=False):
    if deep_chain and ranks != 8:
        raise ValueError("Deep chain requires TP8")
    points = DEEP_CHAINS if deep_chain else schedule(ranks)
    nodes = ranks // DEVICES
    if not re.fullmatch(r"[0-9]+\.[0-9]+", step):
        raise ValueError("expected one numeric Slurm job.step")
    if not re.fullmatch(r"[a-f0-9]{64}", native_sha256):
        raise ValueError("expected an exact Native SHA256")
    if len(packets) != nodes:
        raise ValueError("incomplete node census")

    by_rank = {}
    seen_nodes = set()
    hosts = set()
    for packet in packets:
        if packet["actual_step"] != step or packet["native_sha256"] != native_sha256:
            raise ValueError("node report provenance mismatch")
        host = packet["host"]
        report = packet["report"]
        node = report["node"]
        if (type(node) is not int or node not in range(nodes) or node in seen_nodes
                or not isinstance(host, str) or not host.strip() or host in hosts):
            raise ValueError("invalid or duplicated node identity")
        seen_nodes.add(node)
        hosts.add(host)
        if report["TP"] != ranks or report["passed"] is not True:
            raise ValueError("node did not finish successfully")
        owners = report["owners"]
        if len(owners) != DEVICES:
            raise ValueError("incomplete device census")
        for owner in owners:
            device, rank = owner["device"], owner["rank"]
            if (type(device) is not int or device not in range(DEVICES)
                    or type(rank) is not int or rank != node * DEVICES + device
                    or rank in by_rank):
                raise ValueError("invalid or duplicated rank/device")
            if (owner["close_succeeded"] is not True or owner["error"] is not None
                    or owner["close_error"] is not None):
                raise ValueError("owner did not complete Native close")
            by_rank[rank] = owner["points"]
            for point, result in zip(points, owner["points"], strict=True):
                if deep_chain:
                    validate_chain_report(result, rank, *point)
                else:
                    validate_report(result, ranks, rank, point)

    rows = []
    for index, point in enumerate(points):
        samples = [by_rank[rank][index]["samples"] for rank in range(ranks)]
        clocks = {}
        submit_clock = "enqueue_phase_wall_ms" if deep_chain else "host_enqueue_ms"
        for clock in ("stream_elapsed_ms", submit_clock, "wall_ms"):
            maxima = [max(series[i][clock] for series in samples)
                      for i in range(3, 3 + SAMPLES)]
            clocks[clock] = {
                "matched_iteration_max_ms": maxima,
                "median_ms": statistics.median(maxima),
                "min_ms": min(maxima),
                "max_ms": max(maxima),
            }
        if deep_chain:
            past, fence_each = point
            rows.append({"past": past, "arm": index % 4, "timed_fences": 107 if fence_each else 1,
                         "calls_per_rank": 107, "clocks": clocks})
        else:
            dtype, size, inplace = point
            rows.append({"dtype": {9: "BF16", 10: "F32"}[dtype], "bytes_per_rank": size,
                         "inplace": inplace, "clocks": clocks})
    return {
        "scope": ("raw Deep r467 107-AllReduce chain with one versus 107 waits"
                  if deep_chain else "raw AllReduce component with full per-rank numerical checks"),
        "actual_step": step, "native_sha256": native_sha256, "TP": ranks,
        "hosts": sorted(hosts), "owners_native_close_succeeded": ranks,
        "warmup_iterations_excluded": 3, "measured_iterations_per_point": SAMPLES,
        "model_performance_inferred": False, "independent_cleanup_verified": False,
        "clock_scope": "Max of matching rank-local durations. Includes arrival/wait and "
                       "does not reconstruct a global timeline or exclusive device time.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(8, 16), required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--deep-chain", action="store_true")
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("node_reports", nargs="+", type=Path)
    args = parser.parse_args()
    raw = [path.read_bytes() for path in args.node_reports]
    result = summarize([json.loads(data) for data in raw], args.tp, args.step, args.native_sha256,
                       deep_chain=args.deep_chain)
    result["node_report_sha256"] = sorted(hashlib.sha256(data).hexdigest() for data in raw)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
