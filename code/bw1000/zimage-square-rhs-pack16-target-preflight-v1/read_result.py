"""Audit an original pack16 result after independently captured Slurm terminal accounting."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import re
import statistics
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
import precheck as pins


LABEL = "O01-zimage-square-rhs-pack16-v1"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "DEADLINE", "BOOT_FAIL", "REVOKED"}
ORDER = (("transposed", "packed"), ("packed", "transposed"), ("transposed", "packed"))
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


def require(test, gate):
    if not test:
        raise ValueError(gate)


def doc(path, cap, expected=None):
    raw = pins.read(path, cap, expected)
    return json.loads(raw), pins.pin(raw)


def actual_step(accounting, step, node):
    require(accounting["accounting_columns"] == COLUMNS and
            all(len(row) == len(COLUMNS) for row in accounting["accounting"]), "ACCOUNTING_COLUMNS")
    matches = [row for row in accounting["accounting"] if row[1] == LABEL]
    require(len(matches) == 1 and matches[0][0] == step and re.fullmatch(r"1103606\.[0-9]+", step),
            "GPU_STEP_NOT_UNIQUE")
    [row] = matches
    tres = row[6].split(",")
    require(row[5] == node and len(tres) == len(set(tres)) and
            set(tres) in ({"cpu=4", "gres/dcu=1", "mem=8G", "node=1"},
                          {"cpu=4", "gres/dcu=1", "mem=8192M", "node=1"}), "ACTUAL_GPU_RESOURCES")
    require(row[4].isdecimal() and re.fullmatch(r"[0-9]+:[0-9]+", row[3]), "ACCOUNTING_ROW_SHAPE")
    return row


def statistics_for(plan):
    require(isinstance(plan, list) and len(plan) == 3, "PAIR_DENOMINATOR")
    totals = {key: [] for key in ("wall_seconds", "total_event_ms", "GEMM_event_ms")}
    verified, copies = 0, 0
    peak = []
    for i, (pair, order) in enumerate(zip(plan, ORDER, strict=True), 1):
        require(pair["pair"] == i and pair["order"] == list(order)
                and pair["status"] in {"PLANNED", "PARTIAL", "COMPLETE"}, "PAIR_IDENTITY")
        arms = pair["arms"]
        require([arm["layout"] for arm in arms] == list(order[:len(arms)])
                and len(arms) <= 2 and (pair["status"] != "PLANNED" or not arms), "PAIR_ARMS")
        for arm in arms:
            bursts = arm["bursts"]
            requested_copy = int(arm["layout"] == "packed")
            calls = arm["verified_measured_calls"]
            warm = arm["verified_warm_calls"]
            require(type(calls) is type(warm) is int and 0 <= calls <= 16 * len(bursts) <= 64
                    and 0 <= warm <= 32 and calls % 16 == warm % 16 == 0, "MEASURED_CALL_COUNTS")
            for batch in bursts:
                require(type(batch["copies"]) is int and batch["copies"] == requested_copy
                        and all(type(batch[key]) in (int, float) and math.isfinite(batch[key]) and batch[key] > 0 for key in totals)
                        and type(batch["first_interval_ms"]) in (int, float)
                        and math.isfinite(batch["first_interval_ms"]) and batch["first_interval_ms"] >= 0,
                        "BURST_TIMING_OR_COPIES")
                copies += batch["copies"]
            verified += calls
        if pair["status"] == "COMPLETE":
            require(len(arms) == 2 and all(len(arm["bursts"]) == 4 and arm["verified_measured_calls"] == 64
                    and arm["verified_warm_calls"] == 32 for arm in arms), "COMPLETE_PAIR_COUNTS")
            by_layout = {arm["layout"]: arm for arm in arms}
            for arm in arms:
                alloc, reserved = arm["peak_allocated_bytes"], arm["peak_reserved_bytes"]
                require(type(alloc) is type(reserved) is int and 0 < alloc <= reserved, "PEAK_MEMORY_EVIDENCE")
                peak.append({"pair": i, "arm": arm["layout"], "allocated_bytes": alloc, "reserved_bytes": reserved})
            for key, ratios in totals.items():
                measured = [sum(v[key] for v in by_layout[layout]["bursts"]) for layout in ("transposed", "packed")]
                ratios.append(measured[0] / measured[1])
    median = {key: statistics.median(values) if len(values) == 3 else None for key, values in totals.items()}
    return {"pairs_planned": len(plan), "status_counts": dict(Counter(row["status"] for row in plan)),
            "verified_mm_calls": verified, "measured_copy_receipts": copies,
            "wall_speedup": totals["wall_seconds"], "medians": median, "arm_allocator_peaks": peak,
            "arithmetic_performance_gate": verified == 384 and copies == 12 and median["wall_seconds"] is not None
            and median["wall_seconds"] >= 1.05 and min(totals["wall_seconds"]) >= 1.0}


def trace_module(runtime):
    mod = spec_from_file_location("pack16_verified_hipprof", runtime / "hipprof_trace.py")
    loader = module_from_spec(mod)
    mod.loader.exec_module(loader)
    return loader


def audit_routes(runtime, work, scope, report):
    planned, _ = doc(runtime / "SCOPED_FILES.json", 8192, pins.RUNTIME_SHA)
    trace = trace_module(runtime)
    require(set(report["routes"]) == set(planned["stock_kernel_by_arm"]), "ROUTE_ARMS")
    verified = {}
    for arm, symbol in planned["stock_kernel_by_arm"].items():
        archived = report["routes"][arm]
        if archived["state"] != "QUALIFIED":
            continue
        route_dir = work / arm
        witness, pin = doc(route_dir / "witness.json", 8192, archived["witness_SHA256"])
        require(witness == archived["witness"] and witness["layout"] == arm and witness["arm"] == "cublas"
                and witness["physical_PCI"] == scope["PCI"] and witness["trace_device_index"] == 0
                and witness["complete"] is True and witness["profiled_mm_calls"] == 1
                and witness["start_rc"] == witness["stop_rc"] == 0
                and witness["bitwise_CPU_reference"] is witness["requested_preference_verified"] is True,
                "ROUTE_WITNESS_IDENTITY")
        result = trace.read_trace(route_dir / "trace.json", witness)
        require(archived["tool_exit"] == 0 and result["unique_contraction"] == symbol == archived["unique_contraction"]
                and result["raw_trace_SHA256"] == archived["raw_trace_SHA256"]
                and result["counts"] == archived["counts"]
                and result["counts"]["complete_HIPOPS"] == result["counts"]["attached"] == 1, "ACTUAL_ROUTE_OR_TRACE")
        verified[arm] = {"symbol": symbol, "witness_SHA256": pin["SHA256"], "trace_SHA256": result["raw_trace_SHA256"]}
    return verified


def read_result(run, runtime, scope_path, scope_sha, account_path, account_sha, step):
    scope, scope_pin = doc(scope_path, 16384, scope_sha)
    node = scope["node"]
    require(scope["request"] == pins.GPU_REQUEST and scope["label"] == LABEL and scope["parent"] == pins.PARENT
            and scope["source_publication"] == pins.GPU_PUBLICATION and scope["runtime_SHA256"] == pins.RUNTIME_SHA
            and scope["root_admitted"] is True and node != "f12r2n11" and PCI.fullmatch(scope["PCI"])
            and scope["deadline_epoch"] == scope["epoch"] + 240, "GPU_SCOPE")
    account, account_pin = doc(account_path, 131072, account_sha)
    row = actual_step(account, step, node)
    ended = row[2].split()[0].rstrip("+") in TERMINAL
    result = {"request": pins.GPU_REQUEST, "actual_accounting": row,
              "scope_SHA256": scope_pin["SHA256"], "accounting_SHA256": account_pin["SHA256"],
              "real_GPU_terminal": ended, "worker_result_passed": False,
              "component_timing_candidate": False, "promotion_allowed": False,
              "independent_post": "UNVERIFIED", "training_performance": False}
    if not ended or not (run / "terminal.json").is_file():
        return result
    try:
        pins.runtime(runtime)
        report, source = doc(run / "terminal.json", 262144)
        result["worker_terminal_SHA256"] = source["SHA256"]
        require(report["request"] == pins.GPU_REQUEST and report["step"] == step and report["node"] == node
                and report["PCI"] == scope["PCI"] and report["scope_SHA256"] == scope_sha
                and report["source_training_step"] == "1103606.1787" and report["training_performance"] is False
                and report["training_RHS_reuse_identity_verified"] is False
                and report["same_rhs_reuse_per_burst"] == 16 and report["packed_RHS_payload_bytes"] == 29_491_200
                and report["planned_extra_pack_guard_bytes"] == 1024, "WORKER_TERMINAL_IDENTITY")
        result["pairs"] = statistics_for(report["pairs"])
        result["routes"] = audit_routes(runtime, run, scope, report)
        result["worker_result_passed"] = (row[2:4] == ["COMPLETED", "0:0"] and report["complete"] is report["passed"] is True
                                            and len(result["routes"]) == 2 and result["pairs"]["verified_mm_calls"] == 384
                                            and result["pairs"]["measured_copy_receipts"] == 12)
        result["component_timing_candidate"] = result["worker_result_passed"] and result["pairs"]["arithmetic_performance_gate"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        text = str(exc)
        result["reader_failure"] = {"type": type(exc).__name__,
                                    "gate": text if re.fullmatch("[A-Z][A-Z0-9_]+", text) else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("run", "runtime", "scope", "accounting"):
        parser.add_argument("--" + role, type=Path, required=True)
    for role in ("scope-sha256", "accounting-sha256", "gpu-step"):
        parser.add_argument("--" + role, required=True)
    args = parser.parse_args()
    row = read_result(args.run, args.runtime, args.scope, args.scope_sha256,
                      args.accounting, args.accounting_sha256, args.gpu_step)
    print(json.dumps(row, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
