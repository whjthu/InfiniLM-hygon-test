"""Run two route witnesses, then six stock BLAS fixed-reuse device arms."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REQUEST = "a31421758fc74d319923dbce7045cc12"
LABEL = "O01-zimage-square-rhs-pack16-v1"
MEMBERS = {"baseline.py", "contract.py", "hipprof_trace.py", "candidate.py", "worker.py"}


def pinned(path, sha, cap):
    if path.is_symlink() or not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("RUNTIME_SOURCE_PIN")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    if not 0 < len(raw) <= cap or sha256(raw).hexdigest() != sha:
        raise ValueError("RUNTIME_SOURCE_BYTES")
    return raw


def bind_runtime(args):
    scope = json.loads(pinned(args.scope, args.scope_sha256, 16384))
    manifest = json.loads(pinned(HERE / "SCOPED_FILES.json", scope["runtime_SHA256"], 8192))
    files = manifest["files"]
    if (manifest["request"] != REQUEST or manifest["label"] != LABEL or set(files) != MEMBERS
            or manifest["provider_preference"] != "cublas"
            or not re.fullmatch("[0-9a-f]{40}", scope["source_publication"])):
        raise ValueError("RUNTIME_SCOPE_IDENTITY")
    for name, pin in files.items():
        raw = pinned(HERE / name, pin["SHA256"], 65536)
        if len(raw) != pin["bytes"]:
            raise ValueError("RUNTIME_SOURCE_SIZE")
    import baseline
    import candidate
    baseline.REQUEST, baseline.LABEL = REQUEST, LABEL
    # Its prior audited tracer wrapper must invoke this worker's child entrypoint.
    baseline.__file__ = str(Path(__file__).resolve(strict=True))
    scope, step = baseline.admission(args.scope, args.scope_sha256)
    candidate.require(scope["node"] != "f12r2n11" and len(os.sched_getaffinity(0)) == 4,
                      "O01_PROTECTED_OR_CPU_SCOPE")
    return scope, step, manifest, baseline, candidate


def require_route(base, candidate, scope, args, manifest, report, persist):
    candidate.require(base.PROFILER.resolve(strict=True) == base.PROFILER
                      and len(pinned(base.PROFILER, base.PROFILER_SHA256, 15_528_344)) == 15_528_344, "HIPPROF_ELF")
    expected = manifest["stock_kernel_by_arm"]
    candidate.require(set(expected) == set(candidate.LAYOUTS)
                      and expected[candidate.DIRECT] == base.spec.ORIGINAL["transposed"]
                      and expected[candidate.PACKED] == base.spec.ORIGINAL["contiguous"], "PINNED_CIJK_ROUTES")
    for layout, entry in report["routes"].items():
        entry["state"] = "ATTEMPTED"
        persist()
        witness = base.trace_cell(scope, args.scope, args.scope_sha256, args.out / layout, layout, "cublas")
        entry.update(witness)
        candidate.require(witness["tool_exit"] == 0 and witness["unique_contraction"] == expected[layout]
                          and witness["counts"]["complete_HIPOPS"] == witness["counts"]["attached"] == 1,
                          "ORIGINAL_ROUTE_OR_TRACE_MISSING")
        entry["state"] = "QUALIFIED"
        persist()


def torch_callers(scope, base, candidate):
    import torch
    torch.set_num_threads(4)
    candidate.require(torch.version.hip == scope["torch_hip"] == "6.3.26113"
                      and torch.__version__ == scope["torch_version"] == "2.10.0"
                      and torch.cuda.device_count() == 1, "TORCH_HIP_OR_LEASE")
    torch.cuda.set_device(0)
    candidate.require(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx936",
                      "GFX936_TARGET_REQUIRED")
    base.pci(scope)
    choose = torch.backends.cuda.preferred_blas_library
    stock = torch.backends.cuda._BlasBackends["cublas"]
    candidate.require(choose("cublas") == choose() == stock, "ORIGINAL_PROVIDER_PREFERENCE")
    return torch


def run_pairs(scope, base, candidate, report, persist):
    torch = torch_callers(scope, base, candidate)

    def deadline():
        candidate.require(time.time() < scope["deadline_epoch"] - 6, "ABSOLUTE_GPU_DEADLINE")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        deadline()
        buffers = candidate.inputs(torch, base.guarded)
        stream = torch.cuda.current_stream()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for event in events:
            event.record(stream)
        stream.synchronize()
        for pair in report["pairs"]:
            pair["status"] = "PARTIAL"
            for layout in pair["order"]:
                arm = {"layout": layout, "bursts": [], "verified_warm_calls": 0, "verified_measured_calls": 0}
                pair["arms"].append(arm)
                persist()
                torch.cuda.reset_peak_memory_stats()
                for i in range(candidate.WARM_BURSTS + candidate.BURSTS):
                    deadline()
                    data, values, order = candidate.burst(torch, buffers, layout, events, stream,
                                                          pair["pair"] + i)
                    measured = i >= candidate.WARM_BURSTS
                    if measured:
                        # Preserve the attempted burst and timings before the numerical gate.
                        arm["bursts"].append(data)
                        persist()
                    candidate.check_values(torch, base, values, order, *buffers, layout)
                    key = "verified_measured_calls" if measured else "verified_warm_calls"
                    arm[key] += len(values)
                    del values
                    persist()
                arm.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            pair["status"] = "COMPLETE"
            persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", nargs=2)
    args = parser.parse_args()
    scope, step, manifest, base, candidate = bind_runtime(args)
    if args.cell:
        layout, preference = args.cell
        candidate.require(layout in candidate.LAYOUTS and preference == "cublas", "WITNESS_ARM")
        return base.witness(scope, args.out, layout, preference)
    args.out.mkdir(mode=0o700)
    report = {"request": REQUEST, "scope_SHA256": args.scope_sha256, "step": step, "PCI": scope["PCI"],
              "node": scope["node"], "routes": {layout: {"state": "PLANNED"} for layout in candidate.LAYOUTS},
              "pairs": candidate.pairs(), "source_training_step": "1103606.1787",
              "same_rhs_reuse_per_burst": candidate.CALLS, "packed_RHS_payload_bytes": candidate.PACK_BYTES,
              "planned_extra_pack_guard_bytes": 4 * base.spec.GUARD,
              "copy_logical_read_plus_write_bytes": candidate.PACK_LOGICAL_TRAFFIC_BYTES,
              "training_RHS_reuse_identity_verified": False,
              "warm_bursts_per_arm": candidate.WARM_BURSTS, "measured_bursts_per_arm": candidate.BURSTS,
              "complete": False, "passed": False, "training_performance": False}

    def persist():
        base.write(args.out / "terminal.json", report)

    try:
        require_route(base, candidate, scope, args, manifest, report, persist)
        run_pairs(scope, base, candidate, report, persist)
        report["passed"] = True
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        message = str(error)
        report["failure"] = {"type": type(error).__name__,
                             "gate": message if re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", message) else None}
    finally:
        report["summary"] = candidate.summarize(report["pairs"])
        report["complete"] = True
        persist()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
