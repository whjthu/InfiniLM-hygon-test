"""Execute the fixed one-card Z-Image square-mm preference comparison."""

import argparse
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as spec


REQUEST = "1356dd5feb414f77b46b7bdf43fa502b"
LABEL = "O01-zimage-square-mm-blas-pair-v2"
HIP = Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"


def admission(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(16385)
    spec.need(len(raw) <= 16384 and sha256(raw).hexdigest() == expected, "SCOPE_PIN")
    scope = json.loads(raw)
    node = socket.gethostname().split(".")[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    spec.need(scope["request"] == REQUEST and scope["label"] == LABEL and scope["root_admitted"] is True
              and scope["node"] == node and scope["parent"] == parent == "1103606" and step.isdecimal()
              and re.fullmatch("1103606\\.[0-9]+", scope["pre_step"]) and scope["pre_step"] != parent + "." + step
              and re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", scope["PCI"])
              and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == scope["boot_id"],
              "PHYSICAL_SCOPE_IDENTITY")
    start, end = scope["epoch"], scope["deadline_epoch"]
    spec.need(all(type(x) in (int, float) and math.isfinite(x) for x in (start, end))
              and end == start + 240 and start <= time.time() < end - 30, "ORIGINAL_DEADLINE")
    spec.need(os.environ["SLURM_CPUS_PER_TASK"] == "4" and len(os.sched_getaffinity(0)) >= 4,
              "CPU_BINDING")
    return scope, parent + "." + step


def pci(scope):
    spec.need(HIP.resolve(strict=True) == HIP and HIP.stat().st_size == 12_349_016
              and sha256(HIP.read_bytes()).hexdigest() == HIP_SHA, "HIP_PROVIDER_PIN")
    hip = C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    query = hip.hipDeviceGetPCIBusId
    query.argtypes, query.restype = [C.c_void_p, C.c_int, C.c_int], C.c_int
    address = C.create_string_buffer(32)
    spec.need(query(address, len(address), 0) == 0, "HIP_PCI_UNAVAILABLE")
    got = address.value.decode("ascii").lower()
    if len(got) == 7:
        got = "0000:" + got
    spec.need(got == scope["PCI"], "HIP_PHYSICAL_CARD_DIFFERS")
    return got


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def guarded(torch, shape, stride, reference=None):
    count = math.prod(shape)
    base = torch.full((count + 2 * spec.GUARD,), spec.GUARD_VALUE, device="cuda", dtype=torch.bfloat16)
    view = base.as_strided(shape, stride, storage_offset=spec.GUARD)
    if reference is not None:
        view.copy_(reference)
    spec.need(tuple(view.shape) == shape and tuple(view.stride()) == stride and view.storage_offset() == spec.GUARD,
              "ACTUAL_GUARDED_LAYOUT")
    return SimpleNamespace(base=base, view=view, reference=reference)


def check_guard(buf):
    spec.need(bool((buf.base[:spec.GUARD] == spec.GUARD_VALUE).all().item())
              and bool((buf.base[-spec.GUARD:] == spec.GUARD_VALUE).all().item()), "REDZONE_CHANGED")


def same_bits(torch, actual, reference):
    return (actual.dtype == torch.bfloat16 and tuple(actual.shape) == tuple(reference.shape)
            and torch.equal(actual.contiguous().view(torch.int16), reference.contiguous().view(torch.int16)))


def fixed_inputs(torch):
    rows = []
    for seed in spec.SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lhs = torch.randint(-8, 9, (spec.M, spec.K), generator=generator, dtype=torch.int32).float().div_(32)
        rhs = torch.randint(-8, 9, (spec.K, spec.N), generator=generator, dtype=torch.int32).float().div_(32)
        # Inputs and products are exact binary fractions. |sum integer products| <= 3840*64 < 2**24.
        expected = torch.mm(lhs, rhs).to(torch.bfloat16)
        spec.need(bool(torch.isfinite(expected).all().item()), "CPU_REFERENCE_NONFINITE")
        rows.append((lhs.to(torch.bfloat16), rhs.to(torch.bfloat16), expected))
    return rows


def open_layout(torch, cpu, stride):
    return [(guarded(torch, (spec.M, spec.K), (spec.K, 1), lhs),
             guarded(torch, (spec.K, spec.N), stride, rhs), reference.to("cuda")) for lhs, rhs, reference in cpu]


def audit_inputs(torch, rows):
    for a, b, _ in rows:
        for buffer in (a, b):
            check_guard(buffer)
            spec.need(same_bits(torch, buffer.view.cpu(), buffer.reference), "INPUT_PAYLOAD_CHANGED")


def observe(torch, rows, *, output=None):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=False, with_stack=False) as profile:
        if output is None:
            values = [torch.mm(a.view, b.view) for a, b, _ in rows]
        else:
            values = [torch.mm(a.view, b.view, out=buf.view) for (a, b, _), buf in zip(rows, output)]
        torch.cuda.synchronize()
    signatures = spec.kernels(profile, len(rows))
    spec.need(all(same_bits(torch, value, row[2]) for value, row in zip(values, rows)), "PROFILED_OUTPUT_DIFFERENCE")
    return signatures


def measure(torch, rows, ledger, check):
    stream = torch.cuda.current_stream()
    before, after = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    before.record(stream)
    after.record(stream)
    after.synchronize()
    for _ in range(4):
        for a, b, _ in rows:
            torch.mm(a.view, b.view)
    stream.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for batch in range(spec.BATCHES):
        check()
        a, b, expected = rows[batch % len(rows)]
        started = time.monotonic()
        before.record(stream)
        values = [torch.mm(a.view, b.view) for _ in range(spec.CALLS_PER_BATCH)]
        after.record(stream)
        after.synchronize()
        wall = time.monotonic() - started
        elapsed = before.elapsed_time(after)
        ledger["calls"] += len(values)
        spec.need(math.isfinite(elapsed) and elapsed > 0 and math.isfinite(wall) and wall > 0, "INVALID_STREAM_TIME")
        ledger["event_batch_ms"].append(elapsed)
        ledger["wall_batch_seconds"].append(wall)
        for value in values:
            spec.need(same_bits(torch, value, expected), "MEASURED_OUTPUT_DIFFERENCE")
            ledger["bitwise_reference_calls"] += 1
    ledger.update(event_ms=sum(ledger["event_batch_ms"]), timed_wall_seconds=sum(ledger["wall_batch_seconds"]),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())


def exercise(torch, scope, report, check, persist):
    torch.set_num_threads(4)
    spec.need(torch.version.hip == scope["torch_hip"] and torch.__version__ == scope["torch_version"]
              and torch.cuda.device_count() == 1, "TORCH_RUNTIME_OR_LEASE")
    torch.cuda.set_device(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    spec.need(arch == "gfx936", "GFX936_TARGET_REQUIRED")
    report.update(PCI=pci(scope), gfx=arch, torch={"version": torch.__version__, "hip": torch.version.hip})
    backend = torch.backends.cuda
    preference = backend.preferred_blas_library
    inherited = preference()
    report["inherited_effective_preference"] = str(inherited)

    def choose(arm, *, allow_alternative_fallback=False):
        selected = backend._BlasBackends[arm]
        returned = preference(arm)
        observed = preference()
        if returned == observed == selected:
            return True, str(observed)
        fallback = (allow_alternative_fallback and arm == spec.ALTERNATIVE
                    and returned == observed == backend._BlasBackends[spec.BASELINE])
        spec.need(fallback, "BLAS_PREFERENCE_READBACK")
        return False, str(observed)

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            host = fixed_inputs(torch)
            for layout, stride in spec.LAYOUTS.items():
                check()
                rows = open_layout(torch, host, stride)
                audit_inputs(torch, rows)
                outputs = [guarded(torch, (spec.M, spec.N), (spec.N, 1)) for _ in rows]
                expected_routes = {}
                selected = [pair for pair in report["pairs"] if pair["layout"] == layout]
                readback = report["preference_readback"].setdefault(layout, {})
                for arm in (spec.BASELINE, spec.ALTERNATIVE):
                    effective_matches, effective = choose(arm, allow_alternative_fallback=True)
                    readback[arm] = dict(effective=effective, requested_enum_observed=effective_matches)
                    persist()
                    if not effective_matches:
                        for pair in selected:
                            pair["status"] = "ALTERNATIVE_PREFERENCE_FELL_BACK"
                        persist()
                        break
                    signature = observe(torch, rows)
                    expected_routes[arm] = signature
                    report["routes"][layout] = expected_routes
                    persist()
                    if arm == spec.BASELINE:
                        spec.need(signature == (spec.ORIGINAL[layout],), "STOCK_ROUTE_DIFFERS_FROM_T07")
                    guarded_signature = observe(torch, rows, output=outputs)
                    spec.need(guarded_signature == signature, "GUARDED_OUTPUT_ROUTE_DIFFERS")
                    for buf in outputs:
                        check_guard(buf)
                if spec.ALTERNATIVE not in expected_routes:
                    continue
                if not spec.compare_route(layout, expected_routes[spec.BASELINE], expected_routes[spec.ALTERNATIVE]):
                    for pair in selected:
                        pair["status"] = "SAME_VENDOR_KERNEL"
                    persist()
                    continue
                for pair in selected:
                    pair["status"] = "IN_PROGRESS"
                    for arm in pair["preferences"]:
                        check()
                        choose(arm)
                        row = dict(preference=arm, calls=0, bitwise_reference_calls=0, event_batch_ms=[],
                                   wall_batch_seconds=[], passed=False)
                        pair["arms"].append(row)
                        row["pre_route"] = observe(torch, rows)
                        spec.need(row["pre_route"] == expected_routes[arm], "PRE_WINDOW_ROUTE_DRIFT")
                        measure(torch, rows, row, check)
                        row["post_route"] = observe(torch, rows)
                        spec.need(row["post_route"] == expected_routes[arm], "POST_WINDOW_ROUTE_DRIFT")
                        audit_inputs(torch, rows)
                        row["passed"] = True
                        persist()
                    pair["status"] = "COMPLETE"
                    persist()
    finally:
        preference(inherited)
        restored = preference()
        report["restored_effective_preference"] = str(restored)
        spec.need(restored == inherited, "BLAS_PREFERENCE_NOT_RESTORED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scope, step = admission(args.scope, args.scope_sha256)
    args.out.mkdir(mode=0o700)
    report = dict(request=REQUEST, step=step, node=scope["node"], scope_SHA256=args.scope_sha256,
                  source_step="1103606.1787", operand_receipt_SHA256=spec.OPERAND_RECEIPT_SHA256,
                  input_kind="controlled_synthetic_BF16", generated_reference="independent exact-dot FP32 CPU",
                  planned_pairs=6, planned_arms=12, planned_timed_calls=12 * spec.CALLS_PER_ARM,
                  routes={}, preference_readback={}, pairs=spec.planned_pairs(), completed=False, passed=False,
                  provider_DSO_inferred_from_kernel=False, training_performance=False)

    def persist():
        write(args.out / "terminal.json", report)

    def check():
        spec.need(time.time() < scope["deadline_epoch"] - 5, "ORIGINAL_WORK_BUDGET")

    def timeout(*_):
        raise TimeoutError("ORIGINAL_WORK_ALARM")

    persist()
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1, scope["deadline_epoch"] - time.time() - 3))
    try:
        spec.need(not any(os.environ.get(key) not in (None, "", "0") for key in
                          ("ROCBLAS_LAYER", "HIPBLASLT_LOG_LEVEL", "HIPBLASLT_LOG_MASK")),
                  "BLAS_LOGGING_ACTIVE_IN_PAIRED_WINDOWS")
        import torch
        exercise(torch, scope, report, check, persist)
        report["result"] = spec.summarize(report["pairs"])
        report["passed"] = True
    except Exception as error:
        report.update(error_type=type(error).__name__,
                      error_summary=str(error) if re.fullmatch("[A-Z0-9_]{1,96}", str(error)) else None)
        for pair in report["pairs"]:
            if pair["status"] == "IN_PROGRESS":
                pair["status"] = "FAILED"
        report["result"] = spec.summarize(report["pairs"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["completed"] = True
        persist()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
