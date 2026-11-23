"""Prove actual Z-Image square-mm routing in isolated hipprof windows."""

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
import subprocess
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as spec
import hipprof_trace as trace


REQUEST = "5a37baafb61f4979829cb8fa27e7489f"
LABEL = "O01-zimage-square-mm-hipprof-route-v1"
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



PROFILER = Path("/opt/dtk/hip/bin/hipprof")
PROFILER_SHA256 = "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"
CELL_SECONDS = 30


def check_matmul(torch, value, expected, a, b):
    spec.need(same_bits(torch, value, expected), "WITNESS_OUTPUT_DIFFERS")
    audit_inputs(torch, [(a, b, expected)])


def witness(scope, out, layout, arm):
    import torch

    torch.set_num_threads(4)
    spec.need(torch.version.hip == scope["torch_hip"] and torch.__version__ == scope["torch_version"]
              and torch.cuda.device_count() == 1, "TORCH_RUNTIME_OR_LEASE")
    torch.cuda.set_device(0)
    spec.need(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx936", "NOT_GFX936")
    physical = pci(scope)
    pref = torch.backends.cuda.preferred_blas_library
    choice = torch.backends.cuda._BlasBackends[arm]
    returned, effective = pref(arm), pref()
    evidence = dict(pid=os.getpid(), physical_PCI=physical, trace_device_index=0, layout=layout, arm=arm,
                    effective=str(effective), requested_preference_verified=returned == effective == choice,
                    complete=False, timed_calls=0)
    if not evidence["requested_preference_verified"]:
        write(out / "witness.json", evidence)
        return 3
    a, b, reference = open_layout(torch, fixed_inputs(torch)[:1], spec.LAYOUTS[layout])[0]
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        warm = torch.mm(a.view, b.view)
        torch.cuda.synchronize()
        check_matmul(torch, warm, reference, a, b)
        # Resolve these from the live traced process. No additional lib or preload is inserted.
        hip = C.CDLL(None)
        hip.hipProfilerStart.argtypes = hip.hipProfilerStop.argtypes = []
        hip.hipProfilerStart.restype = hip.hipProfilerStop.restype = C.c_int
        torch.cuda.synchronize()
        start = time.time_ns()
        start_rc = hip.hipProfilerStart()
        spec.need(start_rc == 0, "HIPPROF_START_FAILED")
        try:
            value = torch.mm(a.view, b.view)
            torch.cuda.synchronize()
        finally:
            stop_rc = hip.hipProfilerStop()
            end = time.time_ns()
        check_matmul(torch, value, reference, a, b)
    evidence.update(start_rc=start_rc, stop_rc=stop_rc, realtime_ns=[start, end],
                    bitwise_CPU_reference=True, profiled_mm_calls=1, complete=stop_rc == 0)
    write(out / "witness.json", evidence)
    return int(stop_rc != 0)


def size(directory):
    paths = tuple(directory.iterdir())
    spec.need(len(paths) <= 16 and all(path.is_file() and not path.is_symlink() for path in paths),
              "UNEXPECTED_TRACER_OUTPUT_SCOPE")
    return sum(path.stat().st_size for path in paths)


def trace_cell(scope, scope_path, scope_pin, cell, layout, arm):
    spec.need(time.time() < scope["deadline_epoch"] - CELL_SECONDS - 12, "TRACE_WORK_BUDGET")
    cell.mkdir(mode=0o700)
    argv = [str(PROFILER), "--hip-trace", "--trace-off", "--show-pid", "--output-type", "0",
            "-d", str(cell), "-o", str(cell / "trace"), sys.executable, "-B", __file__,
            "--scope", str(scope_path), "--scope-sha256", scope_pin, "--out", str(cell),
            "--cell", layout, arm]
    deadline = time.monotonic() + CELL_SECONDS
    with (cell / "tool.stdout").open("xb") as stdout, (cell / "tool.stderr").open("xb") as stderr:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                spec.need(time.monotonic() < deadline, "TRACE_CELL_TIMEOUT")
                spec.need(size(cell) <= trace.MAX_TRACE_BYTES, "TRACE_DIRECTORY_BUDGET")
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=2)
    spec.need(size(cell) <= trace.MAX_TRACE_BYTES, "TRACE_DIRECTORY_BUDGET")
    receipt = cell / "witness.json"
    row = {"tool_exit": child.returncode, "observed": receipt.is_file(), "unique_contraction": None}
    for log in ("tool.stdout", "tool.stderr"):
        raw = (cell / log).read_bytes()
        row[log] = dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())
    if receipt.is_file():
        raw = receipt.read_bytes()
        spec.need(0 < len(raw) <= 8192, "WITNESS_RECEIPT_SIZE")
        own = json.loads(raw)
        spec.need((own["physical_PCI"], own["layout"], own["arm"]) == (scope["PCI"], layout, arm),
                  "WITNESS_RECEIPT_IDENTITY")
        row["witness_SHA256"] = sha256(raw).hexdigest()
        row["witness"] = own
        if child.returncode == 0 and own["complete"]:
            spec.need(own["start_rc"] == own["stop_rc"] == 0 and own["profiled_mm_calls"] == 1
                      and own["bitwise_CPU_reference"] and own["requested_preference_verified"],
                      "WITNESS_RECEIPT_INVALID")
            row.update(trace.read_trace(cell / "trace.json", own))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", nargs=2)
    args = parser.parse_args()
    scope, step = admission(args.scope, args.scope_sha256)
    if args.cell:
        layout, arm = args.cell
        spec.need(layout in spec.LAYOUTS and arm in (spec.BASELINE, spec.ALTERNATIVE), "CELL_IDENTITY")
        return witness(scope, args.out, layout, arm)
    args.out.mkdir(mode=0o700)
    report = dict(request=REQUEST, step=step, scope_SHA256=args.scope_sha256, node=scope["node"], PCI=scope["PCI"],
                  source_step="1103606.1787", previous_Kineto_step="1103606.1861",
                  GPU_profiler_elf_CPU_step="1103606.1866", planned_cells=4, routes={}, timed_calls=0,
                  training_performance=False, completed=False, passed=False)

    def persist():
        write(args.out / "terminal.json", report)

    try:
        spec.need(PROFILER.resolve(strict=True) == PROFILER and PROFILER.stat().st_size == 15_528_344
                  and sha256(PROFILER.read_bytes()).hexdigest() == PROFILER_SHA256, "HIPPROF_ELF_DRIFT")
        for layout in spec.LAYOUTS:
            rows = report["routes"].setdefault(layout, {})
            for arm in (spec.BASELINE, spec.ALTERNATIVE):
                try:
                    rows[arm] = trace_cell(scope, args.scope, args.scope_sha256, args.out / f"{layout}-{arm}", layout, arm)
                except (OSError, ValueError, KeyError) as error:
                    row = str(error)
                    rows[arm] = {"error_type": type(error).__name__,
                                 "error_summary": row if re.fullmatch(r"[a-zA-Z0-9_ .-]{1,100}", row) else None,
                                 "unique_contraction": None}
                persist()
        report["layout_conclusion"] = {layout: trace.assess(rows, layout, spec.ORIGINAL, spec.BASELINE, spec.ALTERNATIVE)
                                        for layout, rows in report["routes"].items()}
        report["passed"] = all(value != "UNQUALIFIED" for value in report["layout_conclusion"].values())
    finally:
        report["completed"] = True
        persist()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
