"""Check four bounded native hipBLASLt solutions in the original transposed Z-Image geometry."""

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


REQUEST = "4241e3d58506499cb859abead6f937de"
LABEL = "O01-zimage-transposed-lt-unseen-solution-v1"
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




WORKSPACE = 32 * 1024 * 1024
MAX_SOLUTIONS = 4
PROBE = Path("/pair-build/libzimage_lt_probe.so")
PROBE_SHA = "f735cc99e10d0c958b3736fde852bbf9e1f9747827225f1375f0214726279a07"
HIPBLAS = Path("/opt/dtk/lib/libhipblaslt.so.0.10")
HIPBLAS_SHA = "03c8f72e2d1a8172127e98cfd41040631e69a7f92821879387cf31df25b0549d"
TRACER = Path("/opt/dtk/hip/bin/hipprof")
TRACER_SHA = "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"
ORIGINAL_LT = "Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_MT256x256x16_MI16x16x1_SN_GSU1_K1_LRVW2_MIWT8_4_WGM5"
TRACE_LIMIT = 16 << 20


class SearchEvidence(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("catalog_count", "examined_count", "support_checks", "selected_rank")]


class Evidence(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("solution", "heuristic", "support", "runnable")]
    _fields_ += [(name, C.c_uint64) for name in ("heuristic_workspace", "supported_workspace")]


def bind(library, name, fields):
    value = getattr(library, name)
    value.argtypes = fields
    value.restype = C.c_int
    return value


def fingerprint(path, pin):
    spec.need(path.resolve(strict=True) == path and sha256(path.read_bytes()).hexdigest() == pin,
              "BINARY_IDENTITY")


def native_probe():
    fingerprint(PROBE, PROBE_SHA)
    spec.need(PROBE.stat().st_size == 17760, "NATIVE_BINARY_SIZE")
    fingerprint(HIPBLAS, HIPBLAS_SHA)
    lib = C.CDLL(str(PROBE), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    with Path("/proc/self/maps").open(encoding="ascii") as stream:
        providers = {Path(line.strip().split()[-1]).resolve(strict=True)
                     for line in stream if "/libhipblaslt.so" in line}
    spec.need(providers == {HIPBLAS} and C.sizeof(Evidence) == 32, "MAPPED_NATIVE_PROVIDER_OR_ABI")
    address = C.c_void_p
    return (bind(lib, "zimage_lt_create", [C.POINTER(address)]),
            bind(lib, "zimage_lt_enumerate", [address, C.POINTER(Evidence), C.c_int, C.POINTER(C.c_int)]),
            bind(lib, "zimage_lt_matmul", [address, C.c_int, address, address, address, address, C.c_size_t, address]),
            bind(lib, "zimage_lt_close", [address]),
            bind(lib, "zimage_lt_search_receipt", [address, C.POINTER(SearchEvidence)]))


def trace_api():
    local = C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_NOLOAD | os.RTLD_GLOBAL)
    default = C.CDLL(None)
    calls = [bind(default, symbol, []) for symbol in ("hipProfilerStart", "hipProfilerStop")]
    spec.need(all(C.cast(call, C.c_void_p).value == C.cast(getattr(local, call.__name__), C.c_void_p).value
                  for call in calls), "TRACE_API_PROVIDER_DRIFT")
    return calls


def witness(scope, output, step):
    import torch

    torch.set_num_threads(4)
    spec.need(torch.__version__ == scope["torch_version"] and torch.version.hip == scope["torch_hip"]
              and torch.cuda.device_count() == 1, "TORCH_RUNTIME_OR_LEASE")
    torch.cuda.set_device(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    spec.need(arch == "gfx936", "GFX936_TARGET_REQUIRED")
    device = pci(scope)
    create, enumerate_rows, matmul, close, search_receipt = native_probe()
    trace_start, trace_stop = trace_api()
    context = C.c_void_p()
    evidence = dict(request=REQUEST, step=step, PID=os.getpid(), physical_PCI=device, device_index=0,
                    queried_maximum=MAX_SOLUTIONS, workspace_cap_bytes=WORKSPACE,
                    query_rc=None, close_rc=None, attempted_calls=0, solutions=[], trace_windows=[])

    def checkpoint():
        write(output / "candidate-witness.json", evidence)

    def synchronize():
        torch.cuda.synchronize(0)

    evidence["create_rc"] = create(C.byref(context))
    checkpoint()
    if evidence["create_rc"] != 0:
        return 1
    try:
        rows = (Evidence * MAX_SOLUTIONS)()
        count = C.c_int()
        evidence["query_rc"] = enumerate_rows(context, rows, MAX_SOLUTIONS, C.byref(count))
        checkpoint()
        spec.need(evidence["query_rc"] == 0 and 0 <= count.value <= MAX_SOLUTIONS, "HEURISTIC_QUERY_REFUSED")
        search = SearchEvidence()
        spec.need(search_receipt(context, C.byref(search)) == 0 and C.sizeof(SearchEvidence) == 16,
                  "CATALOG_RECEIPT")
        spec.need(0 <= search.catalog_count <= 4096 and 0 <= search.support_checks <= search.examined_count
                  <= min(64, search.catalog_count) and count.value in (0, 1)
                  and (search.selected_rank == search.examined_count - 1 and search.support_checks > 0
                       if count.value else search.selected_rank == -1
                       and search.examined_count == min(64, search.catalog_count)), "CATALOG_DENOMINATOR")
        evidence["catalog"] = {name: getattr(search, name) for name, _ in SearchEvidence._fields_}
        for i in range(count.value):
            line = {"ordinal": i, **{name: getattr(rows[i], name) for name, _ in Evidence._fields_}}
            evidence["solutions"].append(line)
        checkpoint()
        runnable = [row for row in evidence["solutions"] if row["runnable"]]
        spec.need(len({row["solution"] for row in runnable}) == len(runnable)
                  and all(0 <= row["supported_workspace"] <= WORKSPACE for row in runnable), "SOLUTION_ACCOUNTING")
        spec.need(len(runnable) == count.value and all(row["solution"] not in (4021, 4100)
                  for row in runnable), "RETIRED_OR_UNSUPPORTED_SOLUTION")
        if not runnable:
            return 0
        a, b, reference = open_layout(torch, fixed_inputs(torch)[:1], (1, spec.K))[0]
        out = guarded(torch, (spec.M, spec.N), (spec.N, 1))
        work = torch.empty(WORKSPACE, device="cuda", dtype=torch.uint8)
        stream = torch.cuda.current_stream(device=0)

        def launch(ordinal):
            evidence["attempted_calls"] += 1
            return matmul(context, ordinal, a.view.data_ptr(), b.view.data_ptr(), out.view.data_ptr(),
                          work.data_ptr(), WORKSPACE, stream.cuda_stream)

        for candidate in runnable:
            ordinal = candidate["ordinal"]
            spec.need(time.time() < scope["deadline_epoch"] - 30, "NATIVE_OWNED_WORK_BUDGET")
            out.view.fill_(spec.GUARD_VALUE)
            synchronize()
            start = time.time_ns()
            candidate["trace_start_rc"] = trace_start()
            spec.need(candidate["trace_start_rc"] == 0, "HIPPROF_START")
            try:
                candidate["traced_matmul_rc"] = launch(ordinal)
                synchronize()
            finally:
                candidate["trace_stop_rc"] = trace_stop()
                end = time.time_ns()
            candidate["traced_exact_BF16"] = bool(candidate["traced_matmul_rc"] == 0
                                                   and same_bits(torch, out.view, reference))
            check_guard(out)
            audit_inputs(torch, [(a, b, reference)])
            spec.need(candidate["trace_stop_rc"] == 0, "HIPPROF_STOP")
            evidence["trace_windows"].append(dict(ordinal=ordinal, solution=candidate["solution"],
                                                  realtime_ns=[start, end]))
            checkpoint()
    finally:
        try:
            synchronize()
        finally:
            evidence["close_rc"] = close(context)
            checkpoint()
    return int(evidence["close_rc"] != 0)


def trace_size(directory):
    members = tuple(directory.iterdir())
    spec.need(len(members) <= 20 and all(path.is_file() and not path.is_symlink() for path in members),
              "UNEXPECTED_TRACE_SCOPE")
    return sum(p.stat().st_size for p in members)


def in_trace(command, directory, deadline):
    fingerprint(TRACER, TRACER_SHA)
    base = [str(TRACER), "--hip-trace", "--trace-off", "--show-pid", "--output-type", "0",
            "-d", str(directory), "-o", str(directory / "trace")]
    with (directory / "tool.stdout.private").open("xb") as out, (directory / "tool.stderr.private").open("xb") as err:
        child = subprocess.Popen([*base, *command], start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err)
        try:
            while child.poll() is None:
                spec.need(trace_size(directory) <= TRACE_LIMIT, "TRACE_DIRECTORY_BUDGET")
                spec.need(time.time() < deadline, "NATIVE_TRACE_TIMEOUT")
                time.sleep(0.15)
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
    spec.need(trace_size(directory) <= TRACE_LIMIT, "TRACE_DIRECTORY_BUDGET")
    return child.returncode


def audit(directory, witness):
    raw_path = directory / "trace.json"
    if not witness["trace_windows"]:
        return [], None
    with raw_path.open("rb") as stream:
        raw = stream.read(TRACE_LIMIT + 1)
    spec.need(0 < len(raw) <= TRACE_LIMIT, "TRACE_OUTPUT_BOUND")
    events = json.loads(raw)["traceEvents"]
    spec.need(len(events) <= 10_000, "TRACE_EVENT_BOUND")
    attached = {row["ordinal"]: [] for row in witness["trace_windows"]}
    for event in events:
        if event.get("cat") != "HIPOPS" or event.get("ph") != "X":
            continue
        data = event["args"]
        spec.need(int(data["pid"]) == witness["PID"] and int(data["devId"]) == witness["device_index"],
                  "TRACE_DEVICE_OWNER")
        begin, end = int(data["BeginNs"]), int(data["EndNs"])
        matches = [row["ordinal"] for row in witness["trace_windows"]
                   if row["realtime_ns"][0] <= begin < end <= row["realtime_ns"][1]]
        spec.need(len(matches) == 1, "TRACE_WINDOW_OWNERSHIP")
        symbol = data["name"]
        spec.need(isinstance(symbol, str) and 0 < len(symbol) <= 1024 and symbol.isascii(), "KERNEL_SYMBOL")
        attached[matches[0]].append(symbol)
    results = []
    for row in witness["solutions"]:
        symbols = attached.get(row["ordinal"], [])
        if row.get("traced_exact_BF16") and len(symbols) == 1:
            symbol = symbols[0]
            if re.fullmatch(r"(?:Custom_)?Cijk_A[a-z]+_Bljk_[A-Za-z0-9_]+", symbol):
                results.append(dict(solution=row["solution"], ordinal=row["ordinal"], symbol=symbol,
                                    exact_1877_transpose_control=symbol == ORIGINAL_LT,
                                    workspace=row["supported_workspace"]))
    return results, sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    scope, step = admission(args.scope, args.scope_sha256)
    if args.child:
        return witness(scope, args.out, step)
    args.out.mkdir(mode=0o700)
    report = dict(request=REQUEST, step=step, node=scope["node"], PCI=scope["PCI"],
                  scope_SHA256=args.scope_sha256, native_binary_SHA256=PROBE_SHA, producer={'step': '1103606.1982', 'SHA256': 'f735cc99e10d0c958b3736fde852bbf9e1f9747827225f1375f0214726279a07', 'bytes': 17760, 'archive_SHA256': 'ab7b5921bf7fec493d0142e106809e298f22b6f330ee1ab70a6ad74036f5c2af'}, planned_maximum_solutions=1,
                  zero_profiled_throughput_pairs=True, training_performance=False, timed_calls=0, passed=False, completed=False)
    try:
        command = [sys.executable, "-B", __file__, *sys.argv[1:], "--child"]
        report["trace_tool_rc"] = in_trace(command, args.out, scope["deadline_epoch"] - 12)
        original = args.out / "candidate-witness.json"
        if original.is_file():
            with original.open("rb") as stream:
                raw = stream.read(32769)
            spec.need(0 < len(raw) <= 32768, "WITNESS_RECEIPT_BOUND")
            info = json.loads(raw)
            spec.need(info["step"] == step and info["physical_PCI"] == scope["PCI"], "WITNESS_SCOPE")
            report.update(witness=info, witness_SHA256=sha256(raw).hexdigest())
            routes, pin = audit(args.out, info)
            report.update(routes=routes, raw_trace_SHA256=pin)
            previous_symbols = ('Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_MT256x256x16_MI16x16x1_SN_GSU1_K1_LRVW2_MIWT8_4_WGM5', 'Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_DTL0_EPS1_GRVWA8_GRVWB8_GSU1_GSUAMB_IU1_K1_LRVW4_MIWT1_1_PGR1_PLR1_SIA1_SU0_SUS0_TLDS0_ULDSWM0_USFGRO0_USSB0_WSS0_WG16_4_1_WGM15')
            spec.need(info["create_rc"] == info["query_rc"] == info["close_rc"] == 0
                      and info["workspace_cap_bytes"] == WORKSPACE and report["trace_tool_rc"] == 0,
                      "DISCOVERY_TERMINAL_IDENTITY")
            spec.need(info["attempted_calls"] == len(info["solutions"]) <= 1, "DISCOVERY_CALL_DENOMINATOR")
            if not info["solutions"]:
                report["decision"] = "NO_SUPPORTED_NEW_IN_PREFIX"
            elif len(routes) != 1:
                report["decision"] = "NEW_SOLUTION_NUMERIC_OR_ROUTE_UNQUALIFIED"
            elif routes[0]["symbol"] in previous_symbols:
                report["decision"] = "NEW_ID_REUSES_PREVIOUS_KERNEL"
            else:
                report["decision"] = "NEW_BF16_KERNEL_QUALIFIED"
                report["passed"] = True
    except (OSError, ValueError, KeyError) as error:
        report.update(error_type=type(error).__name__, error_message=str(error)[:100] if type(error) is ValueError else None)
    finally:
        report["completed"] = True
        write(args.out / "terminal.json", report)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
