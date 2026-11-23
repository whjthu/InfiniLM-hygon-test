"""Render a new exact-DTK transposed Lt solution discovery scope."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CPU_OLD = HERE.with_name("zimage-transposed-lt-native-v1")
GPU_OLD = HERE.with_name("zimage-transposed-lt-native-route-v1")
OLD_CPU_REQUEST = "f329b5e79624463fa97854b93df386d3"
CPU_REQUEST = "fd31777395fb474f86dedacef84dcce9"
GPU_REQUEST = "4241e3d58506499cb859abead6f937de"
GPU_LABEL = "O01-zimage-transposed-lt-unseen-solution-v1"
ORIGINAL_PROBE = "2eae6ce6e77b89f94e82e9bec39400671c13a712b6af2fe4be887ff4ecaf180c"
ORIGINAL_BUILD = "2b9405e770a1d28d3905fbd89dd8be24485710d7280e5840d77581891fd91374"
ORIGINAL_LAUNCHER = "49f5cf433e6a6aa319bee3b0ef4d5947eb17f0c0521b60cfa8625f1724ac3a09"
ORIGINAL_WORKER = "ffbbac58e55e445f20c109e060d907f4ca968260b532c775eab64d7684d78c0e"
CONTRACT = "956c0892690a554fb31549051797f9dff82d851f1c4c79e7a3e7a2e07059f148"
PROVIDER = "03c8f72e2d1a8172127e98cfd41040631e69a7f92821879387cf31df25b0549d"
COMPILER = "2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0"
RESULT = REPO / "results/bw1000/2026-11-23/O01-zimage-transposed-lt-native-original1885-terminal.json"
RESULT_SHA = "164ec044ba4cf61e0fcbcc0f010078e370d93f630fc304eccf78569bea9fa1d2"
EXCLUDED = {4021, 4100}
EXPORTS = {"zimage_lt_create", "zimage_lt_enumerate", "zimage_lt_matmul", "zimage_lt_close",
           "zimage_lt_search_receipt"}


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    with Path(path).open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536 or digest(raw) != expected:
        raise ValueError("source or evidence bytes changed: " + path.name)
    return raw


def replace(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("frozen source boundary changed: " + before[:80])
    return raw.replace(before, after)


def original_routes():
    audit = json.loads(pinned(RESULT, RESULT_SHA))
    traced = audit["raw_trace_independent_audit"]["exact_hardware_routes"]
    if ({row["solution"] for row in traced} != EXCLUDED or len(traced) != 2
            or audit["assessment"]["common_workspace_allocated_bytes"] != 32 << 20):
        raise ValueError("frozen exclusions or workspace differ")
    return {row["solution"]: row["symbol"] for row in traced}


def cpu_inputs():
    original_routes()
    native = pinned(CPU_OLD / "probe.cpp", ORIGINAL_PROBE).decode()
    native = replace(native, "#include <type_traits>\n", "#include <type_traits>\n#include <vector>\n")
    native = replace(native, "struct Context {", """struct SearchEvidence {
    std::int32_t catalog_count{};
    std::int32_t examined_count{};
    std::int32_t support_checks{};
    std::int32_t selected_rank{-1};
};
static_assert(std::is_standard_layout_v<SearchEvidence> && sizeof(SearchEvidence) == 16);

struct Context {
    SearchEvidence search{};""")
    begin = native.index("hipblasStatus_t find_solutions(Context& c)")
    end = native.index("}  // namespace", begin)
    native = native[:begin] + (HERE / "find_solution.inc").read_text() + "\n" + native[end:]
    native = replace(native, "PROBE_API int zimage_lt_matmul", """PROBE_API int zimage_lt_search_receipt(void* address, SearchEvidence* out) {
    if (!address || !out) return HIPBLAS_STATUS_INVALID_VALUE;
    const auto& c = *static_cast<Context*>(address);
    if (!c.enumerated) return HIPBLAS_STATUS_INVALID_VALUE;
    *out = c.search;
    return HIPBLAS_STATUS_SUCCESS;
}

PROBE_API int zimage_lt_matmul""").encode()
    build = pinned(CPU_OLD / "cpu_build.py", ORIGINAL_BUILD).decode()
    build = replace(build, '"zimage_lt_matmul", "zimage_lt_close"}', '"zimage_lt_matmul", "zimage_lt_close", "zimage_lt_search_receipt"}')
    build = replace(build, '"hipblasLtMatmulAlgoGetHeuristic", "hipblasLtMatmul",', '"hipblasLtMatmul",')
    build = replace(build, '"matmulIsAlgoSupported", "getIndexFromAlgo"', '"matmulIsAlgoSupported", "getIndexFromAlgo", "getAllAlgos"')
    build = replace(build, "len(REQUIRED_IMPORTS) + 2", "len(REQUIRED_IMPORTS) + 3")
    build = replace(build, '    version = call([str(compiler), "--version"]',
                    f'    require(digest(compiler) == {COMPILER!r}, "Original .1882 compiler differs")\n'
                    '    version = call([str(compiler), "--version"]')
    build = replace(build, '    require(request["parent"] == PARENT',
                    f'    require(request["request"] == {CPU_REQUEST!r} and request["parent"] == PARENT')
    build = replace(build, 'compiled_GPU_TUs=0, library_dlopen_attempted=False',
                    'compiled_GPU_TUs=0, cpp_SHA256=request["cpp_SHA256"], library_dlopen_attempted=False').encode()
    launcher = pinned(CPU_OLD / "launch_cpu.py", ORIGINAL_LAUNCHER).decode()
    launcher = replace(launcher, OLD_CPU_REQUEST, CPU_REQUEST)
    launcher = replace(launcher, ORIGINAL_PROBE, digest(native))
    launcher = replace(launcher, ORIGINAL_BUILD, digest(build)).encode()
    return {"probe.cpp": native, "cpu_build.py": build, "launch_cpu.py": launcher}


def build_binding(receipt, receipt_sha):
    data = json.loads(pinned(receipt, receipt_sha))
    assessment = data["assessment"]
    rows = data["accounting"]
    record = data["files"]["build-result.json"]["record"]
    artifact = record["artifact"]
    step = record["step"]
    expected = digest(cpu_inputs()["probe.cpp"])
    if (data["request"] != record["request"] or record["request"] != CPU_REQUEST
            or len(rows) != 1 or rows[0][0] != step or not re.fullmatch(r"1103606\.[0-9]+", step)
            or rows[0][2:4] != ["COMPLETED", "0:0"] or rows[0][5] != "f12r2n13"
            or "gres/" in rows[0][6] or assessment["actual_gpu_allocations"] != 0
            or record["passed"] is not True or record["attempted_GPU_API_calls"] != 0
            or record["compiled_GPU_TUs"] != 0 or record["library_dlopen_attempted"] is not False
            or record["cpp_SHA256"] != expected or record["compiler"]["SHA256"] != COMPILER
            or record["dependency_proof"]["actual_MD_matches_pre_M"] is not True
            or artifact["exact_hipblaslt"] != PROVIDER or artifact["blocker_free_ldd"] is not True
            or set(artifact["exports"]) != EXPORTS or artifact["linker_imports"] != 7
            or not 0 < artifact["bytes"] < 1024 * 1024 or not re.fullmatch(r"[0-9a-f]{64}", artifact["SHA256"])):
        raise ValueError("new probe actual zero-GRES compile and provider evidence required")
    return {"step": step, "SHA256": artifact["SHA256"], "bytes": artifact["bytes"], "archive_SHA256": receipt_sha}


def gpu_inputs(build):
    raw = pinned(GPU_OLD / "worker.py", ORIGINAL_WORKER).decode()
    raw = replace(raw, "55f9a103f5834e1c9e675a7c099feab1", GPU_REQUEST)
    raw = replace(raw, "O01-zimage-transposed-lt-native-route-v1", GPU_LABEL)
    raw = replace(raw, "a50abd190367ff3b30335db0a9ba750449ca03f00fb95415941b94f703e26800", build["SHA256"])
    raw = replace(raw, '    fingerprint(PROBE, PROBE_SHA)\n',
                  f'    fingerprint(PROBE, PROBE_SHA)\n'
                  f'    spec.need(PROBE.stat().st_size == {build["bytes"]}, "NATIVE_BINARY_SIZE")\n')
    raw = replace(raw, "class Evidence(C.Structure):", """class SearchEvidence(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("catalog_count", "examined_count", "support_checks", "selected_rank")]


class Evidence(C.Structure):""")
    raw = replace(raw, '            bind(lib, "zimage_lt_close", [address]))',
                  '            bind(lib, "zimage_lt_close", [address]),\n'
                  '            bind(lib, "zimage_lt_search_receipt", [address, C.POINTER(SearchEvidence)]))')
    raw = replace(raw, "create, enumerate_rows, matmul, close = native_probe()",
                  "create, enumerate_rows, matmul, close, search_receipt = native_probe()")
    raw = replace(raw, "        for i in range(count.value):", """        search = SearchEvidence()
        spec.need(search_receipt(context, C.byref(search)) == 0 and C.sizeof(SearchEvidence) == 16,
                  "CATALOG_RECEIPT")
        spec.need(0 <= search.catalog_count <= 4096 and 0 <= search.support_checks <= search.examined_count
                  <= min(64, search.catalog_count) and count.value in (0, 1)
                  and (search.selected_rank == search.examined_count - 1 and search.support_checks > 0
                       if count.value else search.selected_rank == -1
                       and search.examined_count == min(64, search.catalog_count)), "CATALOG_DENOMINATOR")
        evidence["catalog"] = {name: getattr(search, name) for name, _ in SearchEvidence._fields_}
        for i in range(count.value):""")
    raw = replace(raw, '        if not runnable:\n            return 2',
                  '        spec.need(len(runnable) == count.value and all(row["solution"] not in (4021, 4100)\n'
                  '                  for row in runnable), "RETIRED_OR_UNSUPPORTED_SOLUTION")\n'
                  '        if not runnable:\n            return 0')
    warm = raw.index('            out.view.fill_(spec.GUARD_VALUE)\n            candidate["warm_rc"]')
    after = raw.index('            out.view.fill_(spec.GUARD_VALUE)\n            synchronize()', warm)
    raw = raw[:warm] + raw[after:]
    before = raw.index('            controls = [r for r in routes if r["exact_1877_transpose_control"]]')
    after = raw.index("    except (OSError, ValueError, KeyError)", before)
    exclusions = tuple(original_routes().values())
    raw = raw[:before] + f'''            previous_symbols = {exclusions!r}
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
''' + raw[after:]
    raw = replace(raw, "native_binary_SHA256=PROBE_SHA, planned_maximum_solutions=MAX_SOLUTIONS,",
                  f"native_binary_SHA256=PROBE_SHA, producer={build!r}, planned_maximum_solutions=1,")
    return {"worker.py": raw.encode(), "contract.py": pinned(GPU_OLD / "contract.py", CONTRACT)}


def stage(output, files, request):
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    pins = {name: {"bytes": len(raw), "SHA256": digest(raw)} for name, raw in files.items()}
    manifest = (json.dumps({"files": pins, "request": request, "original_1885_SHA256": RESULT_SHA},
                           indent=2, sort_keys=True) + "\n").encode()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="zimage-unseen-lt-", dir=output.parent) as tmp:
        new = Path(tmp) / "package"
        new.mkdir(mode=0o700)
        for name, raw in files.items():
            (new / name).write_bytes(raw)
        (new / "SOURCE_FILES.json").write_bytes(manifest)
        new.rename(output)
    return {"request": request, "stage_SHA256": digest(manifest), "files": pins}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    subparsers.add_parser("cpu").add_argument("output", type=Path)
    gpu = subparsers.add_parser("gpu")
    gpu.add_argument("receipt", type=Path)
    gpu.add_argument("receipt_SHA256")
    gpu.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.kind == "cpu":
        result = stage(args.output, cpu_inputs(), CPU_REQUEST)
    else:
        result = stage(args.output, gpu_inputs(build_binding(args.receipt, args.receipt_SHA256)), GPU_REQUEST)
    print(json.dumps(result, sort_keys=True))
