"""Exercise frozen packet materialization, host wrapper compilation and target command order."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compile_once as C
import prepare


SHIM = r"""
#include <cstdint>
#include <cstring>
#define __HIPCC__
#define __device__
#define __global__
#define __shared__
using cudaStream_t = void*;
struct Index { unsigned x = 0; };
inline Index threadIdx, blockIdx;
inline unsigned grid, threads, launches;
inline int status;
inline void* device_stream;
inline void __syncthreads() {}
inline float __shfl_down(float input, unsigned, unsigned) { return input; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __uint_as_float(std::uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
inline std::uint32_t __float_as_uint(float x) { std::uint32_t b; std::memcpy(&b, &x, 4); return b; }
inline int cudaGetLastError() { return status; }
template<class Kernel, class ...Args>
void record(Kernel, unsigned g, unsigned t, unsigned, void* stream, Args...) {
    ++launches;
    grid = g;
    threads = t;
    device_stream = stream;
}
"""
CHECK = r"""
#include <cassert>
#include <limits>
int main() {
    int dst = 0, a = 1, b = 2, c = 3, stream = 4;
    auto call = kimi_output_residual_scores_v1_launch;
    assert(call(&dst, &a, &b, &c, 1.0e-6f, &stream) == 0);
    assert(launches == 1 && grid == 4608 && threads == 256 && device_stream == &stream);
    status = 73;
    assert(call(&dst, &a, &b, &c, 1.0e-6f, &stream) == 73 && launches == 2);
    assert(call(nullptr, &a, &b, &c, 1.0e-6f, &stream) == 1);
    assert(call(&dst, nullptr, &b, &c, 1.0e-6f, &stream) == 1);
    assert(call(&dst, &a, nullptr, &c, 1.0e-6f, &stream) == 1);
    assert(call(&dst, &a, &b, nullptr, 1.0e-6f, &stream) == 1);
    assert(call(&dst, &a, &b, &c, 0.f, &stream) == 1);
    assert(call(&dst, &a, &b, &c, -1.f, &stream) == 1);
    assert(call(&dst, &a, &b, &c, std::numeric_limits<float>::infinity(), &stream) == 1);
    assert(call(&dst, &a, &b, &c, std::numeric_limits<float>::quiet_NaN(), &stream) == 1);
    assert(launches == 2);
}
"""


def wrapper_check(pack, temp):
    launch = ("scores_kernel<<<kimi_output_residual_scores_v1::rows, kimi_output_residual_scores_v1::lanes,\n"
              "                    0, static_cast<cudaStream_t>(stream)>>>(")
    for arm in C.ARMS:
        body = (pack / arm / "fused_scores.cu").read_text(encoding="utf-8")
        assert body.count(launch) == body.count("#include <cuda_runtime.h>") == 1
        # Local Windows COFF has no ELF hidden visibility attribute.
        hidden = '__attribute__((visibility("hidden"))) '
        assert body.count(hidden) == 1
        body = body.replace(hidden, "", 1)
        body = body.replace("#include <cuda_runtime.h>", SHIM).replace(
            launch, "record(scores_kernel, 4608, 256, 0, stream,")
        translated, exe = temp / (arm + ".cpp"), temp / (arm + ".exe")
        translated.write_text(body + CHECK, encoding="utf-8")
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        "-ffp-contract=off", "-I" + str(pack / arm), str(translated), "-o", str(exe)],
                       check=True, timeout=25)
        subprocess.run([str(exe)], check=True, timeout=5)


def commands_check(pack, temp, plan):
    # A recording runner exercises the actual orchestration without nvcc or Slurm.
    header1, header2 = temp / "cuda_runtime.h", temp / "alternate_runtime.h"
    header1.write_bytes(b"local recorded header one")
    header2.write_bytes(b"local recorded header two")
    plan["compiler"]["cuda_runtime_header"] = str(header1)
    known = {str(path): C.digest(path) for path in (header1, header2)}
    for scenario in ("valid", "second-M-drift", "actual-MD-drift"):
        out = temp / scenario
        out.mkdir()
        report = dict(arms={arm: {} for arm in C.ARMS}, actual_device_TUs=0)
        observed = []

        def command(label, argv, seconds):
            observed.append(label)
            arm, phase = label.rsplit("-", 1)
            src = pack / arm / "fused_scores.cu"
            files = {str(src), str(src.with_name("score_math.hpp")), str(header1)}
            if arm == "wave32":
                files.add(str(src.with_name("reduction.hpp")))
            if phase == "M" and scenario == "second-M-drift" and arm == "wave32":
                files.add(str(header2))
            if phase == "compile" and scenario == "actual-MD-drift":
                files.add(str(header2))
            if phase in {"M", "compile"}:
                md = Path(argv[argv.index("-MF") + 1])
                md.write_text(json.dumps(sorted(files)), encoding="utf-8")
                if phase == "compile":
                    Path(argv[-1]).write_bytes(arm.encode())
            else:
                (out / "symbols").write_text("000 T " + C.SYMBOL + "\n", encoding="utf-8")
                return out / "symbols"

        with patch.object(C, "dependencies", side_effect=lambda path, obj: set(json.loads(path.read_bytes()))):
            try:
                C.compile_arms(pack, out, plan, known, command, report, lambda: None)
            except ValueError as error:
                if scenario == "second-M-drift":
                    assert str(error) == "UNEQUAL_BASELINE_CANDIDATE_DTK_CLOSURE"
                    assert observed == ["full-tree-M", "wave32-M"] and report["actual_device_TUs"] == 0
                else:
                    assert scenario == "actual-MD-drift" and str(error) == "ACTUAL_MD_DEPENDENCY_DRIFT"
                    assert observed == ["full-tree-M", "wave32-M", "full-tree-compile"]
            else:
                assert scenario == "valid"
                assert observed == ["full-tree-M", "wave32-M", "full-tree-compile", "full-tree-symbols",
                                    "wave32-compile", "wave32-symbols"] and report["actual_device_TUs"] == 2


def main():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        pack = root / "first"
        first, second = prepare.prepare(pack), prepare.prepare(root / "second")
        assert first == second and first["target_compiles"] == 0
        assert (pack / "SOURCE_FILES.json").read_bytes() == (root / "second/SOURCE_FILES.json").read_bytes()
        plan = C.admitted_sources(pack.resolve(), first["manifest_SHA256"])
        wrapper_check(pack, root)
        commands_check(pack, root, plan)
        dep = root / "test.d"
        dep.write_text("/refactor/leaf.o: /first.h " + chr(92) + "\n /second.h\n", encoding="utf-8")
        assert C.dependencies(dep, "/refactor/leaf.o") == {"/first.h", "/second.h"}
        (pack / "wave32/reduction.hpp").write_bytes(b"drift")
        try:
            C.admitted_sources(pack.resolve(), first["manifest_SHA256"])
        except ValueError as error:
            assert str(error) == "SOURCE_PACK_BYTES"
        else:
            raise AssertionError("modified generated source admitted")
    print(json.dumps(dict(first, deterministic_packet_rebuilds=2, host_ABI_compiles=2, valid_wrapper_calls=4,
                          invalid_no_launch=16, recorded_compiler_order_cases=3, source_tamper_rejections=1,
                          actual_device_or_model_calls=0, actual_slurm_steps=0), indent=2))


if __name__ == "__main__":
    main()
