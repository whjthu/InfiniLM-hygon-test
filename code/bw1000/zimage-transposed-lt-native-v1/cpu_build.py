"""Compile the frozen hipBLASLt probe on the target without loading it."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import time


ROOT = Path("/smoke")
PARENT = "1103606"
NODE = "f12r2n13"
HEADERS = {
    "/opt/dtk/include/hipblaslt/hipblaslt.h": "3d04be988e619a3230ec08aefc8b1d8d92c03a573d7ebf723d4fe332bdf5ccdb",
    "/opt/dtk/include/hipblaslt/hipblaslt-ext.hpp": "e381c98dfd58cd1a8539b5985eb3db8df2080818cd4cba03538b1b871da9164d",
}
HIPBLAS = Path("/opt/dtk/lib/libhipblaslt.so.0.10")
HIPBLAS_SHA = "03c8f72e2d1a8172127e98cfd41040631e69a7f92821879387cf31df25b0549d"
EXPORTS = {"zimage_lt_create", "zimage_lt_enumerate", "zimage_lt_matmul", "zimage_lt_close"}
REQUIRED_IMPORTS = {"hipblasLtMatmulAlgoGetHeuristic", "hipblasLtMatmul", "hipblasLtMatmulDescSetAttribute",
                    "hipblasLtMatrixLayoutSetAttribute", "hipblasLtMatmulPreferenceSetAttribute"}


def digest(path):
    sha = sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output(name, record):
    p = ROOT / name
    with p.open("xb") as f:
        f.write((json.dumps(record, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def call(argv, label, deadline):
    seconds = min(45.0, deadline - time.time() - 5.0)
    require(seconds >= 1, "Compile chain has no residual time")
    result = subprocess.run(argv, capture_output=True, timeout=seconds)
    for kind, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
        with (ROOT / (label + "." + kind + ".private")).open("xb") as f:
            f.write(raw)
    require(result.returncode == 0, label + " failed")
    return result.stdout.decode("utf8", errors="replace")


def deps(path):
    raw = path.read_bytes()
    require(0 < len(raw) <= 256 * 1024 and raw.startswith(b"probe: "), "Unexpected dependency file")
    files = set(shlex.split(raw.decode("utf8").replace("\\\n", " ").split(":", 1)[1]))
    return files


def build(record, deadline):
    for path, expected in HEADERS.items():
        require(digest(Path(path)) == expected, "hipBLASLt header source changed")
    require(HIPBLAS.resolve(strict=True) == HIPBLAS and digest(HIPBLAS) == HIPBLAS_SHA, "hipBLASLt binary changed")
    compiler = Path(shutil.which("g++")).resolve(strict=True)
    require(compiler.name.startswith(("g++", "x86_64-linux-gnu-g++")), "Unexpected C++ compiler")
    version = call([str(compiler), "--version"], "version", deadline).splitlines()[0]
    record["compiler"] = dict(SHA256=digest(compiler), name=compiler.name, version=version)
    common = [str(compiler), "-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden", "-D__HIP_PLATFORM_AMD__=1",
              "-isystem", "/opt/dtk/include", "-isystem", "/opt/dtk/hip/include",
              "-Wall", "-Wextra", "-Werror"]
    source = ROOT / "probe.cpp"
    target = ROOT / "libzimage_lt_probe.so"
    planned = ROOT / "precompile.M"
    actual = ROOT / "actual.MD"
    require(not target.exists(), "Probe object already exists")
    call([*common, "-M", "-MT", "probe", "-MF", str(planned), str(source)], "preprocess", deadline)
    expected_files = deps(planned)
    require(set(HEADERS) <= expected_files and str(source) in expected_files, "Expected dependency is not in fresh -M")
    call([*common, "-MD", "-MT", "probe", "-MF", str(actual), "-shared", str(source), str(HIPBLAS),
          "-Wl,--no-undefined", "-o", str(target)], "compile", deadline)
    observed_files = deps(actual)
    require(observed_files == expected_files, "Actual -MD dependency set differs from -M")
    require(all(digest(Path(path)) == pin for path, pin in HEADERS.items()), "Header changed during compile")
    require(digest(HIPBLAS) == HIPBLAS_SHA, "Provider changed during compile")
    record["dependency_proof"] = dict(entries=len(expected_files), actual_MD_matches_pre_M=True,
                                      pre_M_SHA256=digest(planned), actual_MD_SHA256=digest(actual), headers=HEADERS)
    sn = call(["nm", "-D", "-C", str(target)], "symbols", deadline)
    exports = {line.split()[-1] for line in sn.splitlines() if re.search(r"\bT\s+zimage_lt_", line)}
    require(exports == EXPORTS, "C ABI exports differ")
    require(all(re.search(r"\bU\s+" + name + r"(?:@|\s|$)", sn, re.M) for name in REQUIRED_IMPORTS),
            "Native imported C symbols differ")
    require(all("hipblaslt_ext::" + name + "(" in sn for name in ("matmulIsAlgoSupported", "getIndexFromAlgo")),
            "Native imported C++ support symbols differ")
    report = call(["ldd", "-r", str(target)], "ldd", deadline)
    require("not found" not in report and "undefined symbol" not in report, "Probe dependency unresolved")
    providers = [Path(line.split("=>")[1].strip().split()[0]).resolve(strict=True) for line in report.splitlines()
                 if re.match(r"\s*libhipblaslt\.so(?:\.\d+)?\s+=>", line)]
    require(providers == [HIPBLAS], "Resolved hipBLASLt provider differs")
    record["artifact"] = dict(bytes=target.stat().st_size, SHA256=digest(target), exports=sorted(exports),
                              exact_hipblaslt=HIPBLAS_SHA, blocker_free_ldd=True,
                              linker_imports=len(REQUIRED_IMPORTS) + 2)


def main():
    request = json.loads((ROOT / "intent.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(node == NODE and re.fullmatch(PARENT + r"\.[0-9]+", step) is not None
            and os.environ.get("SLURM_CPUS_PER_TASK") == "2"
            and all(os.environ.get(n) == "" for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")),
            "CPU step identity or GRES mask changed")
    require(request["parent"] == PARENT and request["node"] == NODE and request["planned_DCU"] == 0,
            "Frozen CPU claim does not cover the step")
    deadline = request["deadline"]
    require(deadline == request["epoch"] + 130 and request["epoch"] <= time.time() < deadline - 10,
            "Original epoch budget is unavailable")
    record = dict(passed=False, node=node, step=step, request=request["request"], attempted_GPU_API_calls=0,
                  compiled_GPU_TUs=0, library_dlopen_attempted=False, formal_performance=False)
    try:
        require(digest(ROOT / "probe.cpp") == request["cpp_SHA256"], "Frozen C++ source bytes changed")
        build(record, deadline)
        record["passed"] = True
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        record.update(error_type=type(error).__name__, error_message=str(error)[:140] if type(error) is ValueError else None)
    output("build-result.json", record)
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return int(not record["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
