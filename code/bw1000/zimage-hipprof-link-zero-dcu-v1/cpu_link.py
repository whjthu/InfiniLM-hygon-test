"""Test dynamic HIP API visibility in the exact hipprof child with zero GRES."""

import ctypes as C
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


HIP = Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_SHA256 = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
PROFILER = Path("/opt/dtk/hip/bin/hipprof")
PROFILER_SHA256 = "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"
SYMBOLS = ("hipProfilerStart", "hipProfilerStop")


class DlInfo(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ("file", "base", "symbol", "address")]


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def fingerprint(path, size, pin):
    need(path.resolve(strict=True) == path and path.stat().st_size == size
         and sha256(path.read_bytes()).hexdigest() == pin, "Tool or HIP ELF changed")


def source_pin(fn):
    info = DlInfo()
    libdl = C.CDLL("libdl.so.2")
    lookup = libdl.dladdr
    lookup.argtypes, lookup.restype = [C.c_void_p, C.POINTER(DlInfo)], C.c_int
    ptr = C.cast(fn, C.c_void_p)
    need(lookup(ptr, C.byref(info)) == 1 and info.file and info.symbol, "dladdr could not bind the symbol")
    path = Path(os.fsdecode(C.string_at(info.file))).resolve(strict=True)
    return dict(exact_pinned_HIP_ELF=path == HIP, loader_symbol=os.fsdecode(C.string_at(info.symbol)))


def child(path):
    import torch

    need(torch.__version__ == "2.10.0" and torch.version.hip == "6.3.26113", "Torch runtime drift")
    need(not torch.cuda.is_initialized(), "Unexpected GPU context")
    fingerprint(HIP, 12_349_016, HIP_SHA256)
    lookup = C.CDLL(None)
    local = C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    before = {k: hasattr(lookup, k) for k in SYMBOLS}
    local_definitions = {k: source_pin(getattr(local, k)) for k in SYMBOLS}
    # Promote the already loaded, exact native runtime object in this process only.
    C.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_NOLOAD | os.RTLD_GLOBAL)
    after = {k: source_pin(getattr(lookup, k)) for k in SYMBOLS}
    same_api = {k: C.cast(getattr(lookup, k), C.c_void_p).value == C.cast(getattr(local, k), C.c_void_p).value
                for k in SYMBOLS}
    need(not torch.cuda.is_initialized(), "Unexpected GPU context")
    passed = all(v["exact_pinned_HIP_ELF"] and v["loader_symbol"] == k for k, v in after.items()) and all(same_api.values())
    path.write_text(json.dumps(dict(passed=passed, before_local_only=before, local_definitions=local_definitions,
                                    after_global=after, same_API=same_api, profiler_API_calls=0,
                                    torch_GPU_initialized=False), sort_keys=True), encoding="utf8")
    return int(not passed)


def main():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    need(step.startswith("1103606.") and step.split(".")[1].isdecimal()
         and socket.gethostname().split(".", 1)[0] == "f12r2n13"
         and all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
         "Expected zero-GRES target scope")
    fingerprint(PROFILER, 15_528_344, PROFILER_SHA256)
    with tempfile.TemporaryDirectory(prefix="zimage-hipprof-elf-link-") as d:
        base = Path(d)
        path = base / "child.json"
        cmd = [str(PROFILER), "--hip-trace", "--trace-off", "--show-pid", "--output-type", "0",
               "-d", str(base), "-o", str(base / "trace"), sys.executable, "-B", __file__, "--check", str(path)]
        result = subprocess.run(cmd, capture_output=True, timeout=40)
        payload = json.loads(path.read_bytes()) if path.is_file() and 0 < path.stat().st_size <= 4096 else None
        out = dict(step=step, tool_exit=result.returncode, tracer_stdout_SHA256=sha256(result.stdout).hexdigest(),
                   tracer_stderr_SHA256=sha256(result.stderr).hexdigest(), child=payload,
                   no_actual_kernel_qualified=True, passed=bool(result.returncode == 0 and payload and payload["passed"]))
        print(json.dumps(out, sort_keys=True, allow_nan=False), flush=True)
    return int(not out["passed"])


if __name__ == "__main__":
    raise SystemExit(child(Path(sys.argv[2])) if len(sys.argv) == 3 and sys.argv[1] == "--check" else main())
