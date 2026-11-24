"""Identify the target CPU NumPy provider, then reuse the original .1318 read-only audit."""

from contextlib import redirect_stdout
from hashlib import sha256
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import sysconfig
import time


SOURCE = Path("/checkout")
OUTPUT = Path("/evidence")
RUNTIME = Path("/training-runtime/site")
ENTRY = "code/bw1000/training-vggt-original1318-dev24-audit-v1/audit.py"
ENTRY_SHA = "32f2eb1c523487654a3ce69e1dc0e8a01025119d4ab91cf8179105c883ac2173"
INPUT_SHA = "3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6"
LIMIT = 100
DENOM = {"frozen_dev24_windows": 24, "local_reference_available_inputs": 22, "original_missing_windows": 2,
         "predictions": 0, "model_loads": 0, "weight_payload_bytes": 0, "optimizer_updates": 0}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def encode(result):
    return (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def exact_file(path, maximum):
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        need(path.resolve(strict=True) == path and path.is_file() and 0 < first.st_size <= maximum,
             "EXACT_READ_ONLY_PROVIDER_FILE")
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        last = os.fstat(stream.fileno())
    need((first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) ==
         (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns), "PROVIDER_BYTES_MOVED")
    return {"bytes": first.st_size, "SHA256": digest.hexdigest()}


def save(path, data):
    raw = encode(data)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return pin(raw)


def binding(request, parent, node, epoch, now, env, host, cpus):
    job, task = env.get("SLURM_JOB_ID", ""), env.get("SLURM_STEP_ID", "")
    need(re.fullmatch("[0-9a-f]{32}", request) and job == parent and re.fullmatch("[0-9]+", task)
         and node == host and cpus == 2 and env.get("SLURM_CPUS_PER_TASK") == "2", "ACTUAL_NUMERIC_CPU_STEP")
    step_tres = env.get("SLURM_TRES_PER_TASK", "").lower()
    need(not env.get("SLURM_STEP_GPUS") and not any(word in step_tres for word in ("gpu", "dcu"))
         and all(env.get(key, "") in ("0", "") for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"))
         and all(env.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
         "EXACT_ZERO_GRES_WORKER_ENVIRONMENT")
    need(epoch <= now < epoch + LIMIT, "ORIGINAL_CPU_EPOCH")
    return {"request": request, "parent": parent, "node": node, "step": parent + "." + task,
            "epoch": epoch, "deadline": epoch + LIMIT, "cpus": cpus, "GPU": 0}


def numpy_provider(roots):
    need("numpy" not in sys.modules and "torch" not in sys.modules, "PROVIDER_MUST_BE_FRESH")
    spec = importlib.util.find_spec("numpy")
    need(spec is not None and spec.origin is not None, "CONTAINER_NUMPY_UNAVAILABLE")
    module = importlib.import_module("numpy")
    site = Path(module.__file__).resolve(strict=True)
    need(site == Path(spec.origin).resolve(strict=True) and any(site.is_relative_to(root) for root in roots),
         "CONTAINER_NUMPY_ORIGIN")
    cores = {Path(current.__file__).resolve(strict=True) for name, current in sys.modules.copy().items()
             if name in ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath")
             and isinstance(getattr(current, "__loader__", None), importlib.machinery.ExtensionFileLoader)}
    need(len(cores) == 1, "ONE_ACTUALLY_LOADED_NUMPY_CORE_EXTENSION")
    core, = cores
    abi = sysconfig.get_config_var("SOABI")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    need(core.is_relative_to(site.parent) and core.suffix in (".so", ".pyd")
         and isinstance(extension_suffix, str) and (sys.platform != "linux" or isinstance(abi, str)),
         "CONTAINER_NUMPY_CORE_SCOPE")
    need(module.__version__ == importlib.metadata.version("numpy")
         and float(module.linalg.norm(module.asarray([3., 4.]))) == 5.0 and "torch" not in sys.modules,
         "NUMPY_IMPORT_VERSION_AND_CPU_OPERATION")
    return {"version": module.__version__, "module_path": site.as_posix(),
            "module": exact_file(site, 1024 * 1024), "core_extension_path": core.as_posix(),
            "core_extension": exact_file(core, 64 * 1024 * 1024), "python_executable": sys.executable,
            "python_SOABI": abi, "python_extension_suffix": extension_suffix, "loaded_torch": False}


def frozen_audit(source):
    path = source / ENTRY
    need(exact_file(path, 65536)["SHA256"] == ENTRY_SHA, "FROZEN_1318_AUDIT_SOURCE")
    prior = sys.argv[:]
    captured = io.StringIO()
    try:
        sys.argv = [str(path), "target-read-only"]
        with redirect_stdout(captured):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = prior
    raw = captured.getvalue().encode()
    need(0 < len(raw) <= 16384 and "torch" not in sys.modules, "EXACT_READONLY_AUDIT_RESULT")
    result = json.loads(raw)
    need(result["predictions"] == 0, "ZERO_NATIVE_PREDICTIONS")
    return result, pin(raw)


def reason(error):
    message = str(error)
    if re.fullmatch(r"[A-Z][A-Z0-9_]{3,90}", message):
        return message
    return type(error).__name__


def execute(output, bound, provider_call, audit_call):
    result = dict(DENOM, binding=bound, actual_target_inputs_qualified=0,
                  actual_provider_succeeded=False, frozen_audit_attempted=False,
                  state="CPU_PROVIDER_UNQUALIFIED", eligibility="DEVELOPMENT_PROVENANCE_ONLY")
    phase = "provider"
    try:
        provider = provider_call()
        result["provider_receipt"] = save(output / "PROVIDER.json",
                                          {"schema": "vggt.1318.cpu-provider.v1", "binding": bound,
                                           "provider": provider, **DENOM})
        result["actual_provider_succeeded"] = True
        result["frozen_audit_attempted"] = True
        phase = "audit"
        audit, raw = audit_call()
        result.update(audit_result=audit, audit_stdout=raw)
        if audit["state"] == "READ_ONLY_HEAD_RECORDING_AND_INPUT_PROVENANCE_CHECKED":
            need(audit["qualified_common_windows"] == 22 and audit["original_windows"] == 24
                 and audit["actual_weight_bytes_read"] == 0, "ORIGINAL_TARGET_22_OF_24_BOUNDARY")
            result.update(state="PROVIDER_AND_SAME_22_OF_24_PROVENANCE_CHECKED", actual_target_inputs_qualified=22)
        else:
            need(audit["state"] in ("STOP_EXACT_HEAD_AND_DEV24_ALREADY_SCORED", "STOP_NO_SEQUENCE_DISJOINT_INPUT"),
                 "FROZEN_READ_ONLY_AUDIT_STATE")
            result.update(state=audit["state"])
    except (OSError, ValueError, TypeError, ImportError, KeyError) as error:
        result.update(state="STOP_" + phase.upper() + "_FAILED", error=reason(error))
    proof = save(output / "RESULT.json", result)
    return result, proof


def main():
    request, parent, node, started = sys.argv[1:]
    need(sys.platform == "linux" and sys.flags.dont_write_bytecode and sys.flags.no_user_site
         and OUTPUT.is_dir(), "REAL_CONTAINER_WORKER_ONLY")
    bound = binding(request, parent, node, float(started), time.time(), os.environ,
                    socket.gethostname().split(".")[0], len(os.sched_getaffinity(0)))
    roots = tuple(path.resolve(strict=True) for path in (RUNTIME, Path("/usr"), Path("/opt/dtk")) if path.is_dir())
    result, proof = execute(OUTPUT, bound, lambda: numpy_provider(roots), lambda: frozen_audit(SOURCE))
    print(json.dumps({"result": proof, "state": result["state"], "request": request, "step": bound["step"]}))
    return int(result["state"] != "PROVIDER_AND_SAME_22_OF_24_PROVENANCE_CHECKED")


if __name__ == "__main__":
    raise SystemExit(main())
