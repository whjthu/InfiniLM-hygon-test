"""Run the fixed AB/BA/AB comparison on the one Slurm-assigned card."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from compare_pair import compare
from arm_watchdog import wait


HERE = Path(__file__).resolve().parent
ROOT = Path("/refactor")
ORDER = (0, 1, 1, 0, 0, 1)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def physical_identity(pre, claim):
    physical = os.environ["SLURM_STEP_GPUS"]
    if (not re.fullmatch("[0-7]", physical) or os.environ["HIP_VISIBLE_DEVICES"] != physical
        or socket.gethostname().split(".")[0] != claim["node"]):
        raise ValueError("wrong Slurm GPU mask or host")
    libc = C.CDLL("libcudart.so.12")
    count = C.c_int()
    address = C.create_string_buffer(32)
    if libc.cudaGetDeviceCount(C.byref(count)) or count.value != 1:
        raise ValueError("expected one visible CUDA-compatible device")
    if libc.cudaDeviceGetPCIBusId(address, len(address), 0):
        raise ValueError("physical device PCI query failed")
    pci = address.value.decode("ascii").lower()
    if pci not in pre["PCI"]:
        raise ValueError("allocated PCI missing from the preflight inventory")
    return physical, pci


def write(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def run():
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    out = ROOT / "results" / claim["label"]
    deadline = float(os.environ["KIMI_RESIDUAL_EPOCH"]) + claim["worker_deadline_s"]
    pre = json.loads((out / "pre.json").read_bytes())
    result = dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  planned_arms=list(ORDER), records=[], pairs=[], complete=False,
                  actual_checkpoint_bytes_read=0, worker_pid=os.getpid(),
                  boot_id=pre["boot_id"], performance_qualified=False, model_qualified=False)
    whole_deadline = time.monotonic() + deadline - time.time()
    remaining = lambda: whole_deadline - time.monotonic()
    try:
        result["physical_ID"], result["PCI"] = physical_identity(pre, claim)
        library = Path(claim["component"]["path"])
        if digest(library.read_bytes()) != claim["component"]["SHA256"]:
            raise ValueError("component DSO differs")
        for path, sha in claim["provider_SHA256"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError("frozen linked provider differs")
        original = ROOT / "results/Kimi461-output-residual-component-GPU-v1"
        raw_binding = (original / "inputs.json").read_bytes()
        if digest(raw_binding) != claim["original_input_receipt_SHA256"]:
            raise ValueError("original input receipt differs")
        binding = json.loads(raw_binding)
        inputs = original / "inputs"
        for ordinal, mode in enumerate(ORDER):
            candidate = out / f"arm-{ordinal}"
            if remaining() < 10:
                raise TimeoutError("original experiment deadline before next arm")
            rec = {"ordinal": ordinal, "mode": mode, "complete": False}
            result["records"].append(rec)
            with (out / f"arm-{ordinal}.log").open("xb") as log:
                started = time.monotonic()
                child = subprocess.Popen([sys.executable, "-B", str(HERE/"run_arm.py"), str(library),
                                          str(inputs), str(original/"inputs.json"), str(candidate),
                                          "--mode", str(mode)], stdout=log, stderr=subprocess.STDOUT)
                rec["PID"] = child.pid
                try:
                    rec["startticks"] = int(Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()[19])
                except FileNotFoundError:
                    pass
                try:
                    rec["rc"], startup = wait(child, candidate / "startup.json", started, whole_deadline - 3,
                                              claim["startup_bound_s"], claim["native_bound_s"])
                except (TimeoutError, ValueError):
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=2)
                    raise
            if rec["rc"] != 0:
                raise RuntimeError(f"actual residual arm {ordinal} failed")
            report = json.loads((candidate / "result.json").read_bytes())
            if report["complete"] is not True or report["binding"] != binding or report["mode"] != mode:
                raise ValueError("arm evidence does not match fixed input or mode")
            if startup is None or report["startup"] != startup or not (
                startup["ready_monotonic_ns"] <= report["finished_monotonic_ns"]
                <= startup["ready_monotonic_ns"] + claim["native_bound_s"] * 1_000_000_000
                and report["finished_monotonic_ns"] * 1e-9 < whole_deadline
            ):
                raise ValueError("missing or expired startup/native evidence")
            rec.update(complete=True, startup=startup, seconds=report["seconds"],
                       sampled_driver_free_total=report["sampled_driver_free_total"],
                       output_SHA256=report["output_SHA256"])
            if ordinal % 2 == 1:
                i = ordinal - 1
                old, tiled = sorted(result["records"][i:i+2], key=lambda arm: arm["mode"])
                a, b = [(out / f'arm-{row["ordinal"]}' / "output.bf16").read_bytes() for row in (old,tiled)]
                if digest(a) != old["output_SHA256"] or digest(b) != tiled["output_SHA256"]:
                    raise ValueError("saved output digest changed")
                row = compare(a,b)
                row["baseline_ordinal"], row["tiled_ordinal"] = old["ordinal"], tiled["ordinal"]
                result["pairs"].append(row)
        result["complete"] = True
        result["bitwise_equal_all_pairs"] = all(v["changed_words"] == 0 for v in result["pairs"])
    except Exception as error:
        result["failure"] = dict(kind=type(error).__name__, reason=str(error)[:180])
        raise
    finally:
        result["missing_arms"] = len(ORDER) - sum(row["complete"] for row in result["records"])
        write(out / "worker-result.json", result)


if __name__ == "__main__":
    run()
