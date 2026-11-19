"""One original-path diagnostic, with a durable last stage on timeout."""
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

from inspect_stages import summarize


HERE = Path(__file__).resolve().parent
ROOT = Path("/refactor")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


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


def run():
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    out = ROOT / "results" / claim["label"]
    deadline = float(os.environ["KIMI_RESIDUAL_EPOCH"]) + claim["worker_deadline_s"]
    pre = json.loads((out / "pre.json").read_bytes())
    data = dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                boot_id=pre["boot_id"], records=[], complete=False, missing_arms=1,
                expected_arms=1, mode=0, performance_qualified=False, model_qualified=False)
    try:
        data["physical_ID"], data["PCI"] = physical_identity(pre, claim)
        library = Path(claim["component"]["path"])
        if digest(library.read_bytes()) != claim["component"]["SHA256"]:
            raise ValueError("stage diagnostic library differs")
        for path, sha in claim["provider_SHA256"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError("frozen linked provider differs")
        original = ROOT / "results/Kimi461-output-residual-component-GPU-v1"
        raw = (original / "inputs.json").read_bytes()
        if digest(raw) != claim["original_inputs_SHA256"]:
            raise ValueError("original input receipt differs")
        binding = json.loads(raw)
        if binding["pre_transpose"] is not False or binding["component_SHA256"] != claim["original_component_SHA256"]:
            raise ValueError("original input postload binding differs")
        binding["component_SHA256"] = claim["component"]["SHA256"]
        write(out / "inputs.json", binding)
        remaining = deadline - time.time()
        if remaining < 100:
            raise TimeoutError("original diagnostic epoch cannot accommodate the fixed bound")
        record = dict(ordinal=0, mode=0, complete=False)
        data["records"].append(record)
        log_path = out / "arm-0.log"
        with log_path.open("xb") as log:
            proc = subprocess.Popen([sys.executable, "-B", str(HERE/"run_arm.py"), str(library),
                str(original/"inputs"), str(out/"inputs.json"), str(out/"arm-0"), "--mode","0"],
                stdout=log, stderr=subprocess.STDOUT)
            record["PID"] = proc.pid
            try:
                record["startticks"] = int(Path("/proc",str(proc.pid),"stat").read_text().rsplit(") ",1)[1].split()[19])
            except FileNotFoundError:
                pass
            try:
                record["rc"] = proc.wait(timeout=claim["arm_timeout_s"])
            except subprocess.TimeoutExpired:
                proc.kill()
                record["rc"] = proc.wait(timeout=3)
                record["timed_out"] = True
        with log_path.open("rb") as log:
            raw = log.read(512*1024+1)
        if len(raw) > 512*1024:
            raise ValueError("stage marker log exceeds bound")
        data["markers"] = summarize(raw)
        data["marker_log_SHA256"] = digest(raw)
        if record["rc"] != 0:
            raise RuntimeError("diagnostic arm failed or timed out; retain the stage prefix")
        outcome = json.loads((out/"arm-0/result.json").read_bytes())
        if (outcome["complete"] is not True or outcome["binding"] != binding
            or data["markers"]["innermost_unreturned"]):
            raise ValueError("diagnostic completed without closed marks or exact binding")
        output = (out/"arm-0/output.bf16").read_bytes()
        if len(output) != 512*7168*2 or digest(output) != outcome["output_SHA256"]:
            raise ValueError("diagnostic BF16 output differs")
        data["complete"] = record["complete"] = True
        data["missing_arms"] = 0
        data["output_SHA256"] = outcome["output_SHA256"]
    except Exception as error:
        data["failure"] = dict(kind=type(error).__name__, reason=str(error)[:180])
        raise
    finally:
        write(out / "worker-result.json", data)


if __name__ == "__main__":
    run()
