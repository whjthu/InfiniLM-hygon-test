"""Run the fixed AB/BA/AB comparison on the one Slurm-assigned card."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time

from compare_pair import compare


HERE = Path(__file__).resolve().parent
ROOT = Path("/refactor")
ROLES = ("norm", "projection")
ORDER = (0, 1, 1, 0, 0, 1)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bf16_fixture():
    # Signed binary fractions, exactly representable in BF16. Vary across
    # all three axes, including across the 64-token tile boundaries.
    rows = []
    for start in range(31):
        row = bytearray()
        for column in range(7168):
            bits, = struct.unpack("<I", struct.pack("<f", ((start + 7*column) % 31 - 15) / 32))
            row += struct.pack("<H", bits >> 16)
        rows.append(bytes(row))
    return b"".join(rows[(3 * token + 5 * slot) % 31] for token in range(512) for slot in range(9))


def checkpoint_inputs(weights):
    path = Path("/shared/Kimi-K3") / weights["safetensors_file"]
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        size, = struct.unpack("<Q", file.read(8))
        if size != weights["header_bytes"] or before.st_size != weights["safetensors_file_bytes"]:
            raise ValueError("checkpoint file identity differs")
        header = file.read(size)
        if digest(header) != weights["header_SHA256"]:
            raise ValueError("checkpoint header differs")
        metadata = json.loads(header)
        values = {}
        for role in ROLES:
            pin = weights["bindings"][role]
            lo, hi = pin["data_offsets"]
            expected = {name: pin[name] for name in ("dtype", "shape", "data_offsets")}
            if metadata[pin["name"]] != expected:
                raise ValueError("checkpoint tensor metadata differs")
            file.seek(8 + size + lo)
            raw = file.read(hi - lo)
            if len(raw) != pin["bytes"] or digest(raw) != pin["SHA256"]:
                raise ValueError("original weight range hash differs")
            values[role] = raw
        after = os.fstat(file.fileno())
        identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("checkpoint changed during the two range reads")
    return values


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
    remaining = lambda: deadline - time.time()
    try:
        result["physical_ID"], result["PCI"] = physical_identity(pre, claim)
        library = Path(claim["component"]["path"])
        if digest(library.read_bytes()) != claim["component"]["SHA256"]:
            raise ValueError("component DSO differs")
        for path, sha in claim["provider_SHA256"].items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError("frozen linked provider differs")
        weights = json.loads((HERE / "CHECKPOINT_WEIGHTS.json").read_bytes())
        raw = bf16_fixture()
        if len(raw) != claim["residual_fixture"]["bytes"] or digest(raw) != claim["residual_fixture"]["SHA256"]:
            raise ValueError("fixed BF16 fixture differs")
        values = checkpoint_inputs(weights)
        result["actual_checkpoint_bytes_read"] = sum(map(len, values.values()))
        values["residual"] = raw
        inputs = out / "inputs"
        inputs.mkdir(mode=0o700)
        for role, body in values.items():
            with (inputs / (role + ".bf16")).open("xb") as file:
                file.write(body)
        binding = {key: weights[key] for key in ("config_SHA256", "index_SHA256", "rms_norm_eps")}
        binding.update(pre_transpose=False, component_SHA256=claim["component"]["SHA256"],
                       BF16_files={role: dict(bytes=len(body), SHA256=digest(body)) for role, body in values.items()})
        write(out / "inputs.json", binding)
        del raw, values
        for ordinal, mode in enumerate(ORDER):
            candidate = out / f"arm-{ordinal}"
            if remaining() < 10:
                raise TimeoutError("original experiment deadline before next arm")
            rec = {"ordinal": ordinal, "mode": mode, "complete": False}
            result["records"].append(rec)
            with (out / f"arm-{ordinal}.log").open("xb") as log:
                child = subprocess.Popen([sys.executable, "-B", str(HERE/"run_arm.py"), str(library),
                                          str(inputs), str(out/"inputs.json"), str(candidate),
                                          "--mode", str(mode)], stdout=log, stderr=subprocess.STDOUT)
                rec["PID"] = child.pid
                try:
                    rec["startticks"] = int(Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()[19])
                except FileNotFoundError:
                    pass
                try:
                    rec["rc"] = child.wait(timeout=min(28, remaining() - 3))
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
                    raise
            if rec["rc"] != 0:
                raise RuntimeError(f"actual residual arm {ordinal} failed")
            report = json.loads((candidate / "result.json").read_bytes())
            if report["complete"] is not True or report["binding"] != binding or report["mode"] != mode:
                raise ValueError("arm evidence does not match fixed input or mode")
            rec.update(complete=True, seconds=report["seconds"], sampled_driver_free_total=report["sampled_driver_free_total"],
                       output_SHA256=report["output_SHA256"])
        for i in (0, 2, 4):
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
