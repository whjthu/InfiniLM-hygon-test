"""Bind a Slurm-owned DP2 rank pair to the original registered training recipe."""
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PREFLIGHT = RESULTS / "training-hunyuanvideo-cached-dp2-target-preflight-v4"
MODEL = RESULTS / "training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5"
WEIGHT = MODEL / "transformer/480p_t2v/diffusion_pytorch_model.safetensors"
BACKEND = "generation/src/gencuda/backends/hunyuanvideo_data_parallel.py"


def need(value, detail):
    if not value:
        raise ValueError(detail)


def dump(path, document):
    with path.open("x", encoding="utf8") as stream:
        json.dump(document, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    need(len(raw) <= 65536, "DP2 local receipt exceeded bounds")
    return json.loads(raw)


def hip_pcis():
    hip = ctypes.CDLL("libamdhip64.so")
    n = ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(n)) == 0 and n.value == 2, "HIP did not expose two Slurm DCUs")
    result = []
    for device in range(n.value):
        value = ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(value, len(value), device) == 0, "assigned HIP PCI lookup failed")
        address = value.value.decode("ascii").lower()
        if len(address) == 7:
            address = "0000:" + address
        need(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", address), "invalid assigned PCI")
        domain, suffix = address.split(":", 1)
        result.append(f"{int(domain, 16):04x}:{suffix}")
    need(len(set(result)) == 2, "DP2 rank devices are not physically distinct")
    return result


def snapshot_entry(run, phase):
    import snapshot

    intent = load(run / "intent.json")
    snapshot.NODES = {intent["node"]}
    snapshot.STEP = re.compile(intent["parent"] + r"\.[0-9]+")
    row = snapshot.snapshot(True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 29500))
    need(row["kfd_holder_count"] == 0, "DP2 node has a KFD or own GPU FD holder")
    dump(run / f"{phase}-devices.json", row)


def run_gpu(run, phase):
    intent = load(run / "intent.json")
    before = load(run / "pre-devices.json")
    parent = os.environ["SLURM_JOB_ID"]
    step = parent + "." + os.environ["SLURM_STEP_ID"]
    node = socket.gethostname().split(".", 1)[0]
    need((parent, node) == (intent["parent"], intent["node"]) and re.fullmatch(parent + r"\.[0-9]+", step),
         "GPU worker escaped its owned Slurm step")
    cards = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    need(re.fullmatch(r"(?:0|[1-9][0-9]*),(?:0|[1-9][0-9]*)", cards) and len(set(cards.split(","))) == 2,
         "expected two exact Slurm CUDA-visible selectors")
    pcis = hip_pcis()
    need(set(pcis).issubset(before["pci"]), "assigned DP2 PCI is outside the preflight inventory")
    current = WEIGHT.stat()
    identity = [current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns]
    need(identity == intent["weight_identity_tail"], "DP2 mounted weight snapshot changed")
    framework = PREFLIGHT / "framework"
    need(sha256((framework / BACKEND).read_bytes()).hexdigest() == intent["backend_SHA256"],
         "executed DP2 provider differs from the compute preflight")
    entrypoint = framework / "tools/hygon_training_recipe.py"
    argv = [sys.executable, "-B", str(entrypoint), "T09.dp2", "--a100-launch", str(PREFLIGHT / "original-launch.json"),
            "--asset-map", str(PREFLIGHT / "asset-map.json"), "--run", "--cards", cards, "--source-root", "/t09-native",
            "--output-root", str(run / "outputs"), "--run-name", phase, "--lock-root", str(run / "locks"), "--lock-host", node]
    if phase != "continuous":
        checkpoint = 10 if phase == "resume10" else 20
        source = run / f"outputs/hunyuanvideo-1.5/train/continuous/checkpoint-{checkpoint}"
        need((source / "COMMITTED.json").is_file(), "continuous DP2 checkpoint is not committed")
        argv.extend(("--resume-from-checkpoint", str(source)))
    dump(run / f"{phase}-assignment.json", dict(phase=phase, step=step, node=node, pci=pcis, epoch=time.time(),
                                                backend_SHA256=intent["backend_SHA256"]))
    limit = min(intent["phase_seconds"][phase], int(intent["deadline_epoch"] - time.time()) - 60)
    need(limit > 0, "Original DP2 deadline elapsed before fresh phase")
    return subprocess.run(argv, timeout=limit).returncode


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    phase = sys.argv[2]
    if phase in ("pre", "post"):
        snapshot_entry(directory, phase)
    elif phase == "gpu":
        for training_phase in ("continuous", "resume10", "resume20"):
            code = run_gpu(directory, training_phase)
            if code:
                raise SystemExit(code)
    else:
        raise ValueError("Unknown DP2 phase")
