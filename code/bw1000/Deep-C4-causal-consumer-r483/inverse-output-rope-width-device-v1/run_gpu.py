"""Verify the actually admitted card and loader before the existing paired binary."""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


CUDA = Path("/opt/dtk/cuda/cuda/lib64/libcudart.so.12")
WORK = Path(__file__).resolve().parent
OUT = Path("/audit-out")
BINARY = Path("/original-link/artifacts/compare")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    with path.open("rb") as handle:
        digest = sha256()
        for data in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(data)
        return {"bytes": handle.tell(), "SHA256": digest.hexdigest()}


def canonical_pci(text):
    value = text.lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = "0000:" + value
    need(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value), "Invalid runtime PCI")
    return value


def runtime_card():
    api = ctypes.CDLL(str(CUDA))
    api.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    api.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    need(api.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1, "Expected one visible CUDA device")
    pci = ctypes.create_string_buffer(64)
    need(api.cudaDeviceGetPCIBusId(pci, len(pci), 0) == 0, "CUDA PCI lookup failed")
    print(canonical_pci(pci.value.decode("ascii")), flush=True)


def main():
    if sys.argv[1:] == ["--visible-card"]:
        runtime_card()
        return 0
    claim = json.loads((WORK / "CLAIM.json").read_bytes())
    job, step = (os.environ[name] for name in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    node = socket.gethostname().split(".")[0]
    gres = os.environ.get("SLURM_STEP_GPUS")
    need(job == claim["parent"] and step.isdecimal() and node == claim["node"]
         and gres and re.fullmatch(r"[0-9]+", gres), "Expected one assigned Slurm GPU")
    need(pin(BINARY) == claim["executable"] and pin(CUDA) == claim["CUDA_provider"], "Original link or CUDA provider differs")
    loader = subprocess.check_output(["/usr/bin/ldd", str(BINARY)], text=True, timeout=8)
    matches = re.findall(r"(?m)^\s*libcudart\.so\.12\s+=>\s+(/\S+)\s+\(", loader)
    need(len(matches) == 1 and Path(matches[0]).resolve(strict=True) == CUDA.resolve(strict=True),
         "Resolved CUDA library differs")
    # Use a fresh child so any runtime state from identity lookup ends before timing.
    device = subprocess.check_output([sys.executable, "-I", "-S", "-B", str(Path(__file__)),
                                      "--visible-card"], text=True, timeout=12).strip()
    pci_name = canonical_pci(device)
    admitted = sorted(claim["observed_node_PCIs"])
    need(len(admitted) == len(set(admitted)) == 8 and pci_name in admitted, "Visible card is outside the admitted node")
    with (OUT / "device-identity.json").open("x", encoding="utf-8") as stream:
        json.dump({"request": claim["request"], "step": job + "." + step, "node": node, "pci": pci_name,
                   "slurm_step_gpu": gres, "CUDA_provider": claim["CUDA_provider"]}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    output = OUT / "samples.jsonl"
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    os.execv(str(BINARY), [str(BINARY), str(output), pci_name])


if __name__ == "__main__":
    raise SystemExit(main())
