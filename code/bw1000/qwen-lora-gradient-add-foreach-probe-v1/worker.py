"""Measure the FP32 addition stage after BF16 gradient conversion on eight devices."""

import ctypes
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ORDER = ("serial", "foreach64", "foreach64", "serial", "serial", "foreach64")
MEASURE = 8
WARM = 2
FLOOR = 4 * 1024**3


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def accumulate(torch, target, sources, mode):
    need(mode in ("serial", "foreach64"), "Unknown accumulation route")
    if mode == "serial":
        for source in sources:
            for out, delta in zip(target, source, strict=True):
                out.add_(delta)
    else:
        for source in sources:
            for start in range(0, len(target), 64):
                torch._foreach_add_(target[start:start+64], source[start:start+64])


def pci_address(index):
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    need(hip.hipDeviceGetPCIBusId(buf, len(buf), index) == 0, "HIP PCI lookup failed")
    result = buf.value.decode().lower()
    return "0000:" + result if len(result) == 7 else result


def device_probe(index, root):
    import torch

    torch.set_num_threads(1)
    need(torch.version.hip and torch.cuda.device_count() == 8, "Expected eight HIP DCUs")
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    expected = json.loads((root / "before.json").read_bytes())["PCIs"]
    pci = pci_address(index)
    need(pci in expected, "HIP PCI differs from host inventory")
    shapes = json.loads((HERE / "layout.json").read_bytes())["shapes"]
    need(len(shapes) == 504 and sum(m*n for m, n in shapes) == 43646976, "Qwen LoRA layout differs")
    begin = time.monotonic()
    generator = torch.Generator(device=device).manual_seed(241780 + index)
    reference = [torch.zeros(shape, device=device, dtype=torch.float32) for shape in shapes]
    candidate = [torch.zeros_like(tensor) for tensor in reference]
    # Casting is completed before measurement. Addition inputs have exact binary fractions.
    sources = []
    for _ in range(4):
        batch = []
        for shape in shapes:
            quantized = torch.randint(-127, 128, shape, device=device, dtype=torch.int32, generator=generator)
            batch.append((quantized.to(torch.bfloat16) * (1 / 128)).float())
        sources.append(batch)
    del quantized
    torch.cuda.synchronize()
    preparation_seconds = time.monotonic() - begin
    need(torch.cuda.mem_get_info(index)[0] >= FLOOR, "Initial memory floor")
    accumulate(torch, reference, sources, "serial")
    accumulate(torch, candidate, sources, "foreach64")
    need(all(torch.equal(a, b) and bool(a.abs().max() > 0)
             for a, b in zip(reference, candidate, strict=True)), "Gradient accumulation lacks exact full-tensor parity")
    rows = []
    for mode in ORDER:
        events = []
        for iteration in range(WARM + MEASURE):
            torch._foreach_zero_(candidate)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host = time.monotonic_ns()
            start.record()
            accumulate(torch, candidate, sources, mode)
            end.record()
            end.synchronize()
            if iteration >= WARM:
                events.append({"event_ms": start.elapsed_time(end), "synchronized_host_ms":
                               (time.monotonic_ns() - host) * 1e-6})
        need(all(torch.equal(a, b) for a, b in zip(reference, candidate, strict=True)),
             "Timed accumulation differs from the validated result")
        rows.append({"mode": mode, "updates": MEASURE, "samples": events,
                     "event_median_ms": statistics.median(row["event_ms"] for row in events),
                     "host_median_ms": statistics.median(row["synchronized_host_ms"] for row in events)})
    need(torch.cuda.mem_get_info(index)[0] >= FLOOR, "Final memory floor")
    result = {"device_index": index, "PCI": pci, "exact_tensors": len(reference), "preparation_seconds": preparation_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(index), "rows": rows,
              "step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]}
    with (root / f"card-{index}.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, allow_nan=False)


def main(root):
    need(os.environ["SLURM_JOB_ID"] == "1103606" and os.environ["SLURM_STEP_ID"].isdigit()
         and socket.gethostname().split(".")[0] == "f12r2n13", "Unexpected Slurm scope")
    workers = [subprocess.Popen([sys.executable, "-B", "-u", __file__, str(root), str(index)]) for index in range(8)]
    try:
        status = [worker.wait() for worker in workers]
        need(status == [0] * 8, "One or more physical DCU workers failed")
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        device_probe(int(sys.argv[2]), Path(sys.argv[1]))
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]))
    else:
        raise SystemExit("Invalid worker arguments")
