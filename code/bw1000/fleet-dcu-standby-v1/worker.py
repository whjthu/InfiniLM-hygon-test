"""Run an interruptible resident BF16 workload on eight assigned DCUs."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time


STOP = threading.Event()
DIM = 8192
DEPTH = 8


def stop(_signal, _frame):
    STOP.set()


def replace_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def pci_devices():
    hip = ctypes.CDLL("libamdhip64.so")
    count = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 8:
        raise RuntimeError("expected eight assigned HIP devices")
    identifiers = []
    for index in range(count.value):
        buffer = ctypes.create_string_buffer(24)
        if hip.hipDeviceGetPCIBusId(buffer, len(buffer), index) != 0:
            raise RuntimeError("assigned HIP PCI lookup failed")
        identifier = buffer.value.decode("ascii").lower()
        identifiers.append(("0000:" + identifier) if len(identifier) == 7 else identifier)
    if len(set(identifiers)) != 8:
        raise RuntimeError("assigned PCI devices are not unique")
    return identifiers


def read_busy(pci):
    path = Path("/sys/bus/pci/devices") / pci / "gpu_busy_percent"
    try:
        value = int(path.read_text(encoding="ascii"))
    except (ValueError, OSError):
        return None
    return value if 0 <= value <= 100 else None


def run_device(torch, index, pci, directory, deadline):
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    with torch.inference_mode():
        left = torch.full((DIM, DIM), 0.5, device=device, dtype=torch.bfloat16)
        right = torch.full((DIM, DIM), 0.25, device=device, dtype=torch.bfloat16)
        output = torch.empty_like(left)
        torch.mm(left, right, out=output)
        if output[0, 0].item() != DIM * 0.125 or output[-1, -1].item() != DIM * 0.125:
            raise RuntimeError("BF16 reference check failed")
        calls = 0
        path = directory / f"card-{index}.json"
        next_receipt = 0
        while not STOP.is_set() and time.monotonic() < deadline:
            for _ in range(DEPTH):
                torch.mm(left, right, out=output)
            torch.cuda.synchronize(device)
            calls += DEPTH
            now = time.monotonic()
            if now >= next_receipt:
                replace_json(path, dict(ready=True, active=True, calls=calls, pci=pci, epoch=time.time(),
                                        busy=read_busy(pci), dimension=DIM, dtype="BF16", synthetic=True))
                next_receipt = now + 5
        replace_json(path, dict(ready=True, active=False, calls=calls, pci=pci, epoch=time.time(),
                                dimension=DIM, dtype="BF16", synthetic=True))


def main():
    import torch

    directory = Path(sys.argv[1])
    expected = json.loads((directory / "expected.json").read_bytes())
    node = socket.gethostname().split(".", 1)[0]
    job = os.environ["SLURM_JOB_ID"]
    step = f"{job}.{os.environ['SLURM_STEP_ID']}"
    if (node, job) != (expected["node"], expected["parent"]) or not step.rsplit(".", 1)[1].isdigit():
        raise RuntimeError("standby step is outside its owned node")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(1)
    if not torch.version.hip or torch.cuda.device_count() != 8:
        raise RuntimeError("standby requires the eight-device HIP runtime")
    devices = pci_devices()
    if set(devices) != set(expected["pci"]):
        raise RuntimeError("assigned PCI set differs from independent host admission")
    replace_json(directory / "worker.json", dict(node=node, step=step, pci=devices, worker_pid=os.getpid(),
                                                epoch=time.time(), synthetic=True))
    deadline = time.monotonic() + expected["seconds"]

    def handle(index, pci):
        try:
            run_device(torch, index, pci, directory, deadline)
        except BaseException:
            STOP.set()
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        workers = [pool.submit(handle, index, pci) for index, pci in enumerate(devices)]
        for work in workers:
            work.result()
    replace_json(directory / "terminal.json", dict(node=node, step=step, state="WORKER_RETURNED", epoch=time.time()))


if __name__ == "__main__":
    main()
