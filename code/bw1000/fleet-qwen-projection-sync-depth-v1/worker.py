"""Measure host synchronization cadence for preallocated BF16 projections."""

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import threading
import time

from node_sample import fresh_inventory

# Fixed effective rows from the three prior resident projection shape groups.
SHAPES = (15, 184, 570)
ORDER = (8, 64, 64, 8)
K, N = 4096, 1024
CASE_SECONDS = 5.0


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def compare(device, rows, seed, deadline, first_sample, halted):
    import torch

    generator = torch.Generator().manual_seed(seed)
    cpu_weight = torch.randint(-2, 3, (K, N), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (rows, K), generator=generator, dtype=torch.int8)
                  for _ in range(2)]
    gold = [(x.float() @ cpu_weight[:, :16].float()).bfloat16() for x in cpu_inputs]
    weight = cpu_weight.to(device, dtype=torch.bfloat16)
    inputs = [x.to(device, dtype=torch.bfloat16) for x in cpu_inputs]
    outputs = [torch.empty((rows, N), device=device, dtype=torch.bfloat16) for _ in inputs]
    references = [torch.mm(x, weight) for x in inputs]

    def submit(count):
        for i in range(count):
            j = i % len(inputs)
            torch.mm(inputs[j], weight, out=outputs[j])
        torch.cuda.synchronize(device)

    def check():
        if not all(torch.equal(result, expected) and torch.equal(result[:, :16].cpu(), cpu)
                   for result, expected, cpu in zip(outputs, references, gold, strict=True)):
            raise ValueError("PREALLOCATED_OUTPUT_OR_CPU_REFERENCE_MISMATCH")

    for count in (8, 64):
        submit(count)
        check()

    values = {depth: [] for depth in (8, 64)}
    for depth in ORDER:
        torch.cuda.reset_peak_memory_stats(device)
        drains = 0
        start = time.monotonic()
        first_sample()
        while time.monotonic() - start < CASE_SECONDS:
            if halted.is_set():
                raise InterruptedError("COOPERATIVE_TRAINING_HANDOFF")
            if time.time() >= deadline:
                raise TimeoutError("BOUNDED_OPERATOR_EPOCH")
            submit(depth)
            drains += 1
        seconds = time.monotonic() - start
        check()
        calls = drains * depth
        values[depth].append(dict(seconds=seconds, drains=drains, projections=calls,
                                  rows=calls * rows, peak_bytes=torch.cuda.max_memory_allocated(device)))
    medians = {depth: statistics.median(s["rows"] / s["seconds"] for s in records)
               for depth, records in values.items()}
    return dict(M=rows, K=K, N=N, numeric=True, distinct_inputs=2, measurements=values,
                depth64_over_depth8=medians[64] / medians[8])


def main(root, last, expected):
    node = socket.gethostname().split(".", 1)[0]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    if sorted(fresh_inventory()) != sorted(expected[node]) or any(Path("/sys/class/kfd/kfd/proc").iterdir()):
        raise ValueError("FRESH_PCI_KFD_IS_NOT_CLEAR")
    import torch

    torch.set_num_threads(1)
    if not torch.version.hip or torch.cuda.device_count() != 8:
        raise ValueError("ACTUAL_EIGHT_DEVICE_INVENTORY_MISMATCH")
    directory = root / node
    directory.mkdir(mode=0o700)
    devices = {p.resolve().parent.name: p for p in Path("/sys/class/drm").glob("renderD*/device/gpu_busy_percent")}
    stop = threading.Event()
    halted = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: halted.set())
    seed = int.from_bytes(hashlib.sha256(node.encode()).digest()[:4], "little")

    def monitor():
        data = []
        while not stop.is_set():
            try:
                row = {pci: int(devices[pci].read_text()) for pci in expected[node]}
                if len(row) == 8 and all(0 <= x <= 100 for x in row.values()):
                    data.append(dict(epoch=time.time(), busy=row))
            except (OSError, KeyError, ValueError):
                pass
            stop.wait(2)
        save(directory / "busy.json", dict(node=node, step=step, rows=data))

    def work(card):
        device = torch.device("cuda", card)
        def active():
            path = directory / f"active-{card}.json"
            if not path.exists():
                save(path, dict(node=node, step=step, card=card, measured_epoch=time.time()))

        results, error = [], None
        try:
            with torch.cuda.device(device), torch.no_grad():
                for index, rows in enumerate(SHAPES):
                    results.append(compare(device, rows, seed + card * 13 + index, last, active, halted))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)[:100]
        save(directory / f"card-{card}.json", dict(node=node, step=step, card=card, cases=results, error=error))
        return error is None

    from concurrent.futures import ThreadPoolExecutor
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            checks = list(pool.map(work, range(8)))
    finally:
        stop.set()
        thread.join(timeout=4)
    save(directory / "terminal.json", dict(node=node, step=step, planned_cards=8, qualified_cards=sum(checks),
                                           cooperative_stop_requested=halted.is_set()))
    return int(not all(checks))


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2]), json.loads(Path(sys.argv[3]).read_bytes())))
