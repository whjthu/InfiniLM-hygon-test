"""Measure independent versus packed BF16 projections on a shared decoder weight."""

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

# 4096x1024 is the measured Qwen projection matrix. These groups bracket the
# earlier M=1,16 graph cases and the recorded native M=186 projection.
GROUPS = ((1, 2, 4, 8), (8, 16, 32, 128), (64, 128, 186, 192))
ORDER = ("separate", "packed", "packed", "separate")
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
    cpu_data = [[torch.randint(-2, 3, (count, K), generator=generator, dtype=torch.int8)
                 for count in rows] for _ in range(2)]
    gold = [[(x.float() @ cpu_weight[:, :16].float()).bfloat16() for x in batch] for batch in cpu_data]
    weight = cpu_weight.to(device, dtype=torch.bfloat16)
    batches = [[x.to(device, dtype=torch.bfloat16) for x in batch] for batch in cpu_data]

    def separate(batch):
        return [torch.mm(x, weight) for x in batch]

    def packed(batch):
        return torch.mm(torch.cat(batch, dim=0), weight).split(rows, dim=0)

    for batch, reference in zip(batches, gold, strict=True):
        left, right = separate(batch), packed(batch)
        if not all(torch.equal(a, b) and torch.equal(a[:, :16].cpu(), c)
                   for a, b, c in zip(left, right, reference, strict=True)):
            raise ValueError("PAIRED_FULL_OUTPUT_OR_CPU_REFERENCE_MISMATCH")
    torch.cuda.synchronize(device)
    del left, right
    values = {"separate": [], "packed": []}
    for label in ORDER:
        arm = separate if label == "separate" else packed
        torch.cuda.reset_peak_memory_stats(device)
        calls = 0
        start = time.monotonic()
        first_sample()
        while time.monotonic() - start < CASE_SECONDS:
            if halted.is_set():
                raise InterruptedError("COOPERATIVE_TRAINING_HANDOFF")
            if time.time() >= deadline:
                raise TimeoutError("BOUNDED_OPERATOR_EPOCH")
            for i in range(8):
                output = arm(batches[i % len(batches)])
            torch.cuda.synchronize(device)
            calls += 8
        del output
        seconds = time.monotonic() - start
        values[label].append(dict(seconds=seconds, batches=calls, rows=calls * sum(rows),
                                  peak_bytes=torch.cuda.max_memory_allocated(device)))
    medians = {arm: statistics.median(s["rows"] / s["seconds"] for s in records)
               for arm, records in values.items()}
    return dict(rows=rows, K=K, N=N, numeric=True, distinct_inputs=2, measurements=values,
                packed_over_separate=medians["packed"] / medians["separate"])


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
                for index, rows in enumerate(GROUPS):
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
