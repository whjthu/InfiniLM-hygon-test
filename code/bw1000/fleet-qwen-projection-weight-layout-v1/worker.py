"""Measure resident BF16 weight layout at the fixed Qwen projection shape."""

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

# This shape had a positive sync-depth result on all 608 observed cards.
M = 184
ORDINARY, PACKED = "checkpoint_transpose_view", "pretransposed_contiguous"
ORDER = (ORDINARY, PACKED, PACKED, ORDINARY)
K, N = 4096, 1024
DEPTH = 64
CASE_SECONDS = 5.0


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def compare(device, seed, deadline, first_sample, halted):
    import torch

    torch.cuda.synchronize(device)
    generator = torch.Generator().manual_seed(seed)
    cpu_weight = torch.randint(-2, 3, (N, K), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (M, K), generator=generator, dtype=torch.int8)
                  for _ in range(2)]
    gold = [(x.float() @ cpu_weight[:16].float().T).bfloat16() for x in cpu_inputs]
    ordinary = cpu_weight.to(device, dtype=torch.bfloat16).T
    inputs = [x.to(device, dtype=torch.bfloat16) for x in cpu_inputs]
    outputs = [torch.empty((M, N), device=device, dtype=torch.bfloat16) for _ in inputs]
    references = [torch.mm(x, ordinary) for x in inputs]
    torch.cuda.synchronize(device)
    memory_before_pack = torch.cuda.memory_allocated(device)
    start = time.monotonic()
    packed = ordinary.contiguous()
    torch.cuda.synchronize(device)
    pack_seconds = time.monotonic() - start
    memory_after_pack = torch.cuda.memory_allocated(device)
    if (tuple(ordinary.stride()) != (1, K) or tuple(packed.stride()) != (N, 1)
            or ordinary.data_ptr() == packed.data_ptr() or not torch.equal(ordinary, packed)
            or torch.equal(inputs[0], inputs[1]) or torch.equal(references[0], references[1])):
        raise ValueError("WEIGHT_LAYOUT_OR_CHANGED_INPUT_DID_NOT_REACH_OUTPUT")
    weights = {ORDINARY: ordinary, PACKED: packed}

    def submit(weight):
        for i in range(DEPTH):
            j = i % len(inputs)
            torch.mm(inputs[j], weight, out=outputs[j])
        torch.cuda.synchronize(device)

    def check():
        if not all(torch.equal(result, expected) and torch.equal(result[:, :16].cpu(), cpu)
                   for result, expected, cpu in zip(outputs, references, gold, strict=True)):
            raise ValueError("PREALLOCATED_OUTPUT_OR_CPU_REFERENCE_MISMATCH")

    for weight in weights.values():
        submit(weight)
        check()

    values = {mode: [] for mode in weights}
    for mode in ORDER:
        torch.cuda.reset_peak_memory_stats(device)
        drains = 0
        start = time.monotonic()
        first_sample()
        while time.monotonic() - start < CASE_SECONDS:
            if halted.is_set():
                raise InterruptedError("COOPERATIVE_TRAINING_HANDOFF")
            if time.time() >= deadline:
                raise TimeoutError("BOUNDED_OPERATOR_EPOCH")
            submit(weights[mode])
            drains += 1
        seconds = time.monotonic() - start
        check()
        calls = drains * DEPTH
        values[mode].append(dict(seconds=seconds, drains=drains, projections=calls, rows=calls*M,
                                 peak_bytes=torch.cuda.max_memory_allocated(device)))
    medians = {mode: statistics.median(s["projections"] / s["seconds"] for s in records)
               for mode, records in values.items()}
    saving = 1/medians[ORDINARY] - 1/medians[PACKED]
    return dict(M=M, K=K, N=N, numeric=True, distinct_inputs=2, sync_depth=DEPTH,
                arm_order=list(ORDER),
                layout_strides={mode:list(weights[mode].stride()) for mode in weights},
                packed_logical_bytes=K*N*2, pack_wall_s=pack_seconds,
                extra_resident_allocated_bytes=memory_after_pack-memory_before_pack,
                measurements=values, median_projections_per_s=medians,
                packed_over_view=medians[PACKED]/medians[ORDINARY],
                amortized_projections=(pack_seconds/saving if saving>0 else None))


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
                results.append(compare(device, seed + card * 13, last, active, halted))
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
                                           torch_version=str(torch.__version__),hip_version=str(torch.version.hip),
                                           cooperative_stop_requested=halted.is_set()))
    return int(not all(checks))


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2]), json.loads(Path(sys.argv[3]).read_bytes())))
