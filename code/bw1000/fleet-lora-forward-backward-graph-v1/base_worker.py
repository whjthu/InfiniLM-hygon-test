"""One real BF16 GEMM survey on all eight Slurm-assigned DCUs on a node."""
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time

import torch


SHAPES = [(m, 4096, n) for m in (1, 16, 128, 512) for n in (1024, 4096, 8192)]
REFERENCE = (4096, 4096, 4096)


def write_json(path, result):
    pending = path.with_suffix('.tmp')
    with pending.open('x', encoding='utf-8') as f:
        json.dump(result, f, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    pending.replace(path)


def measure(torch_device, m, k, n, soak_seconds=0):
    g = torch.Generator(device='cpu').manual_seed(23 + m + k + n)
    a_cpu = torch.randint(-2, 3, (min(m, 8), k), generator=g, dtype=torch.int8)
    b_cpu = torch.randint(-2, 3, (k, min(n, 16)), generator=g, dtype=torch.int8)
    a = torch.randint(-2, 3, (m, k), device=torch_device, dtype=torch.int8).bfloat16()
    b = torch.randint(-2, 3, (k, n), device=torch_device, dtype=torch.int8).bfloat16()
    a[:a_cpu.shape[0]].copy_(a_cpu)
    b[:, :b_cpu.shape[1]].copy_(b_cpu)
    out = torch.empty((m, n), device=torch_device, dtype=torch.bfloat16)
    expected = (a_cpu.float() @ b_cpu.float()).bfloat16()
    before = torch.cuda.Event(enable_timing=True)
    after = torch.cuda.Event(enable_timing=True)
    repeats = 24 if m <= 512 else 12

    def timed(reuse):
        before.record()
        if reuse:
            for _ in range(repeats):
                torch.mm(a, b, out=out)
            current = out
        else:
            for _ in range(repeats):
                current = torch.mm(a, b)
        after.record()
        after.synchronize()
        return before.elapsed_time(after) / repeats, current

    # Both output paths must match the independent CPU exact integer-dot reference.
    for reuse in (False, True):
        _, current = timed(reuse)
        if not torch.equal(current[:expected.shape[0], :expected.shape[1]].cpu(), expected):
            raise ValueError('BF16_CPU_REFERENCE_MISMATCH')
    if soak_seconds:
        # Fixed wall windows and work counts are kept to quantify across-card and sustained drift.
        windows = []
        for _ in range(12):
            begin = time.monotonic()
            count, ms = 0, 0.0
            while time.monotonic() - begin < soak_seconds / 12:
                duration, _ = timed(True)
                count += repeats
                ms += duration * repeats
            wall = time.monotonic() - begin
            windows.append(dict(gemm_calls=count, device_ms=ms, wall_seconds=wall,
                                completed_TFLOPS=count * 2 * m * k * n / (wall * 1e12)))
        return dict(shape=[m,k,n], steady=windows)
    samples = {'allocate': [], 'reuse': []}
    for pair in range(9):
        order = (True, False) if pair % 2 else (False, True)
        for reuse in order:
            ms, _ = timed(reuse)
            samples['reuse' if reuse else 'allocate'].append(ms)
    return dict(shape=[m,k,n], sample_dots=expected.numel(), samples_ms=samples,
                median_ms={k: statistics.median(v) for k, v in samples.items()})


def main(root, epoch):
    torch.set_num_threads(1)
    if torch.version.hip is None or torch.cuda.device_count() != 8:
        raise RuntimeError('EXPECTED_HIP_AND_EIGHT_VISIBLE_DCU')
    node = socket.gethostname().split('.')[0]
    job, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    barrier = threading.Barrier(8, timeout=90)
    stop = threading.Event()
    busy = []

    def sample():
        paths = {p.resolve().parent.name: p for p in Path('/sys/class/drm').glob('renderD*/device/gpu_busy_percent')}
        while not stop.is_set():
            start = time.time()
            try:
                values = {k: int(p.read_text()) for k, p in paths.items()}
                if len(values) == 8 and all(0 <= v <= 100 for v in values.values()):
                    busy.append(dict(epoch=start, busy=values))
                    if len(busy) % 5 == 0:
                        write_json(root/f'live-{node}.json', dict(node=node, step=job+'.'+step,
                                   epoch=start, samples=busy[-5:]))
            except OSError:
                pass
            stop.wait(2)

    def run_card(index):
        device = torch.device('cuda', index)
        with torch.cuda.device(device), torch.no_grad():
            prop = torch.cuda.get_device_properties(index)
            torch.cuda.synchronize(device)
            lead = barrier.wait() == 0
            if lead:
                write_json(root/f'ready-{node}.json', dict(node=node, step=job+'.'+step, visible=8,
                           epoch=time.time(), hip=torch.version.hip, torch=torch.__version__))
            barrier.wait()
            hot = measure(device, *REFERENCE, soak_seconds=96)
            measurements = [measure(device, *shape) for shape in SHAPES]
            torch.cuda.synchronize(device)
            return dict(device=index, model=prop.name, memory_bytes=prop.total_memory,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        steady=hot, measurements=measurements)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    reports, errors = [], []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_card, i): i for i in range(8)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    reports.append(future.result())
                except Exception as exc:
                    errors.append(dict(device=futures[future], type=type(exc).__name__, message=str(exc)[:200]))
                    barrier.abort()
    finally:
        stop.set()
        monitor.join(timeout=5)
    write_json(root/f'final-{node}.json', dict(node=node, step=job+'.'+step, launch_epoch=epoch,
               finished_epoch=time.time(), planned_cards=8, cards=reports, error=errors, driver_busy=busy))
    return bool(errors) or len(reports) != 8


if __name__ == '__main__':
    raise SystemExit(int(main(Path(sys.argv[1]), float(sys.argv[2]))))
