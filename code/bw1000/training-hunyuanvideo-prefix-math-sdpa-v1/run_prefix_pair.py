"""Compare original masked math SDPA with prefix math SDPA on one assigned HIP GPU."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

from assigned_kfd import assigned_kfd
from prefix_sdpa import PrefixLayout, torch_attention
from runtime_pci import runtime_pci


ORDER = ((False, True), (True, False), (False, True))
NUMERICAL_LIMIT = 0.01


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def assigned_device(run, parent, node):
    step = os.environ.get("SLURM_STEP_ID", "")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (not step.isdecimal() or os.environ.get("SLURM_JOB_ID") != parent
            or socket.gethostname().split(".", 1)[0] != node
            or re.fullmatch(r"0|[1-9][0-9]*", selector) is None):
        raise ValueError("Unexpected single-DCU Slurm step")
    handle = parent + "." + step
    gate = json.loads((run / "gpu-gate.json").read_bytes())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (gate["step"] != handle or gate["node"] != node or gate["boot_id"] != boot
            or gate["kfd_holder_count"] != 0 or not 0 <= time.monotonic_ns() - gate["monotonic_ns"] <= 30_000_000_000):
        raise ValueError("The assigned GPU gate is stale or differs")
    pci = runtime_pci()
    if pci not in gate["pci"]:
        raise ValueError("Assigned device is outside the audited node")
    save(run / "assigned-device.json", {"step": handle, "boot_id": boot, "device": assigned_kfd(pci)})
    return handle


def inputs(torch, layout, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bshd = (1, layout.length, layout.heads, layout.width)

    def tensor():
        return torch.randn(bshd, generator=gen, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    operands = tuple(tensor() for _ in range(3))
    upstream = tensor()
    mask = (torch.arange(layout.length, device="cuda") < layout.valid).view(1, 1, 1, -1)
    if any(x.shape != layout.shape or x.stride() != layout.stride for x in (*operands, upstream)):
        raise ValueError("The reproduced fixture strides differ")
    return operands, upstream, mask


def evaluate(torch, fixture, layout, prefix):
    operands, upstream, mask = fixture
    qkv = tuple(t.detach().requires_grad_(True) for t in operands)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start, fwd, stop = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    host_start = time.perf_counter()
    start.record()
    y = torch_attention(torch.nn.functional, *qkv, mask, layout, use_prefix=prefix)
    fwd.record()
    gradients = torch.autograd.grad(y, qkv, upstream, create_graph=False)
    stop.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - host_start
    stats = {"forward_device_ms": start.elapsed_time(fwd), "backward_device_ms": fwd.elapsed_time(stop),
             "forward_backward_wall_seconds": wall,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
             "starting_allocated_bytes": allocated}
    if (not all(math.isfinite(stats[k]) and stats[k] > 0 for k in (
            "forward_device_ms", "backward_device_ms", "forward_backward_wall_seconds"))
            or stats["peak_allocated_bytes"] < allocated):
        raise ValueError("Invalid duration or allocator evidence")
    return (y.detach(), *gradients), stats


def compare(torch, reference, candidate, valid):
    diagnostics = {}
    for name, a, b in zip(("output", "dq", "dk", "dv"), reference, candidate, strict=True):
        if (not bool(torch.isfinite(a).all().item()) or not bool(torch.isfinite(b).all().item())
                or torch.count_nonzero(a[..., valid:, :]).item() or torch.count_nonzero(b[..., valid:, :]).item()):
            raise ValueError("Nonfinite values or nonzero padding in output/gradient")
        a32, b32 = a.float(), b.float()
        reference_norm = float(torch.linalg.vector_norm(a32).item())
        error = float(torch.linalg.vector_norm(a32 - b32).item()) / max(reference_norm, 1e-12)
        max_abs = float((a32 - b32).abs().max().item())
        diagnostics[name] = {"relative_l2": error, "absolute_max": max_abs, "padding_zero": True,
                             "passed": math.isfinite(error) and error <= NUMERICAL_LIMIT}
    return diagnostics


def main():
    run = Path(sys.argv[1])
    parent, node = sys.argv[2:]
    step = assigned_device(run, parent, node)
    raw = (run / "layout.json").read_bytes()
    scope = json.loads(raw)
    layout = PrefixLayout(**scope["layout"])

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError("Expected the original one-device HIP runtime")
    records = []
    prior = inputs(torch, layout, 7021)
    with sdpa_kernel(SDPBackend.MATH):
        for mode in (False, True):
            _, _stats = evaluate(torch, prior, layout, mode)
            del _
        del prior
        for pair, arms in enumerate(ORDER, 1):
            fixture = inputs(torch, layout, 7021 + pair)
            first, first_cost = evaluate(torch, fixture, layout, arms[0])
            second, second_cost = evaluate(torch, fixture, layout, arms[1])
            lut = {arms[0]: (first, first_cost), arms[1]: (second, second_cost)}
            original, baseline = lut[False]
            compact, candidate = lut[True]
            checks = compare(torch, original, compact, layout.valid)
            row = {"pair": pair, "order": ["prefix" if a else "baseline" for a in arms],
                   "baseline": baseline, "prefix": candidate, "checks": checks}
            save(run / f"pair-{pair}.json", row)
            records.append(row)
            del first, second, original, compact, lut, fixture
    save(run / "probe-complete.json", {"step": step, "layout_SHA256": hashlib.sha256(raw).hexdigest(),
         "numerical_tolerance_relative_l2": NUMERICAL_LIMIT, "complete_pairs": len(records),
         "torch_version": str(torch.__version__), "HIP_version": str(torch.version.hip),
         "all_numerics_passed": all(x["passed"] for row in records for x in row["checks"].values()),
         "workload": "synthetic BF16 tensors using the native attention layout", "formal_model_performance": False})


if __name__ == "__main__":
    main()
