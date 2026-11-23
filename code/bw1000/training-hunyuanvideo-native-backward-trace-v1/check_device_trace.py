"""Exercise the installed profiler against a small BF16 backward on its assigned DCU."""

import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from assigned_kfd import assigned_kfd
from hunyuanvideo_backward_trace import BackwardTrace, write_json
from runtime_pci import runtime_pci


def main():
    run = Path(sys.argv[1])
    parent, node = sys.argv[2:]
    step = os.environ.get("SLURM_STEP_ID", "")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (not step.isdecimal() or os.environ.get("SLURM_JOB_ID") != parent
            or socket.gethostname().split(".", 1)[0] != node
            or not re.fullmatch(r"0|[1-9][0-9]*", selector)):
        raise ValueError("Unexpected single-DCU Slurm step")
    step = parent + "." + step
    gate = json.loads((run / "gpu-gate.json").read_bytes())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (gate["step"] != step or gate["node"] != node or gate["boot_id"] != boot
            or gate["kfd_holder_count"] != 0 or not 0 <= time.monotonic_ns() - gate["monotonic_ns"] <= 30_000_000_000):
        raise ValueError("Device gate is stale or differs")
    pci = runtime_pci()
    if pci not in gate["pci"]:
        raise ValueError("Assigned PCI is outside the audited node")
    write_json(run / "assigned-device.json", {
        "step": step, "boot_id": boot, "device": assigned_kfd(pci),
    })

    import torch

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError("Expected one HIP device")
    capture = BackwardTrace(torch.profiler, run, training_step=None)
    left, right = (torch.ones((256, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(2))
    loss = (left @ right).float().sum()
    torch.cuda.synchronize()
    with capture.capture():
        loss.backward()
    if any(tensor.grad is None or not bool((tensor.grad == 256).all().item()) for tensor in (left, right)):
        raise ValueError("Original Torch test backward returned an unexpected gradient")
    write_json(run / "probe-complete.json", {
        "step": step, "torch_version": str(torch.__version__), "hip_version": str(torch.version.hip),
        "expected_gradients_verified": 2, "actual_device_trace": True,
    })


if __name__ == "__main__":
    main()
