"""Probe the deployed profiler metadata ABI with a real CPU matrix multiply."""

import json
import os
import socket

import torch


torch.set_num_threads(1)
left = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
right = left.transpose(0, 1)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True,
                            profile_memory=False, with_stack=False) as profile:
    product = torch.mm(left, right)

result = profile.profiler.kineto_results
flat = [event for event in result.events() if event.name() == "aten::mm"]
if len(flat) != 1:
    raise ValueError("expected one physical CPU aten::mm observation")
host = flat[0]
identifier = host.correlation_id()
if (identifier <= 0 or str(host.device_type()) != "DeviceType.CPU"
        or tuple(map(tuple, host.shapes()[:2])) != ((64, 64), (64, 64))):
    raise ValueError("flat CPU matrix operator has incorrect identity or shape")

pending, matches, count = list(result.experimental_event_tree()), [], 0
while pending:
    event = pending.pop()
    count += 1
    if count > 512:
        raise ValueError("CPU metadata tree exceeds bound")
    pending.extend(event.children)
    if event.name == "aten::mm" and event.correlation_id == identifier:
        matches.append([(tuple(t.sizes), tuple(t.strides), str(t.dtype), str(t.device), str(t.layout))
                        for t in event.extra_fields.inputs[:2]])

if matches != [[((64, 64), (64, 1), "torch.float32", "cpu", "torch.strided"),
                ((64, 64), (1, 64), "torch.float32", "cpu", "torch.strided")]]:
    raise ValueError("target metadata tree lost operand order, layout or strides")
if torch.cuda.is_initialized() or not torch.isfinite(product).all().item():
    raise ValueError("CPU-only runtime did not meet the finite-result or GPU initialization guard")

print(json.dumps({
    "passed": True, "node": socket.gethostname().split(".", 1)[0],
    "step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
    "torch": str(torch.__version__), "hip_build_version": torch.version.hip,
    "device": "cpu", "gpu_initialized": False, "metadata_joins": len(matches),
    "positive_correlation_id": True, "operands": matches[0],
    "qualifies_gpu_correlation": False,
}, allow_nan=False))
