"""Compare the staged SAGE scheduler on one assigned DCU with the frozen CPU replay."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

import torch
from assigned_kfd import assigned_kfd

root = Path(__file__).resolve().parents[1]
intent = json.loads((root / "intent.json").read_bytes())
for name, digest in intent["source_files"].items():
    assert hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() == digest
reference = json.loads((root / "src/CPU_REFERENCE.json").read_bytes())["CPU_result"]
node = socket.gethostname().split(".")[0]
step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
gate = json.loads((root / "gpu-gate" / (node + ".json")).read_bytes())
assert gate["step"] == step and gate["kfd_holder_count"] == 0
assert gate["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert torch.version.hip == reference["torch_hip"] and torch.__version__ == reference["torch"]
assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
pci_buffer = ctypes.create_string_buffer(32)
hip = ctypes.CDLL("libamdhip64.so")
get_pci = hip.hipDeviceGetPCIBusId
get_pci.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int)
get_pci.restype = ctypes.c_int
assert get_pci(pci_buffer, len(pci_buffer), 0) == 0
pci = pci_buffer.value.decode().lower()
if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
    pci = "0000:" + pci
assert pci in gate["pci"]
kfd = assigned_kfd(pci)

path = Path("/sage/hyvideo/schedulers/scheduling_flow_match_discrete.py")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == reference["scheduler_SHA256"]
spec = importlib.util.spec_from_file_location("gencuda_sage_scheduler", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert sys.modules["diffusers"].__file__ == reference["diffusers_path"]
assert torch.__file__ == reference["torch_path"]

device = torch.device("cuda:0")
scheduler = module.FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
scheduler.set_timesteps(4, device=device)
sigmas = scheduler.sigmas.to(device)
assert scheduler.timesteps.is_cuda
assert scheduler.timesteps.tolist() == reference["timesteps"] and sigmas.tolist() == reference["sigmas"]
weights = scheduler.compute_grad_balancing_factors(eta=.7, device=device, sde_type="sage_grpo",
                                                   sde_solver=True, sigmas=sigmas)
assert weights.is_cuda
timesteps = sigmas[:-1] * scheduler.config.num_train_timesteps
sample = torch.linspace(-.375, .375, 16).reshape(1, 2, 2, 2, 2).to(device)
previous = sample.flip(-1) * .5
rows = []
for index, timestep in enumerate(timesteps):
    replay = module.FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
    replay.num_inference_steps = 4
    replay.sigmas = sigmas
    replay.timesteps = timesteps
    replay._step_index = index
    prediction = (sample * .25 + .1).requires_grad_()
    next_sample, _, log_prob, mean, std = replay.sde_step_with_logprob(
        prediction, sample, timestep, .7, prev_sample=previous, grpo=True,
        sde_solver=True, sde_type="sage_grpo")
    gradient, = torch.autograd.grad(log_prob.sum(), prediction)
    assert all(t.is_cuda for t in (next_sample, log_prob, mean, std, gradient))
    assert torch.equal(next_sample, previous) and replay.step_index == index + 1
    assert log_prob.shape == (1,) and gradient.shape == sample.shape
    assert torch.count_nonzero(gradient).item() > 0
    rows.append(dict(index=index, log_prob=log_prob.item(), mean_sum=mean.sum().item(),
                     std=std.item(), gradient=gradient.flatten().tolist()))
torch.cuda.synchronize()

limits = {"balancing": (1e-7, 2e-5), "log_prob": (1e-7, 2e-5), "mean_sum": (1e-7, 2e-5),
          "std": (1e-7, 2e-5), "gradient": (2e-8, 2e-5)}
differences = {}
def compare(name, observed, expected):
    actual = torch.tensor(observed, dtype=torch.float64)
    target = torch.tensor(expected, dtype=torch.float64)
    atol, rtol = limits[name]
    error = (actual - target).abs()
    differences[name] = max(differences.get(name, 0), error.max().item())
    return bool(actual.shape == target.shape and torch.isfinite(actual).all()
                and (error <= atol + rtol * target.abs()).all())

checks = [compare("balancing", weights.tolist(), reference["balancing"])]
for actual, expected in zip(rows, reference["rows"], strict=True):
    assert actual["index"] == expected["index"]
    checks.extend(compare(name, actual[name], expected[name]) for name in limits if name != "balancing")
result = dict(passed=all(checks), step=step, pci=pci, kfd=kfd, scheduler_SHA256=digest,
              fixture="synthetic_FP32_HIP", rows=rows, balancing=weights.tolist(),
              comparisons=len(checks), max_absolute_error=differences, tolerance_atol_rtol=limits)
with (root / "device.json").open("x") as out:
    json.dump(result, out, indent=2, allow_nan=False)
assert result["passed"], differences
