"""Exercise the committed checkpoint helpers in three HIP processes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch

def require_cuda():
    assert torch.version.hip and torch.cuda.is_available()
    return torch

root = Path(__file__).resolve().parents[1]
source = root / "src"
mode = sys.argv[1] if len(sys.argv) == 2 else "supervisor"
modes = ("continuous", "prefix", "restored")

def read(name):
    return json.loads((root / name).read_bytes())

def save(name, value):
    with (root / name).open("x") as output:
        json.dump(value, output, allow_nan=False, indent=2)

if mode == "supervisor":
    intent = read("intent.json")
    for name, digest in intent["source_files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    assert not torch.cuda.is_initialized()
    for arm in modes:
        subprocess.run([sys.executable, "-B", __file__, arm], check=True, timeout=45)
    control, prefix, resumed = [read(arm + ".json") for arm in modes]
    assert len({control["pid"], prefix["pid"], resumed["pid"], os.getpid()}) == 4
    assert prefix["after"] == control["boundary"] == resumed["boundary"]
    assert control["after"] == resumed["after"] and control["loss"] == resumed["loss"]
    assert prefix["after"]["adapter"] != resumed["after"]["adapter"]
    assert prefix["after"]["optimizer"] != resumed["after"]["optimizer"]
    assert all(row["step"] == control["step"] and row["devices"] == 2 and row["pci"] == control["pci"]
               for row in (prefix, resumed))
    save("device.json", dict(passed=True, step=control["step"], arms=[control, prefix, resumed],
                             fixture="synthetic_two_DCU_native_Muon_fresh_process",
                             checkpoint_SHA256=hashlib.sha256((root / "checkpoint-15/state.pt").read_bytes()).hexdigest()))
    raise SystemExit(0)

assert mode in modes and torch.version.hip and torch.cuda.device_count() == 2
sys.path.insert(0, "/hunyuan")
muon_path = Path("/hunyuan/hyvideo/optim/muon.py")
assert hashlib.sha256(muon_path.read_bytes()).hexdigest() == "11862efb0b26bf308d313e4e18c7fcc0dd6934a275c640862ad7769b0bdfaccd"
from hyvideo.optim.muon import get_muon_optimizer
from assigned_kfd import assigned_kfd
import ctypes
import re
import socket

step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
gate = read("gpu-gate/" + socket.gethostname().split(".")[0] + ".json")
assert gate["step"] == step and gate["kfd_holder_count"] == 0
assert gate["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
hip = ctypes.CDLL("libamdhip64.so")
pci_fn = hip.hipDeviceGetPCIBusId
pci_fn.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int)
pci_fn.restype = ctypes.c_int
devices = []
for rank in range(2):
    buf = ctypes.create_string_buffer(32)
    assert pci_fn(buf, len(buf), rank) == 0
    pci = buf.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        pci = "0000:" + pci
    assert pci in gate["pci"]
    devices.append(assigned_kfd(pci))
assert len({row["gpu_id"] for row in devices}) == 2

def load_functions(file, names):
    tree = ast.parse((source / file).read_bytes())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), file, "exec"), globals())

load_functions("training.py", ("_adapter_state", "_cpu_tree", "_optimizer_to_device"))
load_functions("sage.py", ("_adapter_parameter_name", "_resolve_adapter_parameter_names",
                            "_load_state", "_save_rl_checkpoint", "_load_rl_checkpoint"))
random.seed(143)
np.random.seed(143)
torch.manual_seed(143)
torch.cuda.manual_seed_all(143)
model = torch.nn.Module()
model.register_parameter("lora", torch.nn.Parameter(torch.ones((4, 16), device="cuda:0")))
optimizer = get_muon_optimizer(model, lr=1e-3, weight_decay=.01)
checkpoint = root / "checkpoint-15/state.pt"
contract = {"schema": "sage-policy-resume-v1", "warmup_steps": 5}
anchor = _adapter_state(model)

def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def fingerprint():
    return dict(adapter=tensor_digest(model.lora), anchor=tensor_digest(anchor["lora"]),
                optimizer=tensor_digest(optimizer.state[model.lora]["momentum_buffer"]),
                python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                numpy=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),
                cpu=tensor_digest(torch.get_rng_state()),
                gpu=[tensor_digest(torch.cuda.get_rng_state(rank)) for rank in (0, 1)])

def update(global_step):
    actor = torch.rand((4, 16), device="cuda:0")
    reward = torch.rand((), device="cuda:1").to("cuda:0")
    scale = random.random() + float(np.random.rand()) + torch.rand(()).item()
    optimizer.zero_grad(set_to_none=True)
    loss = (model.lora * actor * (reward + scale + global_step * .01)).square().mean()
    loss.backward()
    optimizer.step()
    return loss.item()

if mode == "restored":
    update(99)
    cursor, anchor = _load_rl_checkpoint(checkpoint, torch, model, optimizer, contract, max_measured_step=20)
    assert cursor == 15
else:
    update(20)
boundary = fingerprint()
if mode == "prefix":
    _save_rl_checkpoint(checkpoint, torch, model, optimizer, anchor, 15, contract)
    loss = None
else:
    loss = update(21)
for rank in (0, 1):
    torch.cuda.synchronize(rank)
save(mode + ".json", dict(step=step, pid=os.getpid(), devices=2, pci=[row["pci"] for row in devices],
                          boundary=boundary, after=fingerprint(), loss=loss))
