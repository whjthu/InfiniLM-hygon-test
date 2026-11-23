"""Exercise the recorded-update path with two real, in-place COCO20 operand archives."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


COMMON = Path("/common-framework/generation/src/gencuda")
SOURCE = Path("/replay")
EXPECTED_PROVIDER = {
    "backends/zimage_training.py": "6526be513a7b55977ede0dd176757da1530fffedaefa0080639e4191b1f49d31",
    "hy_resume_checkpoint.py": "aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7",
    "vggt_first_update.py": "57025552b5105996b32a845c29fe0e5324820a1d27dc5100ebac14d16165b71a",
}
PROGRESS = "a1694d2ffc3491d5c89d896cd73260f9ded215bd83e6ea5eba78e5b14403d2d0"


def load_function(path, name, scope):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(tree.body) == 1
    exec(compile(tree, str(path), "exec"), scope)
    return scope[name]


def main():
    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    for name, expected in EXPECTED_PROVIDER.items():
        assert hashlib.sha256((COMMON / name).read_bytes()).hexdigest() == expected, name
    gencuda = ModuleType("gencuda")
    gencuda.__path__ = [str(SOURCE / "gencuda"), str(COMMON)]
    sys.modules[gencuda.__name__] = gencuda
    from gencuda.hy_resume_checkpoint import tree_sha256
    from gencuda.zimage_first_update import instrument_update, recorded_native_update
    from gencuda.zimage_fixed_reference import load_fixed_reference

    archive = load_fixed_reference(torch, Path("/original"), PROGRESS)
    fixed = archive.reference
    assert tree_sha256(fixed["inputs"]) == "ba99b84f02f1b00085d4d56e5fb0b94ac14bdd19a5f4a2ec9762783dd41265ee"
    assert tree_sha256(fixed["training_target"]) == "ba591abf55810d1d50ee715987773927994800c0fb81bdf1da24b74b6e6e5a52"
    source_digest = tree_sha256(fixed)
    native = SimpleNamespace(FlowMatchSFTLoss=load_function(
        SOURCE / "loss.py", "FlowMatchSFTLoss", {"BasePipeline": object, "torch": torch},
    ))
    update = load_function(COMMON / "backends/zimage_training.py", "_train_step",
                           {"Any": Any, "time": time, "require_cuda": lambda: torch, "synchronize": lambda: None})
    native_update = instrument_update(update)
    original = native.FlowMatchSFTLoss
    events = []

    class Pipeline:
        def __init__(self, weight):
            self.scheduler = self
            self.timesteps = torch.arange(1000)
            self.torch_dtype = fixed["timestep"].dtype
            self.device = torch.device("cpu")
            self.in_iteration_models = ("weight",)
            self.weight = weight

        def add_noise(self, latents, noise, timestep):
            return (latents + noise).to(fixed["inputs"]["latents"].dtype)

        def training_target(self, latents, noise, timestep):
            return (noise - latents).to(fixed["training_target"].dtype)

        def training_weight(self, timestep):
            assert torch.equal(timestep, fixed["timestep"])
            return 1.0

        def model_fn(self, *, weight, timestep, **inputs):
            assert tree_sha256(inputs) == tree_sha256(fixed["inputs"])
            assert torch.equal(timestep, fixed["timestep"])
            events.append("model_forward")
            return weight * inputs["latents"]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))
            self.pipe = Pipeline(self.weight)

        def forward(self, inputs):
            return native.FlowMatchSFTLoss(self.pipe, **inputs)

    class Observer:
        def guard(self, call):
            return call()

        def enter(self, step, data):
            assert step == 1 and "latents" not in data
            events.append("enter")

        def prepared(self, inputs, noise, timestep_id, timestep, target):
            actual = dict(inputs=inputs, noise=noise, timestep_id=timestep_id, timestep=timestep,
                          training_target=target)
            assert tree_sha256(actual) == source_digest
            assert inputs["latents"].data_ptr() != fixed["inputs"]["latents"].data_ptr()
            events.append("prepared")

        def forward_value(self, prediction, loss):
            assert tuple(prediction.shape) == (1, 16, 32, 32)
            assert torch.isfinite(loss).item()
            events.append("recorded_forward")

        def gradients(self, stage, norm=None):
            assert torch.isfinite(model.weight.grad).item() and torch.count_nonzero(model.weight.grad).item() == 1
            events.append(stage)

        def updated(self):
            assert int(optimizer.state[model.weight]["step"].item()) == 1
            events.append("updated")

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = model.weight.detach().clone()
    inputs = {key: value for key, value in fixed["inputs"].items() if key != "latents"}
    inputs["input_latents"] = inputs["input_latents"].add(1)
    assert tree_sha256(inputs["input_latents"]) != tree_sha256(fixed["inputs"]["input_latents"])
    output = recorded_native_update(native_update, native, Observer(), 1, model, optimizer,
                                    inputs, 20260912, 1.0, fixed_reference=archive)
    assert events == ["enter", "prepared", "model_forward", "recorded_forward", "preclip", "postclip", "updated"]
    assert archive.applied and native.FlowMatchSFTLoss is original and not torch.equal(initial, model.weight)
    assert tree_sha256(fixed) == source_digest and not torch.cuda.is_initialized()
    print(json.dumps({
        "passed": True, "torch": str(torch.__version__), "original_progress_SHA256": PROGRESS,
        "prepared_input_fields": len(fixed["inputs"]), "events": events,
        "two_archives": ["prepared.pt", "target.pt"], "same_runtime_real_archive": True,
        "private_bytes_exported": 0, "actual_GPU_cards": 0, "actual_model_updates": 0,
        "cpu_fixture_optimizer_steps": 1, "loss": output["loss"], "grad_norm": output["grad_norm"],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
