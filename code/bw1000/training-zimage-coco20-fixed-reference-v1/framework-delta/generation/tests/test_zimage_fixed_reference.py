"""Run the pinned upstream loss with real CPU tensor operands."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from gencuda.zimage_fixed_reference import FixedReference, replay_native_loss


LATENT = (1, 16, 32, 32)


def original_loss(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "FlowMatchSFTLoss"]
    scope = {"torch": torch, "BasePipeline": object}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["FlowMatchSFTLoss"]


class Pipeline:
    def __init__(self):
        self.scheduler = self
        self.timesteps = torch.arange(1000, dtype=torch.float32)
        self.torch_dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.in_iteration_models = ("weight",)
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))
        self.calls = []

    def add_noise(self, latents, noise, timestep):
        return latents + noise

    def training_target(self, latents, noise, timestep):
        return noise - latents

    def training_weight(self, timestep):
        self.last_timestep = timestep.clone()
        return 1.0

    def model_fn(self, *, weight, input_latents, latents, cond, prompt, timestep):
        self.calls.append(tuple(t.clone().detach() for t in (input_latents, latents, cond, timestep)))
        assert prompt == "pinned test prompt"
        return weight * latents + cond


def latent(value):
    return torch.full(LATENT, value, dtype=torch.bfloat16)


def fixtures():
    native = {"input_latents": latent(1), "cond": latent(2), "prompt": "pinned test prompt"}
    target = {"training_target": latent(2)}
    frozen = {"inputs": {**native, "input_latents": latent(3), "latents": latent(4), "cond": latent(1)},
              "noise": latent(3), "timestep_id": torch.tensor([726]),
              "timestep": torch.tensor([726], dtype=torch.bfloat16)}
    return native, frozen, target


def main(loss_path):
    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    original = original_loss(loss_path)
    module = SimpleNamespace(FlowMatchSFTLoss=original)
    pipe = Pipeline()
    native, frozen, target = fixtures()
    reference = FixedReference(torch, frozen, target)
    with replay_native_loss(module, reference):
        loss = module.FlowMatchSFTLoss(pipe, **native)
    assert float(loss.detach()) == 9.0
    loss.backward()
    assert float(pipe.weight.grad) == 24.0
    assert len(pipe.calls) == 1 and float(pipe.last_timestep) == 728.0  # BF16 cast of the fixed index
    actual = pipe.calls[0]
    assert all(torch.equal(a, b) for a, b in zip(actual, (
        frozen["inputs"]["input_latents"], frozen["inputs"]["latents"], frozen["inputs"]["cond"],
        frozen["timestep"],
    ), strict=True))
    assert module.FlowMatchSFTLoss is original and reference.applied

    for add_bad in (lambda d: d.update(prompt="another prompt"), lambda d: d.pop("cond")):
        pipe = Pipeline()
        broken = dict(native)
        add_bad(broken)
        try:
            with replay_native_loss(module, FixedReference(torch, frozen, target)):
                module.FlowMatchSFTLoss(pipe, **broken)
        except ValueError:
            assert module.FlowMatchSFTLoss is original and pipe.calls == []
        else:
            raise AssertionError("Mismatched realized inputs entered the model")

    try:
        with replay_native_loss(module, FixedReference(torch, frozen, target)):
            pass
    except RuntimeError:
        assert module.FlowMatchSFTLoss is original
    else:
        raise AssertionError("Unused replay was accepted")
    assert not torch.cuda.is_initialized()
    print(json.dumps({"passed": True, "torch": str(torch.__version__), "positive_checks": 1,
                      "negative_checks": 3, "native_loss_body": True, "synthetic_CPU_fixture": True,
                      "loss": float(loss.detach()), "gradient": 24.0, "actual_GPU_cards": 0}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
