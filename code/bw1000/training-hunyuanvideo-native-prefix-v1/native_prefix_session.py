"""Opt-in guarded attention and same-input first-update comparison for native T09."""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace


ENV = "HUNYUAN_T09_NATIVE_PREFIX"
EXPECTED_CALLS = 54
RELATIVE_L2_LIMIT = 0.01
MAX_TENSOR_RELATIVE_L2_LIMIT = 0.05


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def cpu_copy(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def tensor_digest(torch, tensor):
    snapshot = cpu_copy(tensor)
    body = snapshot.view(torch.uint8).numpy().tobytes()
    return sha256(body).hexdigest()


class NativePrefixSession:
    def __init__(self, torch, numpy, output, model, optimizer):
        self.torch, self.numpy = torch, numpy
        self.output = Path(output)
        self.model, self.optimizer = model, optimizer
        self.trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.frozen = {name: p for name, p in model.named_parameters() if not p.requires_grad}
        self.buffers = {name: p for name, p in model.named_buffers()}
        if not self.trainable or any("lora_" not in name for name in self.trainable):
            raise ValueError("Unexpected native trainable scope")
        self.layout = PrefixLayout(16, 5326, 128, 4785)
        self.expected_mask = (torch.arange(self.layout.length, device="cuda") < self.layout.valid).view(1, 1, 1, -1)
        self.pending = []
        self.mode = None
        self.first = None
        self.position = 0
        self.phase_counts = {}
        self.receipts = self.output / "native-prefix"
        self.receipts.mkdir(mode=0o700)
        save(self.receipts / "intent.json", {
            "shape": self.layout.shape, "stride": self.layout.stride, "valid": self.layout.valid,
            "planned_updates": 25, "expected_mask_checks_per_phase": EXPECTED_CALLS,
            "same_input_update": 1, "formal_A100": False,
        })

    def eligible(self, args, kwargs):
        mask = kwargs.get("attn_mask")
        if len(args) != 3 or set(kwargs) != {"attn_mask", "dropout_p", "is_causal"}:
            return False
        if kwargs["dropout_p"] != 0 or kwargs["is_causal"] is not False or mask is None:
            return False
        if (tuple(mask.shape) != (1, 1, 1, self.layout.length) or mask.dtype != self.torch.bool
                or mask.device != self.expected_mask.device or not mask.is_contiguous()):
            return False
        return all(tuple(t.shape) == self.layout.shape and tuple(t.stride()) == self.layout.stride
                   and t.dtype == self.torch.bfloat16 and t.device == self.expected_mask.device for t in args)

    @contextmanager
    def intercept(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        functional = self.torch.nn.functional
        original = functional.scaled_dot_product_attention
        implementation = SimpleNamespace(scaled_dot_product_attention=original, pad=functional.pad)

        def call(*args, **kwargs):
            if not self.eligible(args, kwargs):
                return original(*args, **kwargs)
            if len(self.pending) >= EXPECTED_CALLS:
                raise ValueError("The native attention phase exceeded its frozen call count")
            mask = kwargs["attn_mask"]
            self.pending.append((mask == self.expected_mask).all())
            with sdpa_kernel(SDPBackend.MATH):
                return torch_attention(implementation, *args, mask, self.layout, use_prefix=True)

        functional.scaled_dot_product_attention = call
        try:
            yield
        finally:
            functional.scaled_dot_product_attention = original

    def check_phase(self, phase):
        if self.mode != "prefix":
            return
        checks, self.pending = self.pending, []
        if len(checks) != EXPECTED_CALLS or not bool(self.torch.stack(checks).all().item()):
            save(self.receipts / "mask-rejected.json", {"global_step": self.position,
                 "phase": phase, "observed_calls": len(checks), "expected_calls": EXPECTED_CALLS})
            raise ValueError("Actual native attention mask differs from the frozen prefix")
        self.phase_counts[phase] = len(checks)

    def inputs(self, **values):
        if self.first is not None:
            self.first["inputs"] = {name: tensor_digest(self.torch, tensor) for name, tensor in values.items()}

    def forward(self, prediction, loss):
        self.check_phase("forward")
        if self.first is not None:
            self.first["forward"] = {"prediction": cpu_copy(prediction), "loss": cpu_copy(loss)}

    def backward(self):
        self.check_phase("backward_recompute")
        self.gradients("unclipped_gradients")

    def gradients(self, key):
        if self.first is None:
            return
        if any(p.grad is None for p in self.trainable.values()):
            raise ValueError("A native trainable has no gradient")
        self.first[key] = {name: cpu_copy(p.grad) for name, p in self.trainable.items()}

    def updated(self):
        if self.first is None:
            return
        self.first["parameters"] = {name: cpu_copy(p) for name, p in self.trainable.items()}
        slots, scalars = {}, {}
        for name, p in self.trainable.items():
            for key, value in self.optimizer.state[p].items():
                label = f"{name}.{key}"
                if self.torch.is_tensor(value):
                    slots[label] = cpu_copy(value)
                elif type(value) in (bool, int, float):
                    scalars[label] = value
                else:
                    raise ValueError("Unexpected Muon state type")
        self.first["optimizer_tensors"], self.first["optimizer_scalars"] = slots, scalars

    def snapshot(self):
        torch = self.torch
        opt = deepcopy(self.optimizer.state_dict())
        if any(torch.is_tensor(v) for state in opt["state"].values() for v in state.values()):
            raise ValueError("Same-input pairing expects a fresh native optimizer")
        total_buffer_bytes = sum(p.numel() * p.element_size() for p in self.buffers.values())
        if total_buffer_bytes > 128 * 1024**2:
            raise ValueError("Native model buffer snapshot exceeds the first-step budget")
        return {
            "parameters": {name: cpu_copy(p) for name, p in self.trainable.items()},
            "buffers": {name: cpu_copy(p) for name, p in self.buffers.items()},
            "optimizer": opt,
            "frozen_versions": {name: p._version for name, p in self.frozen.items()},
            "cpu_rng": torch.get_rng_state(),
            "device_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": self.numpy.random.get_state(),
        }

    def restore(self, state):
        torch = self.torch
        current_parameters = dict(self.model.named_parameters())
        current_buffers = dict(self.model.named_buffers())
        if (current_buffers.keys() != self.buffers.keys()
                or any(current_buffers[n] is not p for n, p in self.buffers.items())
                or current_parameters.keys() != self.trainable.keys() | self.frozen.keys()
                or any(current_parameters[n] is not p for n, p in self.trainable.items())
                or any(current_parameters[n] is not p or p._version != state["frozen_versions"][n]
                       for n, p in self.frozen.items())):
            raise ValueError("Native pairing changed frozen tensors or rebound model state")
        with torch.no_grad():
            for actual, key in ((self.trainable, "parameters"), (self.buffers, "buffers")):
                for name, tensor in actual.items():
                    tensor.copy_(state[key][name])
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if self.optimizer.state_dict() != state["optimizer"]:
            raise ValueError("Restored Muon initial state differs")
        if any(not torch.equal(cpu_copy(p), state["parameters"][name]) for name, p in self.trainable.items()):
            raise ValueError("Restored LoRA parameters differ")
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["device_rng"])
        random.setstate(state["python_rng"])
        self.numpy.random.set_state(state["numpy_rng"])

    def compare_group(self, reference, candidate, initial=None):
        if reference.keys() != candidate.keys() or not reference:
            raise ValueError("Paired native tensor names differ or are absent")
        error_sum = ref_sum = maximum = 0.0
        for name, a in reference.items():
            b = candidate[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("Paired native tensor shape or dtype differs")
            a, b = a.float(), b.float()
            if initial is not None:
                start = initial[name].float()
                a, b = a - start, b - start
            if not bool(self.torch.isfinite(a).all()) or not bool(self.torch.isfinite(b).all()):
                raise ValueError("Native numerical pair contains nonfinite values")
            err = float(self.torch.linalg.vector_norm(b - a).item())
            norm = float(self.torch.linalg.vector_norm(a).item())
            maximum = max(maximum, err / max(norm, 1e-12))
            error_sum += err * err
            ref_sum += norm * norm
        relative = math.sqrt(error_sum) / max(math.sqrt(ref_sum), 1e-12)
        return {"tensors": len(reference), "relative_l2": relative, "maximum_per_tensor_relative_l2": maximum,
                "passed": relative <= RELATIVE_L2_LIMIT and maximum <= MAX_TENSOR_RELATIVE_L2_LIMIT}

    def check_pair(self, base, prefix, initial):
        if (base["inputs"] != prefix["inputs"] or
                base["optimizer_scalars"] != prefix["optimizer_scalars"]):
            raise ValueError("Same-input hashes or scalar Muon state differ")
        groups = {key: self.compare_group(base[key], prefix[key]) for key in
                  ("forward", "unclipped_gradients", "clipped_gradients", "optimizer_tensors")}
        groups["parameter_updates"] = self.compare_group(base["parameters"], prefix["parameters"], initial)
        accepted = all(row["passed"] for row in groups.values())
        save(self.receipts / "first-update.json", {
            "actual_global_step": 1, "baseline_reverted": True, "identical_inputs": base["inputs"],
            "optimizer_scalar_entries_equal": len(base["optimizer_scalars"]), "groups": groups,
            "tolerance": {"relative_l2": RELATIVE_L2_LIMIT, "max_tensor_relative_l2": MAX_TENSOR_RELATIVE_L2_LIMIT},
            "passed": accepted, "formal_A100": False,
        })
        if not accepted:
            raise ValueError("Native same-input prefix first update did not pass")

    def run(self, step, train_one):
        if step != self.position + 1 or not 1 <= step <= 25:
            raise ValueError("Wrong native prefix global step")
        self.position = step
        baseline, initial = None, None
        if step == 1:
            initial = self.snapshot()
            self.mode, self.first = "baseline", {}
            try:
                train_one(step)
                baseline = self.first
            finally:
                self.restore(initial)
        self.mode, self.first = "prefix", {} if step == 1 else None
        self.phase_counts = {}
        try:
            with self.intercept():
                result = train_one(step)
            if self.phase_counts != {"forward": EXPECTED_CALLS, "backward_recompute": EXPECTED_CALLS} or self.pending:
                raise ValueError("Native prefix has incomplete attention mask coverage")
            if self.first is not None:
                self.check_pair(baseline, self.first, initial["parameters"])
            save(self.receipts / f"step-{step:02d}.json", {
                "global_step": step, "actual_mask_checks": self.phase_counts, "passed": True,
            })
        except BaseException:
            if initial is not None:
                self.restore(initial)
            raise
        finally:
            self.mode, self.first = None, None
        return result


def prefix_session_from_environment(torch, output, model, optimizer, *, config, cache, warmup, measured,
                                    restored):
    flag = os.environ.get(ENV)
    if flag is None:
        return None
    if (flag != "5326:4785" or (warmup, measured, restored) != (5, 20, 0)
            or float(config.get("lora_dropout", 0)) != 0
            or not config.get("gradient_checkpointing", True) or config.get("validate_recovery", False)
            or not torch.version.hip or torch.cuda.device_count() != 1):
        raise ValueError("Native prefix requires the frozen fresh single-card training scope")
    for name, length, count in (("text_mask", 300, 15), ("byt5_text_mask", 256, 0)):
        tensor = cache[name]
        expected = (torch.arange(length) < count).view(1, -1).to(dtype=tensor.dtype)
        if tensor.shape != expected.shape or tensor.device.type != "cpu" or not torch.equal(tensor, expected):
            raise ValueError("Native prefix cache mask differs")
    import numpy

    return NativePrefixSession(torch, numpy, output, model, optimizer)
