"""Capture the effective operands of the largest TRELLIS LoRA B gradients."""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F


SITES = (
    "blocks.0.self_attn.to_qkv",
    "blocks.0.self_attn.to_out",
    "blocks.0.cross_attn.to_kv",
    "blocks.1.self_attn.to_qkv",
)


@contextmanager
def capture_lora(model):
    original = F.linear
    modules = dict(model.named_modules())
    active = None
    captured = {}
    handles = []

    def begin(site):
        def enter(*_):
            nonlocal active
            assert active is None
            active = site
        return enter

    def end(*_):
        nonlocal active
        assert active in captured
        active = None

    def dispatch(value, weight, bias=None):
        if active is None or value.shape[-1] != 2 or weight.shape != modules[active].lora_B.shape:
            return original(value, weight, bias)
        assert active not in captured and bias is None
        assert value.ndim in (2, 3) and value.dtype == torch.bfloat16 and value.is_cuda
        effective = []

        def pack(tensor):
            detached = tensor.detach()
            effective.append(detached)
            return detached

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = original(value, weight, bias)
        saved_input, = [x for x in effective if x.numel() == value.numel() and x.shape[-1] == 2]
        assert saved_input.dtype == torch.bfloat16 and torch.equal(saved_input.reshape_as(value), value)
        assert out.shape == (*value.shape[:-1], weight.shape[0]) and out.dtype == torch.bfloat16
        site_record = {"input":saved_input,"weight":weight.detach(),
                       "effective_weight_shape":[list(x.shape) for x in effective if x is not saved_input],
                       "input_shape":list(value.shape),"input_stride":list(value.stride()),"output_stride":list(out.stride())}

        def observe(gradient):
            assert "upstream" not in site_record and gradient.dtype == torch.bfloat16
            site_record["upstream"] = gradient.detach()

        out.register_hook(observe)
        captured[active] = site_record
        return out

    for site in SITES:
        handles.extend((modules[site].register_forward_pre_hook(begin(site)),
                        modules[site].register_forward_hook(end)))
    F.linear = dispatch
    try:
        yield captured
    finally:
        F.linear = original
        for handle in reversed(handles):
            handle.remove()
    assert set(captured) == set(SITES) and all("upstream" in row for row in captured.values())


def weight_vjp_cpu(value, upstream):
    assert value.ndim == upstream.ndim == 2 and value.shape[0] == upstream.shape[0]
    assert value.device.type == upstream.device.type == "cpu"
    return upstream.double().T @ value.double()


def qualify_cpu_vjp():
    rng = torch.Generator().manual_seed(606)
    value = torch.randn((4,2),generator=rng,dtype=torch.float64)
    weight = torch.randn((3,2),generator=rng,dtype=torch.float64,requires_grad=True)
    upstream = torch.randn((4,3),generator=rng,dtype=torch.float64)
    grad, = torch.autograd.grad(F.linear(value,weight),weight,upstream)
    torch.testing.assert_close(weight_vjp_cpu(value,upstream),grad,rtol=1e-14,atol=1e-14)


def weight_vjp_device(record,dtype):
    value = record["input"].reshape(record["input_shape"]).to(dtype).detach().requires_grad_()
    weight = record["weight"].to(dtype).detach().requires_grad_()
    upstream = record["upstream"].to(dtype)
    output = F.linear(value,weight)
    grad, = torch.autograd.grad(output,weight,upstream)
    return grad.detach().cpu()


def squared_error(actual, expected):
    a, b = actual.double(), expected.double()
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a-b).square().sum())


def measure(actual, expected):
    norm_sq = squared_error(expected, torch.zeros_like(expected))
    assert norm_sq > 0
    return {"relative_l2": (squared_error(actual,expected)/norm_sq)**0.5,
            "exact_fraction": float((actual==expected).double().mean())}


def compare(captured, native, a100, output: Path, sha):
    replacement = dict(native)
    measured, payload = {}, {}
    for site in SITES:
        row = captured[site]
        name = site + ".lora_B"
        value, upstream = row["input"].cpu(), row["upstream"].cpu()
        fp64 = weight_vjp_cpu(value.reshape(-1,2),upstream.reshape(-1,upstream.shape[-1]))
        fp64_rounded = fp64.to(torch.bfloat16).float()
        bf16 = weight_vjp_device(row,torch.bfloat16).float()
        fp32 = weight_vjp_device(row,torch.float32)
        native_grad = native[name]
        replacement[name] = fp64_rounded
        measured[name] = {
            "input_shape":row["input_shape"],"input_stride":row["input_stride"],
            "effective_saved_input_shape":list(value.shape),
            "upstream_shape":list(upstream.shape),"upstream_stride":list(row["upstream"].stride()),
            "effective_saved_other_shapes":row["effective_weight_shape"],
            "native_vs_fixed_bf16":measure(native_grad,bf16),
            "native_vs_cpu_f64":measure(native_grad,fp64),
            "native_vs_cpu_f64_rounded_bf16":measure(native_grad,fp64_rounded),
            "hip_fp32_vs_cpu_f64":measure(fp32,fp64),
            "native_vs_a100_weight_gradient":measure(native_grad,a100[name]),
            "cpu_f64_rounded_vs_a100_weight_gradient":measure(fp64_rounded,a100[name]),
            "native_vs_a100_squared_error":squared_error(native_grad,a100[name]),
        }
        payload[name] = {"input":value,"upstream":upstream,"native_grad":native_grad,
                         "hip_bf16_grad":bf16,"hip_fp32_grad":fp32,"cpu_fp64_grad":fp64}
    output.mkdir()
    artifact = output/"lora-projection-operands.pt"
    torch.save(payload,artifact)
    baseline_sq = sum(squared_error(native[n],g) for n,g in a100.items())
    alternate_sq = sum(squared_error(replacement[n],g) for n,g in a100.items())
    reference_sq = sum(squared_error(g,torch.zeros_like(g)) for g in a100.values())
    return {"sites":measured,"captured_share_of_native_gradient_squared_error":
            sum(row["native_vs_a100_squared_error"] for row in measured.values())/baseline_sq,
            "full_adapter_native_relative_l2":(baseline_sq/reference_sq)**0.5,
            "full_adapter_cpu_rounded_substitution_relative_l2":(alternate_sq/reference_sq)**0.5,
            "private_operator_artifact_SHA256":sha(artifact)}, replacement
