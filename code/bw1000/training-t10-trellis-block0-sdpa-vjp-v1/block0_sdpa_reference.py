"""Replay exact TRELLIS block-0 SDPA operands with an independent CPU VJP."""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F


@contextmanager
def capture_block(attention):
    original = F.scaled_dot_product_attention
    record = {}
    active = False
    calls = 0

    def enter(*_):
        nonlocal active
        assert not active
        active = True

    def leave(*_):
        nonlocal active
        active = False

    def stash_grad(name):
        def capture(value):
            assert name not in record
            record[name] = value.detach()
        return capture

    def dispatch(q, k, v, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(q, k, v, *args, **kwargs)
        if active:
            assert calls == 1 and not record and not args and set(kwargs) == {"attn_mask"}
            mask = kwargs["attn_mask"]
            assert q.shape == k.shape == v.shape == result.shape
            assert q.ndim == 4 and q.shape[0] == 1 and q.shape[-2] == 1024
            assert all(x.dtype == torch.bfloat16 and x.is_cuda for x in (q, k, v, result, mask))
            assert mask.shape == (1, 1, 1, 1024) and (mask == 0).all().item()
            record.update(q=q.detach(), k=k.detach(), v=v.detach(), mask=mask.detach(),
                          out=result.detach(), native_backward=str(type(result.grad_fn).__name__),
                          operand_strides={n: list(x.stride()) for n,x in (("q",q),("k",k),("v",v),("mask",mask))})
            for name, value in (("dq",q),("dk",k),("dv",v),("upstream",result)):
                value.register_hook(stash_grad(name))
        return result

    before = attention.register_forward_pre_hook(enter)
    after = attention.register_forward_hook(leave)
    F.scaled_dot_product_attention = dispatch
    try:
        yield record
    finally:
        F.scaled_dot_product_attention = original
        after.remove()
        before.remove()
    assert calls == 60
    assert set(record) == {"q","k","v","mask","out","native_backward","operand_strides","dq","dk","dv","upstream"}


def attention_vjp_cpu(q, k, v, mask, upstream):
    """Compute the full unmasked attention in float64 one head at a time."""
    assert all(x.device.type == "cpu" for x in (q,k,v,mask,upstream))
    assert q.ndim == 4 and q.shape == k.shape == v.shape == upstream.shape
    batch, heads, length, width = q.shape
    assert batch == 1 and mask.shape == (1, 1, 1, length) and torch.count_nonzero(mask) == 0
    result = {name: torch.empty(q.shape, dtype=torch.float64) for name in ("out","dq","dk","dv")}
    scale = width ** -0.5
    for head in range(heads):
        a, b, c, g = (value[0, head].double() for value in (q,k,v,upstream))
        probability = torch.softmax((a @ b.T) * scale, dim=-1)
        dprobability = g @ c.T
        dscore = probability * (dprobability - (probability * dprobability).sum(dim=-1, keepdim=True))
        for name, value in (("out", probability @ c), ("dq", (dscore @ b) * scale),
                            ("dk", (dscore.T @ a) * scale), ("dv", probability.T @ g)):
            result[name][0, head] = value
    return result


def qualify_cpu_oracle():
    rng = torch.Generator().manual_seed(606)
    q, k, v, upstream = (torch.randn((1,2,3,5),generator=rng,dtype=torch.float64) for _ in range(4))
    leaves = [x.requires_grad_() for x in (q,k,v)]
    mask = torch.zeros(1,1,1,3,dtype=torch.float64)
    output = F.scaled_dot_product_attention(*leaves, attn_mask=mask)
    gradients = torch.autograd.grad(output, leaves, grad_outputs=upstream)
    expected = attention_vjp_cpu(q,k,v,mask,upstream)
    for name, actual in zip(("out","dq","dk","dv"),(output,*gradients), strict=True):
        torch.testing.assert_close(expected[name],actual,rtol=1e-12,atol=1e-12)


def error(actual, reference):
    a, b = actual.double(), reference.double()
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    norm = torch.linalg.vector_norm(b).item()
    assert norm > 0
    return {"relative_l2": torch.linalg.vector_norm(a-b).item()/norm,
            "max_abs": (a-b).abs().max().item(),
            "bf16_rounding_equal_fraction": (a==b.to(torch.bfloat16).double()).double().mean().item()}


def replay(record, output: Path, sha):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    names = ("q","k","v","mask","upstream")
    host = {name:record[name].cpu() for name in (*names,"out","dq","dk","dv")}
    for value in host.values():
        assert torch.isfinite(value).all()
    outputs = ("out","dq","dk","dv")
    leaves = [record[name].detach().requires_grad_() for name in ("q","k","v")]
    with sdpa_kernel([SDPBackend.MATH]):
        math = F.scaled_dot_product_attention(*leaves, attn_mask=record["mask"])
        math_grad = torch.autograd.grad(math, leaves, grad_outputs=record["upstream"])
    alt = {name:value.detach().cpu() for name,value in zip(outputs,(math,*math_grad),strict=True)}

    expected = attention_vjp_cpu(*(host[name] for name in names))
    measurements = {name:{
        "native_vs_cpu_f64":error(host[name],expected[name]),
        "dcu_math_vs_cpu_f64":error(alt[name],expected[name]),
        "native_vs_dcu_math":error(host[name],alt[name]),
    } for name in outputs}

    output.mkdir()
    artifact = output/"block0-operands.pt"
    torch.save({"native":host,"dcu_math":alt},artifact)
    return {"site":"blocks.0.self_attn",
            "inputs":{name:{"shape":list(host[name].shape),"dtype":str(host[name].dtype)}
                      for name in names},
            "operand_strides":record["operand_strides"],"native_backward":record["native_backward"],
            "math_backward":type(math.grad_fn).__name__, "operator_measures":measurements,
            "private_operator_artifact_SHA256":sha(artifact)}
