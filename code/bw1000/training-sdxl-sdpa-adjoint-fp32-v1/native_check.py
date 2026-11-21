"""Exercise the actual PyTorch bridge in a zero-GRES target step."""

import json

from intervention import SDPAAdjointPolicy


def run(torch):
    assert torch.cuda.device_count() == 0 and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(733)
    policy = SDPAAdjointPolicy(torch)
    prior_precision = torch.get_float32_matmul_precision()
    cases = ((1, 2, 133, 77, 64), (1, 2, 64, 64, 64), (1, 3, 65, 65, 16))
    for batch, heads, queries, keys, width in cases:
        # Small dy leaves a useful overlap when comparing rounded adjoints.
        q = torch.randn(batch, heads, queries, width, dtype=torch.bfloat16).requires_grad_()
        k = torch.randn(batch, heads, keys, width, dtype=torch.bfloat16).requires_grad_()
        v = torch.randn_like(k).requires_grad_()
        dy = torch.randn_like(q) * 0.25
        scale = width**-0.5
        fp64 = [tensor.detach().double().requires_grad_() for tensor in (q, k, v)]
        expected = ((fp64[0] @ fp64[1].transpose(-1, -2))*scale).softmax(dim=-1) @ fp64[2]
        gradients = torch.autograd.grad(expected, fp64, dy.double())
        native = policy.original(q, k, v)
        policy.active = 'fixture'
        policy.entered.clear()
        patched = policy.sdpa(q, k, v)
        assert torch.equal(patched, native)
        observed = torch.autograd.grad(patched, (q, k, v), dy)
        for got, want in zip(observed, gradients, strict=True):
            assert got.dtype == torch.bfloat16
            torch.testing.assert_close(got.float(), want.float(), atol=0.002, rtol=0.006)
        assert torch.get_float32_matmul_precision() == prior_precision
    assert policy.completed['fixture'] == len(cases) and not torch.cuda.is_initialized()
    return dict(cpu_fixture_shapes=len(cases), native_dtype='bfloat16', gpu_initialized=False)


if __name__ == '__main__':
    import torch

    print(json.dumps(run(torch), allow_nan=False), flush=True)
