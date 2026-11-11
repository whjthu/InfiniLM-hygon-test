"""CPU attribution of paired LoRA weight VJPs to their two saved operands."""
from __future__ import annotations

import torch


def _metrics(actual, reference):
    actual, reference = actual.double(), reference.double()
    difference = actual - reference
    square = float(reference.square().sum())
    error = float(difference.square().sum())
    return dict(relative_l2=(error / square) ** .5 if square else None,
                squared_error=error, reference_squared_norm=square,
                max_abs=float(difference.abs().max()),
                exact_fraction=float((actual == reference).double().mean()))


def _operands(row):
    value, upstream, gradient = (row[name] for name in ('input', 'upstream', 'gradient'))
    for tensor in (value, upstream, gradient):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            raise ValueError('paired operand attribution requires CPU tensors')
        if not tensor.numel() or not torch.isfinite(tensor).all():
            raise ValueError('paired operand tensor is empty or nonfinite')
    if value.dtype != torch.bfloat16 or upstream.dtype != torch.bfloat16 or gradient.dtype != torch.float32:
        raise ValueError('expected original BF16 operands and FP32 parameter gradients')
    if value.ndim not in (2, 3) or upstream.ndim not in (2, 3) or gradient.ndim != 2:
        raise ValueError('unexpected saved operand dimensions')
    value = value.reshape(-1, value.shape[-1]).double()
    upstream = upstream.reshape(-1, upstream.shape[-1]).double()
    if (value.shape[-1] != 2 or upstream.shape[0] != value.shape[0]
            or tuple(gradient.shape) != (upstream.shape[-1], 2)):
        raise ValueError('saved rank-two LoRA operand shapes differ')
    return value, upstream, gradient.double()


def attribute_site(reference, candidate):
    """Compare all four operand combinations without executing a device operator."""
    xa, ya, ga = _operands(reference)
    xh, yh, gh = _operands(candidate)
    if xa.shape != xh.shape or ya.shape != yh.shape:
        raise ValueError('paired original operands have different shapes')
    aa, hh = ya.T @ xa, yh.T @ xh
    replace_input, replace_upstream = yh.T @ xa, ya.T @ xh
    dx, dy = xh - xa, yh - ya
    input_term, upstream_term, interaction = ya.T @ dx, dy.T @ xa, dy.T @ dx
    delta = hh - aa
    residual = delta - input_term - upstream_term - interaction
    tolerance = 128 * torch.finfo(torch.float64).eps * (1 + float(aa.abs().max()) + float(hh.abs().max()))
    if float(residual.abs().max()) > tolerance:
        raise ValueError('CPU operand-difference decomposition did not close')
    terms = [input_term, upstream_term, interaction]
    gram = [[float((left * right).sum()) for right in terms] for left in terms]
    return dict(input=_metrics(xh, xa), upstream=_metrics(yh, ya),
                native_gradient=_metrics(gh, ga),
                oracle_hh_vs_aa=_metrics(hh, aa),
                a100_input_hygon_upstream_vs_aa=_metrics(replace_input, aa),
                hygon_input_a100_upstream_vs_aa=_metrics(replace_upstream, aa),
                a100_native_vs_cpu_oracle=_metrics(ga, aa),
                hygon_native_vs_cpu_oracle=_metrics(gh, hh),
                a100_native_vs_rounded_cpu_oracle=_metrics(ga, aa.to(torch.bfloat16).float()),
                hygon_native_vs_rounded_cpu_oracle=_metrics(gh, hh.to(torch.bfloat16).float()),
                decomposition=dict(order=['input', 'upstream', 'interaction'],
                                   gram=gram, summed_squared_error=sum(sum(row) for row in gram),
                                   actual_squared_error=float(delta.square().sum()),
                                   residual_max_abs=float(residual.abs().max())),
                qualification='CPU FP64 algebra on the two actual saved operand pairs; not a model replay or numerical acceptance threshold')


def attribute_sites(reference, candidate):
    if not reference or set(reference) != set(candidate):
        raise ValueError('paired saved projection sites differ')
    sites = {name: attribute_site(reference[name], candidate[name]) for name in sorted(reference)}
    keys = ('native_gradient', 'oracle_hh_vs_aa', 'a100_input_hygon_upstream_vs_aa',
            'hygon_input_a100_upstream_vs_aa')
    aggregate = {}
    for key in keys:
        error = sum(row[key]['squared_error'] for row in sites.values())
        norm = sum(row[key]['reference_squared_norm'] for row in sites.values())
        aggregate[key] = dict(squared_error=error, reference_squared_norm=norm,
                              relative_l2=(error / norm) ** .5 if norm else None)
    return dict(sites=sites, four_site_aggregate=aggregate, GPU_calls=0,
                qualification='Four saved projections only. Full adapter, optimizer update, model, recovery, performance and task effects remain separate.')
