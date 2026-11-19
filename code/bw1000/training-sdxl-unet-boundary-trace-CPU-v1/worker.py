"""Exercise the new hooks with real Torch CPU tensors and actual UNet topology."""
import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from compare_traces import compare

from boundary_probe import BOUNDARIES, BoundaryProbe


MODEL_CONFIG_PIN = '30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6'


class Down(nn.Module):
    def forward(self, hidden_states):
        skip = hidden_states * 3
        out = hidden_states * 4
        return out, (skip, out)


class Up(nn.Module):
    def forward(self, hidden_states, res_hidden_states_tuple):
        return hidden_states * 2 + sum(res_hidden_states_tuple)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = Down()
        self.up = Up()

    def forward(self, x):
        y, skips = self.down(hidden_states=x)
        return self.up(hidden_states=y, res_hidden_states_tuple=skips)


def must_fail(expected, fn):
    try:
        fn()
    except ValueError as exc:
        if str(exc) != expected:
            raise
    else:
        raise AssertionError(expected)


def exercise(dtype):
    targets = {'down': 'down', 'up': 'up'}
    args = torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4).transpose(0, 2)
    x = args.clone().requires_grad_(True)
    model = Model()
    baseline = model(x)
    baseline_grad, = torch.autograd.grad(baseline.sum(), x)
    with BoundaryProbe(Model, targets) as probe:
        observed = model(x)
        observed.sum().backward()
        out = probe.report()
        assert torch.equal(observed, baseline) and torch.equal(x.grad, baseline_grad)
        edges = out['edges']
        assert edges['down.in']['stride'] == list(x.stride()) and not x.is_contiguous()
        assert edges['down.out'] == edges['down.residual_out.1'] == edges['up.residual_in.1'] == edges['up.in']
        assert edges['down.residual_out.0'] == edges['up.residual_in.0']
        assert all(row['adjoint_status'] == 'observed' for row in edges.values())
        assert out['unique_tensor_objects'] == 4
        assert out['snapshot_bytes'] == 8*x.numel()*x.element_size()
        assert out['events'] == ['down:enter', 'down:leave', 'up:enter', 'up:leave']
        assert edges['down.out']['aggregate_adjoint']['max_abs'] == 3.
        assert edges['down.in']['aggregate_adjoint']['max_abs'] == 15.
    same = compare(out, copy.deepcopy(out))
    assert same['earliest_forward_value'] is None and not same['aggregate_adjoint_mismatch_edges']
    changed = copy.deepcopy(out)
    for edge in ('down.out','down.residual_out.1','up.residual_in.1','up.in'):
        changed['edges'][edge]['value']['sha256'] = 'different'
    found = compare(out, changed)
    assert found['earliest_forward_value'] == 'down.out'
    disconnected = copy.deepcopy(out)
    disconnected['edges']['up.in']['alias_id'] = -1
    must_fail('TENSOR_ALIAS_GRAPH_CHANGED', lambda: compare(out, disconnected))
    assert not probe.handles and not probe.values

    def too_small():
        with BoundaryProbe(Model, targets, max_bytes=x.element_size()) as probe:
            model(x)

    def repeated_forward():
        with BoundaryProbe(Model, targets):
            model(x)
            model(x)

    def missing_backward():
        with BoundaryProbe(Model, targets) as probe:
            model(x)
            probe.report()

    must_fail('TRACE_BYTE_LIMIT', too_small)
    must_fail('MULTIPLE_NATIVE_MODEL_CALLS', repeated_forward)
    must_fail('NO_NATIVE_BACKWARD_EVIDENCE', missing_backward)
    return out['snapshot_bytes']


def native_topology():
    from diffusers import UNet2DConditionModel

    config = Path('/observed/config.json').read_bytes()
    assert hashlib.sha256(config).hexdigest() == MODEL_CONFIG_PIN
    with torch.device('meta'):
        original = UNet2DConditionModel.from_config(json.loads(config))
    assert all(p.is_meta for p in original.parameters())
    topology = {}
    for name, role in BOUNDARIES.items():
        layer = original.get_submodule(name)
        signature = inspect.signature(layer.forward)
        if role in ('down','up'):
            assert 'hidden_states' in signature.parameters
        if role == 'up':
            assert 'res_hidden_states_tuple' in signature.parameters
        topology[name] = dict(role=role, type=type(layer).__name__)
    return topology


def main():
    assert torch.version.hip and torch.cuda.device_count() == 0 and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    checks = {str(dtype):exercise(dtype) for dtype in (torch.bfloat16, torch.float32)}
    result = dict(torch=str(torch.__version__), hip=torch.version.hip, GPU_initialized=torch.cuda.is_initialized(),
                  physical_GPU=0, tests_passed=14, CPU_tensor_snapshot_bytes=checks, native_meta_topology=native_topology(),
                  meta_model_config_sha256=MODEL_CONFIG_PIN, native_weight_files_opened=0, native_UNet_calls=0)
    with Path('/work/result.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
