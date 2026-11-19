"""Capture bounded SDXL UNet edge fingerprints with read-only tensor hooks."""
from __future__ import annotations

import hashlib

import torch


BOUNDARIES = {
    'conv_in': 'plain',
    'down_blocks.0': 'down',
    'down_blocks.1': 'down',
    'down_blocks.2': 'down',
    'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_q': 'plain',
    'mid_block': 'plain',
    'up_blocks.0': 'up',
    'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_q': 'plain',
    'up_blocks.1': 'up',
    'up_blocks.2': 'up',
    'conv_out': 'plain',
}


class BoundaryProbe:
    """Use one Tensor hook per object; shared aliases see the aggregate adjoint."""

    def __init__(self, model_type, targets=None, *, max_bytes=96 * 1024**2):
        self.model_type = model_type
        self.targets = dict(BOUNDARIES if targets is None else targets)
        if not self.targets or any(k not in {'plain', 'down', 'up'} for k in self.targets.values()):
            raise ValueError('INVALID_BOUNDARY_MAP')
        if not isinstance(max_bytes, int) or not 0 < max_bytes <= 96 * 1024**2:
            raise ValueError('INVALID_TRACE_BYTE_LIMIT')
        self.max_bytes = max_bytes
        self.bytes = 0
        self.handles = []
        self.edges = {}
        self.values = {}
        self.events = []
        self.adjoint_events = []
        self.attached = False
        self.visited = set()

    def __enter__(self):
        self.handles.append(torch.nn.modules.module.register_module_forward_pre_hook(self._attach))
        return self

    def __exit__(self, *_):
        while self.handles:
            self.handles.pop().remove()
        self.values.clear()

    def _clone(self, value):
        if value.layout != torch.strided or value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('TRACE_TENSOR_SCHEMA')
        size = value.numel() * value.element_size()
        if not size or self.bytes + size > self.max_bytes:
            raise ValueError('TRACE_BYTE_LIMIT')
        clone = value.detach().clone()
        self.bytes += size
        return clone

    def _edge(self, name, tensor):
        if name in self.edges or not isinstance(tensor, torch.Tensor):
            raise ValueError('REPEATED_OR_NONTENSOR_EDGE')
        key = id(tensor)
        if key not in self.values:
            cell = dict(original=tensor, snapshot=self._clone(tensor), adjoint=None,
                        stride=list(tensor.stride()), requires_grad=tensor.requires_grad,
                        identity=len(self.values))
            self.values[key] = cell
            if tensor.requires_grad:
                def grad_hook(adj):
                    if cell['adjoint'] is not None:
                        raise ValueError('REPEATED_TENSOR_ADJOINT')
                    if adj.shape != tensor.shape:
                        raise ValueError('TENSOR_ADJOINT_SHAPE')
                    cell['adjoint'] = self._clone(adj)
                    self.adjoint_events.append(cell['identity'])
                    return None
                self.handles.append(tensor.register_hook(grad_hook))
        self.edges[name] = key

    def _attach(self, module, _):
        if not isinstance(module, self.model_type):
            return None
        if self.attached:
            raise ValueError('MULTIPLE_NATIVE_MODEL_CALLS')
        self.attached = True
        for name, role in self.targets.items():
            part = module.get_submodule(name)

            def entering(_, args, kwargs, name=name, role=role):
                if name in self.visited:
                    raise ValueError('MULTIPLE_BOUNDARY_CALLS')
                self.visited.add(name)
                self.events.append(name + ':enter')
                if 'hidden_states' in kwargs:
                    if args:
                        raise ValueError('AMBIGUOUS_PRIMARY_ARGUMENT')
                    hidden = kwargs['hidden_states']
                elif args:
                    hidden = args[0]
                else:
                    raise ValueError('MISSING_PRIMARY_ARGUMENT')
                self._edge(name + '.in', hidden)
                if role == 'up':
                    skips = kwargs.get('res_hidden_states_tuple')
                    if not isinstance(skips, (tuple, list)) or not skips:
                        raise ValueError('EXPECTED_UP_BLOCK_RESIDUALS')
                    for i, value in enumerate(skips):
                        self._edge(name + '.residual_in.' + str(i), value)
                return None

            def leaving(_, args, kwargs, output, name=name, role=role):
                self.events.append(name + ':leave')
                if role == 'down':
                    if (not isinstance(output, tuple) or len(output) != 2
                            or not isinstance(output[1], tuple) or not output[1]):
                        raise ValueError('EXPECTED_DOWN_BLOCK_RESIDUALS')
                    self._edge(name + '.out', output[0])
                    for i, value in enumerate(output[1]):
                        self._edge(name + '.residual_out.' + str(i), value)
                else:
                    self._edge(name + '.out', output)
                return None

            self.handles.append(part.register_forward_pre_hook(entering, with_kwargs=True))
            self.handles.append(part.register_forward_hook(leaving, with_kwargs=True))
        return None

    @staticmethod
    def _stats(tensor):
        cpu = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError('NONFINITE_SDXL_TRACE_TENSOR')
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        nums = cpu.double()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), l2=float(nums.norm()),
                    max_abs=float(nums.abs().max()), zero_elements=int((nums == 0).count_nonzero()))

    def report(self):
        if not self.attached or self.visited != set(self.targets) or len(self.events) != 2*len(self.targets):
            raise ValueError('INCOMPLETE_NATIVE_BOUNDARIES')
        cached = {}
        for key, value in self.values.items():
            adj = value['adjoint']
            status = ('observed' if adj is not None else
                      'not_observed' if value['requires_grad'] else 'not_required')
            cached[key] = dict(alias_id=value['identity'], stride=value['stride'],
                               value=self._stats(value['snapshot']), adjoint_status=status,
                               aggregate_adjoint=self._stats(adj) if adj is not None else None)
        if not any(e['adjoint_status'] == 'observed' for e in cached.values()):
            raise ValueError('NO_NATIVE_BACKWARD_EVIDENCE')
        return dict(events=list(self.events), adjoint_events=list(self.adjoint_events), target_count=len(self.targets),
                    snapshot_bytes=self.bytes, unique_tensor_objects=len(cached),
                    scope='Aggregated tensor adjoints, including any shared skip consumers.',
                    edges={name: cached[key] for name, key in self.edges.items()})
