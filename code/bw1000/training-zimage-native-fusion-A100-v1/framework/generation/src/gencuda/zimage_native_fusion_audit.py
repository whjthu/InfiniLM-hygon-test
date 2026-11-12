"""Opt-in observer of one unchanged native preset-fusion call.

The tensor reference uses the same PyTorch backend and BF16 expression as the
original loader. It checks mapping, arithmetic and untouched-parameter identity;
it is not an independent numerical backend or a cross-platform training result.
Only hashes and metadata leave this observer. No tensor values are persisted.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import time


@contextmanager
def observe_one_fusion(loader_type, before, after):
    """Observe exactly one call and restore the exact original method on exit."""
    if not callable(before) or not callable(after):
        raise ValueError('fusion observers must be callable')
    original = loader_type.fuse_lora_to_base_model
    if getattr(original, '_zimage_identity_observer', False):
        raise RuntimeError('a fusion observer is already installed')
    calls = 0

    @wraps(original)
    def observed(loader, model, state_dict, alpha=1.0):
        nonlocal calls
        if calls:
            raise RuntimeError('a second native preset fusion is forbidden')
        calls += 1
        token = before(loader, model, state_dict, alpha)
        started = time.perf_counter()
        result = original(loader, model, state_dict, alpha=alpha)
        native_wall_seconds = time.perf_counter() - started
        after(token, loader, model, state_dict, alpha, native_wall_seconds)
        return result

    observed._zimage_identity_observer = True
    loader_type.fuse_lora_to_base_model = observed
    try:
        yield
        if calls != 1:
            raise RuntimeError('no native preset fusion was observed')
    finally:
        loader_type.fuse_lora_to_base_model = original


class PresetFusionAudit:
    """Hash full loaded base values before/after the native BF16 fusion."""

    def __init__(self, loader_type, expected_shapes, *, deadline, sink, device='cuda'):
        if not expected_shapes or not callable(deadline) or not callable(sink):
            raise ValueError('fusion audit requires target shapes,deadline and sink')
        if device != 'cuda':
            raise ValueError('this audit scope requires its declared CUDA device')
        self.loader_type = loader_type
        self.expected_shapes = dict(expected_shapes)
        self.deadline = deadline
        self.sink = sink
        self.result = None

    def _hash(self, tensor):
        import torch
        self.deadline()
        if tensor.is_meta or tensor.device.type != 'cuda' or tensor.dtype != torch.bfloat16:
            raise RuntimeError('fusion tensor is not a real CUDA BF16 tensor')
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError('nonfinite native fusion tensor')
        flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
        digest = hashlib.sha256()
        for offset in range(0, flat.numel(), 4*1024**2):
            self.deadline()
            digest.update(flat[offset:offset+4*1024**2].cpu().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _rng():
        import torch
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())

    def _before(self, loader, model, state_dict, alpha):
        import torch
        self.deadline()
        if self.result is not None or alpha != 1 or loader.torch_dtype != torch.bfloat16:
            raise RuntimeError('preset fusion alpha,dtype or lifecycle differs')
        if torch.device(loader.device).type != 'cuda':
            raise RuntimeError('preset fusion loader is not on CUDA')
        state = model.state_dict()
        if len(state) != 521 or sum(t.numel() for t in state.values()) != 6154908736:
            raise RuntimeError('full native DiT base extent differs')
        expected_keys = {n+'.lora_'+side+'.weight' for n in self.expected_shapes for side in ('A','B')}
        if set(state_dict) != expected_keys or len(expected_keys) != 480:
            raise RuntimeError('native preset factors are missing or unexpected')
        rng = self._rng()
        factors = {n:self._hash(t) for n,t in sorted(state_dict.items())}
        parameters = {}
        started = time.perf_counter()
        with torch.no_grad():
            for name, value in sorted(state.items()):
                self.deadline()
                before = self._hash(value)
                target = name[:-7] if name.endswith('.weight') else None
                if target in self.expected_shapes:
                    if list(value.shape) != self.expected_shapes[target]:
                        raise RuntimeError('loaded preset target shape differs')
                    a = state_dict[target+'.lora_A.weight']
                    b = state_dict[target+'.lora_B.weight']
                    if list(a.shape) != [32,value.shape[1]] or list(b.shape) != [value.shape[0],32]:
                        raise RuntimeError('loaded preset rank or shape differs')
                    expected = value.detach() + alpha * (b @ a)
                    expected_hash = self._hash(expected)
                    del expected
                else:
                    expected_hash = before
                parameters[name] = dict(shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),
                                        preset_target=target in self.expected_shapes,before_SHA256=before,expected_SHA256=expected_hash)
                self.sink(dict(stage='before_reference_parameter',name=name,completed=len(parameters),planned=521))
        if sum(p['preset_target'] for p in parameters.values()) != 240:
            raise RuntimeError('loaded base target coverage differs')
        return dict(parameters=parameters,factors=factors,rng=rng,before_audit_seconds=time.perf_counter()-started)

    def _after(self, token, loader, model, state_dict, alpha, native_wall_seconds):
        import torch
        started = time.perf_counter()
        torch.cuda.synchronize()
        state = model.state_dict()
        if set(state) != set(token['parameters']):
            raise RuntimeError('native fusion changed the base tensor set')
        changed = 0
        for name,value in sorted(state.items()):
            pin = token['parameters'][name]
            actual = self._hash(value)
            pin['after_SHA256'] = actual
            if actual != pin['expected_SHA256']:
                raise RuntimeError('actual native fusion differs at '+name)
            if actual != pin['before_SHA256']:
                if not pin['preset_target']:
                    raise RuntimeError('native fusion changed an untargeted tensor')
                changed += 1
            self.sink(dict(stage='after_native_parameter',name=name,completed=len([p for p in token['parameters'].values() if 'after_SHA256' in p]),planned=521))
        if set(state_dict) != set(token['factors']) or any(self._hash(t) != token['factors'][n] for n,t in state_dict.items()):
            raise RuntimeError('native fusion mutated the preset factors')
        rng = self._rng()
        if not all(torch.equal(a,b) for a,b in zip(token['rng'],rng)) or changed == 0:
            raise RuntimeError('fusion consumed RNG or no loaded base tensor changed')
        self.result = dict(passed=True,parameters=token['parameters'],preset_factor_SHA256=token['factors'],
                           base_tensors=521,preset_targets=240,preset_factors=480,changed_target_tensors=changed,
                           unmodified_other_tensors=281,preset_input_factors_unchanged=True,RNG_unchanged=True,
                           before_audit_seconds=token['before_audit_seconds'],after_audit_seconds=time.perf_counter()-started,
                           native_host_call_seconds=native_wall_seconds,independent_backend=False,
                           reference='Same CUDA backend BF16 base+(alpha*(B@A)),alpha1;all actual base values checked by fullbyteSHA.',
                           qualification='Loaded native preset fusion identity/arithmetic audit only;not independent kernel or cross-platform training numerics.')

    def observe(self):
        return observe_one_fusion(self.loader_type, self._before, self._after)
