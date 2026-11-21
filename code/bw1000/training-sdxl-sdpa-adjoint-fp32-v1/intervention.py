"""Use the native SDPA forward and replace only the attention adjoint in the SDXL replay."""

from collections import Counter, defaultdict
from functools import partial
from hashlib import sha256
import math
from pathlib import Path
import re
import runpy

from adjoint import vjp


ORIGINAL = Path(__file__).resolve().parent.parent / 'training-sdxl-frozen-conv-dgrad-fp32-v2/hygon_sdxl_first_step.py'
ORIGINAL_SHA256 = 'e8f0e5d333b87ee11a3cc01ae61c37cac7543d32892ddf3129c803b450767142'
HIP_RAW = '8edb12bd77ea3f7c193513551f904f168bf473239c37a2e23a90e87548a86df3'
NAME = re.compile(r'((down_blocks\.\d+|up_blocks\.\d+|mid_block)\..*\.attn[12])\.'
                  r'(to_[qkv]|to_out\.0)\.lora_([AB])\.default\.weight')
PROJECTIONS = {'to_q', 'to_k', 'to_v', 'to_out.0'}
REGIONS = {'down_blocks.1': 8, 'down_blocks.2': 40, 'mid_block': 20, 'up_blocks.0': 60, 'up_blocks.1': 12}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def trainable_attention_sites(names):
    sites = defaultdict(set)
    regions = {}
    for name in names:
        match = NAME.fullmatch(name)
        require(match is not None, 'unexpected trainable outside original SDXL attention')
        site, region, projection, operand = match.groups()
        pair = projection, operand
        require(pair not in sites[site], 'repeated original trainable attention operand')
        sites[site].add(pair)
        regions[site] = region
    expected = {(projection, operand) for projection in PROJECTIONS for operand in ('A', 'B')}
    require(all(pairs == expected for pairs in sites.values()) and Counter(regions.values()) == REGIONS,
            'original 1120-tensor SDXL attention inventory differs')
    return regions


def native_adjoint(torch):
    class AttentionAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, native, q, k, v, scale, policy, site):
            ctx.save_for_backward(q, k, v)
            ctx.scale = scale
            ctx.policy = policy
            ctx.site = site
            return native

        @staticmethod
        def backward(ctx, dy):
            q, k, v = ctx.saved_tensors
            require(dy.shape == q.shape and dy.dtype == q.dtype and dy.device == q.device,
                    'original SDPA output adjoint differs')
            previous = torch.get_float32_matmul_precision()
            try:
                torch.set_float32_matmul_precision('highest')
                with torch.autocast(q.device.type, enabled=False):
                    grads = vjp(q.float(), k.float(), v.float(), dy.float(), ctx.scale,
                                softmax=lambda x: x.softmax(dim=-1), row_sum=lambda x: x.sum(dim=-1, keepdim=True),
                                concatenate=lambda xs: torch.cat(xs, dim=-2))
                    rounded = tuple(grad.to(tensor.dtype) for grad, tensor in zip(grads, (q, k, v), strict=True))
            finally:
                torch.set_float32_matmul_precision(previous)
            ctx.policy.completed[ctx.site] += 1
            return None, *rounded, None, None, None

    return AttentionAdjoint


class SDPAAdjointPolicy:
    def __init__(self, torch):
        self.torch = torch
        self.functional = torch.nn.functional
        self.original = self.functional.scaled_dot_product_attention
        self.adjoint = native_adjoint(torch)
        self.patches = []
        self.entered = Counter()
        self.completed = Counter()
        self.geometries = Counter()
        self.active = None
        self.result = {}

    def attach(self, model, named):
        from diffusers.models.attention_processor import Attention

        self.sites = trainable_attention_sites(named)
        modules = dict(model.named_modules())
        for site in self.sites:
            module = modules[site]
            require(type(module) is Attention, 'unexpected original Diffusers attention class')
            original = module.forward
            self.patches.append((module, original, 'forward' in vars(module)))
            module.forward = partial(self.forward, site, original)
        self.functional.scaled_dot_product_attention = self.sdpa

    def forward(self, site, original, *args, **kwargs):
        require(self.active is None and self.entered[site] == 0, 'repeated SDXL attention site')
        self.active = site
        try:
            output = original(*args, **kwargs)
        finally:
            self.active = None
        require(self.entered[site] == 1, 'selected site did not take the original SDPA path')
        return output

    def sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        kwargs = {} if scale is None else {'scale': scale}
        if enable_gqa:
            kwargs['enable_gqa'] = True
        if self.active is None:
            return self.original(q, k, v, attn_mask, dropout_p, is_causal, **kwargs)
        require(not enable_gqa and attn_mask is None and dropout_p == 0.0 and is_causal is False
                and q.ndim == k.ndim == v.ndim == 4 and q.shape[:2] == k.shape[:2] == v.shape[:2]
                and k.shape == v.shape and q.shape[-1] == k.shape[-1] and q.shape[0] == 1
                and 0 < q.shape[1] <= 32 and 0 < q.shape[2] <= 256 and 0 < k.shape[2] <= 256
                and 0 < q.shape[-1] <= 128 and q.device == k.device == v.device
                and q.dtype == k.dtype == v.dtype == self.torch.bfloat16 and all(t.requires_grad for t in (q, k, v)),
                'unsupported original SDXL SDPA boundary')
        self.entered[self.active] += 1
        require(self.entered[self.active] == 1, 'multiple SDPA calls within one original site')
        effective = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
        require(math.isfinite(effective) and effective > 0, 'invalid original SDPA scale')
        native = self.original(q, k, v, attn_mask, dropout_p, is_causal, **kwargs)
        require(native.shape == q.shape and native.dtype == q.dtype and native.device == q.device,
                'native SDPA output geometry differs')
        self.geometries['x'.join(map(str, (q.shape[1], q.shape[2], k.shape[2], q.shape[3])))] += 1
        # The native call runs under its original grad mode before detaching its output.
        return self.adjoint.apply(native.detach(), q, k, v, effective, self, self.active)

    def verify_prediction(self, raw):
        data = raw.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes()
        digest = sha256(data).hexdigest()
        self.result['raw_prediction_SHA256'] = digest
        require(tuple(raw.shape) == (1, 4, 32, 32) and raw.dtype == self.torch.bfloat16 and digest == HIP_RAW,
                'native HIP forward output differs')

    def finish(self):
        require(self.entered == self.completed == Counter(dict.fromkeys(self.sites, 1)),
                'incomplete original SDXL attention adjoint sites')
        self.result.update(native_forward_exact=True, actual_adjoint_calls=sum(self.completed.values()),
                           sites_by_region=dict(Counter(self.sites.values())), true_operands_by_geometry=dict(self.geometries))

    def close(self):
        self.functional.scaled_dot_product_attention = self.original
        for module, original, overridden in reversed(self.patches):
            if overridden:
                module.forward = original
            else:
                del module.forward
        self.patches.clear()


def replay_once(reference, policy_file, weights, output):
    """Called inside the existing HIP first-step worker after its owner and runtime checks."""
    import torch

    require(sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA256, 'original first-step worker differs')
    worker = runpy.run_path(str(ORIGINAL))
    binding = worker['verify_reference'](reference, policy_file)
    policy = SDPAAdjointPolicy(torch)
    try:
        worker['replay'](reference, weights, output, binding, accelerator='hip', weight_gradient=policy)
    finally:
        policy.close()
    return dict(policy.result, formal_acceptance=False)
