"""Apply a preference only inside actual frozen BF16 text base linear calls."""
import functools
import hashlib
import inspect
from pathlib import Path
import torch
from peft.tuners.lora.layer import Linear as PeftLinear


GROUPS = (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
          ('mlp', ('gate_proj', 'up_proj', 'down_proj')))
ARMS = ('cublas', 'cublaslt', 'cublaslt', 'cublas')
SELECTED = {1: ('decoder_06.v_proj',),
            5: ('decoder_10.q_proj', 'decoder_06.mlp_down_proj')}


def require(ok, code):
    if not ok:
        raise ValueError(code)


class TextBasePreference:
    def __init__(self, core, *, call_limit=252):
        self.arm = None
        self.calls = []
        self.call_limit = call_limit
        self.originals = []
        self.selected = {}
        self.preference = torch.backends.cuda.preferred_blas_library
        self.expected = torch.backends.cuda._BlasBackends
        self.inherited = self.preference()
        text = core.get_submodule('model.language_model')
        require(len(text.layers) == 36, 'ORIGINAL_TEXT_LAYER_COUNT')
        sites = {}
        selected_names = {name for values in SELECTED.values() for name in values}
        base_ids = set()
        provider = Path(inspect.getsourcefile(PeftLinear))
        self.lora_provider = hashlib.sha256(provider.read_bytes()).hexdigest()
        self.inventory = {}
        for index, layer in enumerate(text.layers):
            for group, names in GROUPS:
                source = layer.get_submodule(group)
                for name in names:
                    projection = source.get_submodule(name)
                    site = f'decoder_{index:02d}.' + (f'mlp_{name}' if group == 'mlp' else name)
                    require(type(projection) is PeftLinear and list(projection.active_adapters) == ['default']
                            and not projection.disable_adapters and not projection.merged
                            and not projection.use_dora.get('default', False),
                            'EXACT_ACTIVE_NATIVE_PEFT_PROJECTION')
                    base = projection.get_base_layer()
                    a = projection.lora_A['default']
                    b = projection.lora_B['default']
                    require(type(base) is torch.nn.Linear and id(base) not in base_ids
                            and base.weight.dtype == torch.bfloat16 and not base.weight.requires_grad
                            and (base.bias is None or
                                 (base.bias.dtype == torch.bfloat16 and not base.bias.requires_grad)),
                            'EXACT_FROZEN_BF16_BASE_PROJECTION')
                    require(type(a) is torch.nn.Linear and type(b) is torch.nn.Linear
                            and a.weight.dtype == b.weight.dtype == torch.float32
                            and a.weight.requires_grad and b.weight.requires_grad
                            and a.in_features == base.in_features and b.out_features == base.out_features
                            and a.out_features == b.in_features == 16 and projection.scaling['default'] == 2.0,
                            'EXACT_ACTIVE_FP32_LORA_WEIGHTS')
                    base_ids.add(id(base))
                    sites[site] = base
                    self.inventory[site] = dict(weight_shape=list(base.weight.shape),
                                                 base_weight_dtype=str(base.weight.dtype),
                                                 lora_weight_dtype=str(a.weight.dtype),
                                                 has_base_bias=base.bias is not None)
                    if site in selected_names:
                        self.selected[site] = (base, a, b)
        require(len(sites) == 252 and len(self.selected) == 3, 'ORIGINAL_TEXT_PROJECTION_MEMBERSHIP')
        self.names = set(sites)
        for name, base in sites.items():
            original = base.forward
            self.originals.append((base, 'forward' in base.__dict__, base.__dict__.get('forward')))
            base.forward = self.wrap(name, base, original)

    def branches(self, rank):
        return {name:self.selected[name] for name in SELECTED.get(rank,())}

    def wrap(self, name, module, original):
        @functools.wraps(original)
        def call(input):
            value = input
            require(self.arm in ARMS and len(self.calls) < self.call_limit
                    and value.ndim in (2,3) and value.shape[-1] == module.in_features
                    and value.device.type == 'cuda'
                    and (value.dtype == torch.bfloat16 or
                         (value.dtype == torch.float32 and torch.is_autocast_enabled('cuda')
                          and torch.get_autocast_dtype('cuda') == torch.bfloat16)),
                    'ORIGINAL_TEXT_BASE_INPUT')
            prior = self.preference()
            require(prior == self.inherited, 'TEXT_BASE_PREF_ENTRANCE')
            try:
                actual = self.preference(self.arm)
                require(actual == self.expected[self.arm] == self.preference(), 'TEXT_BASE_PREFERENCE_READBACK')
                out = original(value)
            finally:
                self.preference(prior)
                require(self.preference() == prior, 'TEXT_BASE_PREFERENCE_RESTORE')
            require(out.dtype == torch.bfloat16 and list(out.shape) == [*value.shape[:-1], module.out_features],
                    'ORIGINAL_TEXT_BASE_OUTPUT')
            self.calls.append(dict(site=name, preference=str(actual), input_shape=list(value.shape),
                                   input_stride=list(value.stride()), input_dtype=str(value.dtype)))
            return out
        return call

    def finish(self):
        require(len(self.calls) == 252 and {row['site'] for row in self.calls} == self.names
                and self.preference() == self.inherited, 'ACTUAL_252_TEXT_BASE_CALLS')
        calls = self.calls
        self.calls = []
        return calls

    def close(self):
        for module, had, prior in self.originals:
            if had:
                module.forward = prior
            else:
                del module.forward
        self.originals.clear()
