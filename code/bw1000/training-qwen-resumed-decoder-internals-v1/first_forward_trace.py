"""Capture the first restored native forward on only the two affected ranks."""
import hashlib
import json
from pathlib import Path

POLICY_RANKS = {2: 1, 6: 1}
SELECTED_DECODERS = {2: 26, 6: 5}
_PROBE = None
ORIGINAL_TRACE_SHA256 = "bde06bde48d82675b7d1e6b0424e7d5b5125f7dbfaac9601d11f162e5e54eac8"
MAX_TENSOR_BYTES = 768 * 1024**2


class FirstForward:
    def __init__(self, model, path, check, rank):
        from vlm.runtime.forward_entry_state import tensor_digest

        self.path = path
        self.check = check
        self.rank = rank
        self.digest = tensor_digest
        self.ordinal = POLICY_RANKS[rank]
        self.points = {}
        self.active = False
        self.finished_forward = False
        core = model.get_base_model()
        text = core.get_submodule("model.language_model")
        if path.exists() or len(text.layers) != 36:
            raise ValueError("Unexpected saved Qwen forward trace scope")
        self.handles = [text.layers[0].register_forward_pre_hook(self.first_input, with_kwargs=True)]
        modules = [(f"decoder_{i:02d}", layer) for i, layer in enumerate(text.layers)]
        modules += [("final_norm", text.get_submodule("norm")), ("lm_head", core.get_output_embeddings())]
        self.expected = {"decoder_00_input", *(name for name, _ in modules)}
        self.handles.extend(module.register_forward_hook(self.observe(name)) for name, module in modules)
        selected = text.layers[SELECTED_DECODERS[rank]]
        self.attention = selected.self_attn
        self.add_internals(selected)

    def add_internals(self, layer):
        self.route_sites = {f"sdpa.{name}" for name in ("q", "k", "v", "mask", "out")}
        self.expected.update(self.route_sites)
        self.expected.update({f"selected.{n}" for n in ("input", "position_ids", "mask", "cos", "sin")})
        self.handles.append(layer.register_forward_pre_hook(self.selected_entry, with_kwargs=True))
        for name in ("input_layernorm", "post_attention_layernorm", "mlp", "self_attn"):
            module = layer.get_submodule(name)
            self.watch(f"{name}.out", module)
        self.expected.add("post_attention_layernorm.in")
        self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            self.watch_input("post_attention_layernorm.in")))
        for name in ("q_norm", "k_norm"):
            self.watch(f"self_attn.{name}.out", self.attention.get_submodule(name))
        for branch, paths in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                              ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in paths:
                path = f"{branch}.{name}"
                proj = layer.get_submodule(path)
                self.watch(f"{path}.out", proj)
                drop = proj.get_submodule("lora_dropout.default")
                self.watch(f"{path}.dropout.out", drop)
                self.expected.add(f"{path}.dropout.rng")
                self.handles.append(drop.register_forward_pre_hook(self.watch_rng(f"{path}.dropout.rng")))
        self.expected.add("self_attn.o_proj.in")
        self.handles.append(self.attention.o_proj.register_forward_pre_hook(self.watch_input("self_attn.o_proj.in")))

    def watch(self, name, module):
        self.expected.add(name)
        self.handles.append(module.register_forward_hook(self.observe(name)))

    def watch_input(self, name):
        def before(_module, args):
            if self.active:
                self.record(name, args[0])
        return before

    def watch_rng(self, name):
        def before(_module, args):
            if self.active:
                import torch

                self.record(name, torch.cuda.get_rng_state(args[0].device))
        return before

    def selected_entry(self, _module, args, kwargs):
        if not self.active:
            return
        def operand(name, index):
            return kwargs[name] if name in kwargs else args[index] if index < len(args) else None

        self.record("selected.input", operand("hidden_states", 0))
        cos, sin = operand("position_embeddings", 1)
        self.record("selected.cos", cos)
        self.record("selected.sin", sin)
        self.record_optional("selected.mask", operand("attention_mask", 2))
        self.record_optional("selected.position_ids", operand("position_ids", 3))

    def record_optional(self, name, value):
        if value is None:
            if name in self.points:
                raise ValueError("Repeated optional Qwen boundary")
            self.points[name] = None
        else:
            self.record(name, value)

    def record(self, name, value):
        self.check()
        if name in self.points:
            raise ValueError("Repeated native Qwen forward boundary")
        size = value.numel() * value.element_size()
        if not 0 < size <= MAX_TENSOR_BYTES:
            raise ValueError("Native Qwen forward boundary exceeds hash budget")
        self.points[name] = dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                                 storage_offset=value.storage_offset(), **self.digest(value, check=self.check))

    def first_input(self, _module, args, kwargs):
        if self.active:
            self.record("decoder_00_input", args[0] if args else kwargs["hidden_states"])

    def observe(self, name):
        def after(_module, _args, output):
            if self.active:
                self.record(name, output[0] if isinstance(output, (tuple, list)) else output)
        return after

    def forward_entry(self, ordinal):
        if ordinal == self.ordinal:
            if self.finished_forward or self.active:
                raise ValueError("Repeated selected forward")
            self.active = True

    def forward_returned(self, ordinal):
        if ordinal == self.ordinal:
            if not self.active or self.points.keys() != self.expected:
                raise ValueError("Incomplete selected native Qwen forward")
            self.active = False
            self.finished_forward = True

    def backward_returned(self, _ordinal):
        pass

    def finish(self):
        for handle in self.handles:
            handle.remove()
        if not self.finished_forward or self.active:
            raise ValueError("Selected native Qwen forward was not observed")
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(dict(complete=True, rank=self.rank, ordinal=self.ordinal,
                           selected_decoder=SELECTED_DECODERS[self.rank], capture_order=list(self.points),
                           boundaries=self.points), stream, allow_nan=False)


def observe_first_forward(model, path, check, rank, *, include_suffix, include_internals):
    if rank not in POLICY_RANKS or not include_suffix or include_internals:
        raise ValueError("Native first-forward selection differs from the frozen request")
    global _PROBE
    if _PROBE is not None:
        raise ValueError("Qwen decoder probe already installed in this rank")
    _PROBE = FirstForward(model, path, check, rank)
    return _PROBE


def record_sdpa_inputs(module, query, key, value, mask):
    if _PROBE is not None and _PROBE.active and module is _PROBE.attention:
        for name, tensor in (("q", query), ("k", key), ("v", value), ("mask", mask)):
            _PROBE.record_optional(f"sdpa.{name}", tensor)


def record_sdpa_output(module, value):
    if _PROBE is not None and _PROBE.active and module is _PROBE.attention:
        _PROBE.record("sdpa.out", value[0] if isinstance(value, (tuple, list)) else value)


def install():
    from vlm.runtime import qwen_loss_boundary_trace as original

    digest = hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    if digest != ORIGINAL_TRACE_SHA256 or original.TARGET_MICROBATCH != {1: 1, 4: 2}:
        raise ValueError("Native Qwen trace provider differs")
    original.TARGET_MICROBATCH = POLICY_RANKS.copy()
    original.observe_loss_boundaries = observe_first_forward
