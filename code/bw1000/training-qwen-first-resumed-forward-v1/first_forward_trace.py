"""Capture the first restored native forward on only the two affected ranks."""
import hashlib
import json
from pathlib import Path

POLICY_RANKS = {2: 1, 6: 1}
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
                           capture_order=list(self.points), boundaries=self.points), stream, allow_nan=False)


def observe_first_forward(model, path, check, rank, *, include_suffix, include_internals):
    if rank not in POLICY_RANKS or not include_suffix or include_internals:
        raise ValueError("Native first-forward selection differs from the frozen request")
    return FirstForward(model, path, check, rank)


def install():
    from vlm.runtime import qwen_loss_boundary_trace as original

    digest = hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    if digest != ORIGINAL_TRACE_SHA256 or original.TARGET_MICROBATCH != {1: 1, 4: 2}:
        raise ValueError("Native Qwen trace provider differs")
    original.TARGET_MICROBATCH = POLICY_RANKS.copy()
    original.observe_loss_boundaries = observe_first_forward
