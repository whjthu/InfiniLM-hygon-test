"""A source-pinned visual-input decision for OneVision's legacy None position."""

from contextlib import contextmanager
from hashlib import sha256
import inspect
from pathlib import Path
import textwrap
from types import MethodType

from two_forward import need, query_width, seq_length


ONEVISION = "bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198"
METHOD = "d814c7179ccd3fe90d4cc0cb06e570e03129ede8de52b6196a590842b6c922d1"
HF_GENERATION = "dde2df36821c0d724b5af47cb0ff71c3c4c1990c86d81b821911127ae4dc1254"
HF_CACHE = "7935a009f4131f1c21b53fb731f06813bcbaf8a34fe216cb9e83707aaafcf949"
ANCHOR = 'if model_inputs["cache_position"][0] != 0:'
REPLACEMENT = "if _scoped_should_drop_visual(self, model_inputs, kwargs, use_cache):"


def derive_method(raw):
    need(sha256(raw.rstrip(b"\n")).hexdigest() == METHOD, "EXACT_PINNED_ONEVISION_METHOD_TEXT")
    source = textwrap.dedent(raw.decode())
    needs = "super().prepare_inputs_for_generation("
    need(source.count(ANCHOR) == source.count(needs) == 1, "EXACT_VISUAL_AND_PARENT_SOURCE_ANCHORS")
    # A detached method needs the explicit original class for the identical super lookup.
    return source.replace(needs, "super(_exact_original_onevision_class, self).prepare_inputs_for_generation(", 1).replace(
        ANCHOR, REPLACEMENT, 1)


def provider(callable_, expected):
    fn = inspect.unwrap(callable_)
    path = Path(inspect.getsourcefile(fn)).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read((4 << 20) + 1)
    need(len(raw) <= 4 << 20 and sha256(raw).hexdigest() == expected, "EXACT_ACTUAL_PROVIDER_FILE_SHA")


class VisualDecision:
    def __init__(self, torch, cache_type):
        self.torch, self.cache_type = torch, cache_type
        self.events = []

    def should_drop(self, model, model_inputs, kwargs, use_cache):
        position = model_inputs["cache_position"]
        if position is not None:
            # Keep the exact original operation for positions already supplied by HF.
            drop = bool(position[0] != 0)
            self.events.append(dict(position="supplied_unchanged", drop_visual=drop))
            return drop
        cache = model_inputs.get("past_key_values")
        length = seq_length(cache, self.cache_type, self.torch)
        q, _ = query_width(model_inputs)
        first = kwargs.get("is_first_iteration", False)
        wanted = kwargs.get("next_sequence_length")
        need(model.config._attn_implementation == "eager" and model.config.is_encoder_decoder is False
             and use_cache is True and cache.is_compileable is False and type(first) is bool
             and model_inputs.get("pixel_values_videos") is None
             and ((first and length == 0 and q > 1 and model_inputs.get("pixel_values") is not None)
                 or (not first and length > 0 and q == 1 and type(wanted) is int and wanted == 1)),
             "EXACT_BS1_EAGER_GREEDY_DYNAMIC_CACHE_GEOMETRY")
        self.events.append(dict(position="none_native_forward_fallback", cache_length=length, query_tokens=q,
                                first_iteration=first, drop_visual=not first))
        return not first


@contextmanager
def install(model, native_class, torch, cache_type, generation_mixin):
    name = "prepare_inputs_for_generation"
    need(type(model) is native_class and name not in vars(model), "EXACT_UNPATCHED_NATIVE_OWNER_METHOD")
    method = getattr(model, name)
    function = method.__func__
    inherited = getattr(super(native_class, model), name)
    need(inherited.__func__ is getattr(generation_mixin, name) and
         getattr(native_class, name) is function, "EXACT_INSTALLED_GENERATION_PARENT")
    provider(function, ONEVISION)
    provider(inherited, HF_GENERATION)
    provider(cache_type, HF_CACHE)
    source = derive_method(inspect.getsource(function).encode())
    state = VisualDecision(torch, cache_type)
    ns = dict(function.__globals__, _exact_original_onevision_class=native_class,
              _scoped_should_drop_visual=state.should_drop)
    exec(compile(source, "<pinned-onevision-visual-only-cache-repair>", "exec"), ns)
    replacement = MethodType(ns[name], model)
    need(inspect.signature(method) == inspect.signature(replacement), "ORIGINAL_PREPARE_SIGNATURE_PRESERVED")
    setattr(model, name, replacement)
    try:
        yield state
    finally:
        delattr(model, name)
