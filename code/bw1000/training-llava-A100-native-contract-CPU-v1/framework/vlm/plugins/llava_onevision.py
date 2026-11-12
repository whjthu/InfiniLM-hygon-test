"""Narrow LLaVA-OneVision 1.5 loader compatibility for Swift external plugins.

This plugin changes only the LLaVA model registration used by the invoking
process. It maps legacy checkpoint prefixes and repairs the two rotary buffers
after native loading. For a pinned runtime missing the standard RoPE factory,
it adds that factory without replacing an existing entry. It intentionally
installs no trainer callbacks, hooks, telemetry, diagnostics, or result writers.
"""

from __future__ import annotations

from copy import copy, deepcopy
import hashlib
from pathlib import Path
import sys
import threading
from typing import Any


LOAD_KEY_MAPPING = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}
_TEXT_PATH = "model.language_model.rotary_emb"
_VISION_PATH = "model.visual.rotary_pos_emb"
_TARGET_SOURCE_NAME = "modeling_llavaonevision1_5.py"
_TARGET_SOURCE_SHA256 = "bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198"
_TARGET_CLASS = "LLaVAOneVision1_5_ForConditionalGeneration"
_PINNED_TRANSFORMERS_VERSION = "5.12.1"
_EAGER_PREIMPORT_LOCK = threading.RLock()


class LlavaCompatibilityError(RuntimeError):
    """The loaded OneVision model is outside the compatibility boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LlavaCompatibilityError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_eager(config: Any) -> bool:
    return all(getattr(config, field, None) in (None, "eager") for field in ("_attn_implementation", "attn_implementation"))


def _resolved_eager(config: Any) -> bool:
    return (
        getattr(config, "_attn_implementation", None) == "eager"
        and getattr(config, "_attn_implementation_internal", None) == "eager"
        and getattr(config, "attn_implementation", None) in (None, "eager")
    )


def _explicit_eager_onevision(config: Any, model_kwargs: dict[str, Any]) -> bool:
    components = (config, getattr(config, "text_config", None), getattr(config, "vision_config", None))
    if "attn_implementation" in model_kwargs:
        return model_kwargs["attn_implementation"] == "eager" and all(_explicit_eager(component) for component in components)
    return all(_resolved_eager(component) for component in components)


def _loaded_eager(config: Any) -> bool:
    values = [getattr(config, field, None) for field in ("_attn_implementation", "attn_implementation")]
    return any(value == "eager" for value in values) and all(value in (None, "eager") for value in values)


def _verified_cached_source(module_file: str, cache_root: str) -> tuple[str, Path] | None:
    """Verify the native dynamic-loader cache entry without changing its bytes."""
    module_path = Path(module_file)
    cache = Path(cache_root).resolve()
    source = module_path if module_path.is_absolute() else cache / module_path
    source = source.resolve()
    try:
        relative = source.relative_to(cache)
    except ValueError:
        return None
    if source.name != _TARGET_SOURCE_NAME or _sha256_file(source) != _TARGET_SOURCE_SHA256:
        return None
    module_name = str(relative.with_suffix("")).replace("\\", ".").replace("/", ".")
    return module_name, source


def _verify_loaded_native_eager_classes(model: Any, module_name: str) -> None:
    _require(_loaded_eager(model.config), "loaded root is not explicitly eager")
    expected = (
        ("model.language_model", "layers", "self_attn", "LLaVAOneVision1_5_Attention"),
        ("model.visual", "blocks", "attn", "RiceAttention"),
    )
    for path, blocks_name, attention_name, attention_class in expected:
        component = model.get_submodule(path)
        _require(_loaded_eager(component.config), f"loaded {path} is not explicitly eager")
        _require(type(component).__module__ == module_name, f"loaded {path} is not a native target class")
        try:
            attention = getattr(component, blocks_name)[0]
            attention = getattr(attention, attention_name)
        except (AttributeError, IndexError, TypeError) as error:
            raise LlavaCompatibilityError(f"loaded {path} has no native eager attention") from error
        _require(type(attention).__module__ == module_name, f"loaded {path} attention is outside the native target")
        _require(type(attention).__name__ == attention_class, f"loaded {path} attention is not eager")


def _preimport_verified_eager_class(model_dir: str, config: Any, model_kwargs: dict[str, Any]) -> str | None:
    """Preimport exactly one native class before Swift starts materializing weights.

    The remote source asks a Transformers utility whether Flash Attention is
    available while its module body executes. In the pinned runtime that answer
    can be true even though the requested utility export is absent. The native
    dynamic loader remains responsible for cache resolution and import. The
    temporary predicate is restored before the original Swift loader runs.
    """
    if not _explicit_eager_onevision(config, model_kwargs):
        return None

    import transformers

    if getattr(transformers, "__version__", None) != _PINNED_TRANSFORMERS_VERSION:
        return None
    from transformers import dynamic_module_utils, modeling_flash_attention_utils as flash_utils

    if hasattr(flash_utils, "flash_attn_varlen_func"):
        return None

    module_file = dynamic_module_utils.get_cached_module_file(
        model_dir, _TARGET_SOURCE_NAME, local_files_only=True
    )
    target = _verified_cached_source(module_file, dynamic_module_utils.HF_MODULES_CACHE)
    if target is None:
        return None
    module_name, source = target
    owner_thread = threading.get_ident()
    with _EAGER_PREIMPORT_LOCK:
        original_predicate = getattr(flash_utils, "is_flash_attn_available", None)
        if not callable(original_predicate):
            return None
        active = True

        def unavailable_flash_attention(*predicate_args: Any, **predicate_kwargs: Any) -> bool:
            caller = sys._getframe(1)
            if (
                active
                and threading.get_ident() == owner_thread
                and caller.f_code.co_name == "<module>"
                and Path(caller.f_code.co_filename).resolve() == source
            ):
                return False
            return original_predicate(*predicate_args, **predicate_kwargs)

        flash_utils.is_flash_attn_available = unavailable_flash_attention
        try:
            loaded_class = dynamic_module_utils.get_class_from_dynamic_module(
                f"{Path(_TARGET_SOURCE_NAME).stem}.{_TARGET_CLASS}", model_dir, local_files_only=True
            )
        finally:
            active = False
            flash_utils.is_flash_attn_available = original_predicate
            target_module = sys.modules.get(module_name)
            if getattr(target_module, "is_flash_attn_available", None) is unavailable_flash_attention:
                target_module.is_flash_attn_available = original_predicate
    _require(loaded_class.__module__ == module_name, "native dynamic loader returned an unexpected LLaVA class")
    return module_name


def _prepare_load_config(config: Any) -> None:
    """Apply instance defaults and an add-only pinned-runtime RoPE fallback."""
    text_config = config.text_config
    for name, value in (("pad_token_id", 151643), ("eos_token_id", 151645), ("bos_token_id", None)):
        if not hasattr(text_config, name):
            setattr(text_config, name, value)

    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        def default_rope_parameters(config: Any = None, device: Any = None, seq_len: Any = None, layer_type: Any = None) -> tuple[Any, float]:
            del seq_len
            if hasattr(config, "standardize_rope_params"):
                config.standardize_rope_params()
            parameters = getattr(config, "rope_parameters", {}) or {}
            if layer_type is not None:
                parameters = parameters[layer_type]
            base = parameters.get("rope_theta", getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0)))
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            dimension = int(head_dim * parameters.get("partial_rotary_factor", 1.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dimension, 2, dtype=torch.int64, device=device).float() / dimension))
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters


def _initialize_loaded_rope(model: Any) -> None:
    """Initialize only the two derived, non-persistent OneVision RoPE buffers.

    The loader invokes this before PEFT wraps the model. It never reloads
    weights, reloads parameters, or replaces an existing Transformers registry
    entry. The load preparation may add a missing default RoPE factory.
    """
    import torch

    _require(not getattr(model, "is_quantized", False), "quantized LLaVA loads are unsupported")
    _require(not hasattr(model, "peft_config"), "RoPE repair must run before PEFT wrapping")
    _require(all(parameter.device.type != "meta" for parameter in model.parameters()), "model parameters are not materialized")
    config = model.config
    text_config = config.text_config
    vision_config = config.vision_config
    _require(getattr(text_config, "hidden_size", None) == 4096, "unsupported OneVision text hidden size")
    _require(getattr(text_config, "num_attention_heads", None) == 32, "unsupported OneVision text head count")
    _require(getattr(text_config, "rope_theta", 1_000_000.0) == 1_000_000.0, "unsupported OneVision text RoPE theta")
    _require(getattr(vision_config, "hidden_size", None) == 1024, "unsupported OneVision vision hidden size")
    _require(getattr(vision_config, "num_heads", None) == 16, "unsupported OneVision vision head count")

    text = model.get_submodule(_TEXT_PATH)
    vision = model.get_submodule(_VISION_PATH)
    _require(hasattr(text, "inv_freq") and hasattr(vision, "inv_freq"), "expected OneVision rotary buffers are absent")
    _require(text.inv_freq.device.type != "meta" and vision.inv_freq.device.type != "meta", "RoPE buffer is not materialized")
    if getattr(model, "_vlm_onevision_rope_initialized", False):
        return

    rice_dim = (vision_config.hidden_size // vision_config.num_heads) // 2
    with torch.no_grad():
        # Native constructors supply the exact version-pinned formulas on CPU.
        fresh_text = type(text)(deepcopy(text_config), device=torch.device("cpu"))
        fresh_vision = type(vision)(rice_dim)
        replacements = ((text, fresh_text.inv_freq), (vision, fresh_vision.inv_freq))
        for module, reference in replacements:
            _require(reference.device.type == "cpu" and reference.dtype == torch.float32, "unexpected native RoPE constructor output")
            module.register_buffer("inv_freq", reference.to(device=module.inv_freq.device, dtype=torch.float32).clone(), persistent=False)
        text.original_inv_freq = text.inv_freq
        text.attention_scaling = fresh_text.attention_scaling
        text.max_seq_len_cached = fresh_text.max_seq_len_cached
        text.original_max_seq_len = fresh_text.original_max_seq_len
    model._vlm_onevision_rope_initialized = True


def register_swift_load_mapping() -> type[Any]:
    """Replace the loader for this single Swift model type."""
    from swift.model.register import MODEL_MAPPING

    metadata = MODEL_MAPPING["llava_onevision1_5"]
    loader_cls = metadata.loader
    if getattr(loader_cls, "_vlm_onevision_loader_installed", False):
        return loader_cls

    class MappedOnevisionLoader(loader_cls):
        _vlm_onevision_loader_installed = True

        def get_model(self, model_dir: str, config: Any, processor: Any, model_kwargs: dict[str, Any]) -> Any:
            supplied = model_kwargs.get("key_mapping")
            if supplied is not None and supplied != LOAD_KEY_MAPPING:
                raise LlavaCompatibilityError("conflicting LLaVA checkpoint key_mapping")
            _prepare_load_config(config)
            target_module = _preimport_verified_eager_class(model_dir, config, model_kwargs)
            model = super().get_model(model_dir, config, processor, {**model_kwargs, "key_mapping": dict(LOAD_KEY_MAPPING)})
            if target_module is not None:
                _verify_loaded_native_eager_classes(model, target_module)
            _initialize_loaded_rope(model)
            return model

    replacement = copy(metadata)
    replacement.loader = MappedOnevisionLoader
    MODEL_MAPPING["llava_onevision1_5"] = replacement
    return MappedOnevisionLoader


if __import__("os").environ.get("VLM_PLUGIN_DISABLE_AUTO_INSTALL") != "1":
    register_swift_load_mapping()
