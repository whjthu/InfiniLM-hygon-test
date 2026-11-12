"""Observe Swift's native checkpointing lowering without changing training."""
import hashlib
import inspect
from pathlib import Path


def provider_identity(method):
    function = inspect.unwrap(getattr(method, '__func__', method))
    if (not inspect.isfunction(function) or function.__module__ != 'swift.trainers.mixin'
            or function.__qualname__ != 'SwiftMixin._prepare_gradient_checkpointing'):
        raise ValueError('native checkpointing preparation provider differs')
    path = Path(inspect.getfile(function)).resolve()
    if not 0 < path.stat().st_size <= 2*1024**2:
        raise ValueError('native checkpointing provider source exceeds bound')
    raw = path.read_bytes()
    return dict(module=function.__module__, qualname=function.__qualname__, source_file_bytes=len(raw),
                source_file_SHA256=hashlib.sha256(raw).hexdigest(),
                method_source_SHA256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest())


def validate_checkpointing_lifecycle(core, args, binding, method):
    """The recipe stays ON while Swift lowers only the mutable Trainer flag."""
    lifecycle = binding.get('native_checkpointing_lifecycle')
    spec = binding['spec']
    vision_path = {'T01.sft': 'model.visual', 'T03.sft': 'model.vision_tower'}.get(spec['case'])
    if (not isinstance(lifecycle, dict) or vision_path is None
            or spec['training']['gradient_checkpointing'] is not True
            or lifecycle.get('original_requested') is not True
            or lifecycle.get('native_Trainer_flag') is not False
            or lifecycle.get('language_path') != 'model.language_model'
            or lifecycle.get('vision_path') != vision_path
            or lifecycle.get('language_layer_count') != 36):
        raise ValueError('native checkpointing lifecycle binding differs')
    actual_provider = provider_identity(method)
    if actual_provider != lifecycle.get('provider'):
        raise ValueError('native checkpointing preparation source differs')
    if args.gradient_checkpointing is not False:
        raise ValueError('native lowered Trainer checkpointing flag differs')
    language = core.get_submodule(lifecycle['language_path'])
    layers = list(language.layers)
    if (language.is_gradient_checkpointing is not True or len(layers) != 36
            or any(getattr(layer, 'gradient_checkpointing', None) is not True for layer in layers)
            or language.config.use_cache is not False):
        raise ValueError('native effective language checkpointing differs')
    vision = core.get_submodule(lifecycle['vision_path'])
    vision_flags = [(name, module.gradient_checkpointing) for name, module in vision.named_modules()
                    if hasattr(module, 'gradient_checkpointing')]
    if not vision_flags or any(flag is not False for _, flag in vision_flags):
        raise ValueError('native frozen vision checkpointing differs')
    return dict(original_requested=True, native_Trainer_flag=False, language_checkpointing=True,
                language_checkpointed_layers=len(layers), language_use_cache=False,
                vision_checkpointing=False, vision_checkpointing_flags=len(vision_flags), provider=actual_provider)
