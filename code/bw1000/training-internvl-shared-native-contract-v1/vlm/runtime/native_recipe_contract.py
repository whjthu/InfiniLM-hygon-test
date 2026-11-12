"""Validate native VLM update recording against a frozen per-case binding."""
import math
import re

from .checkpointing_lifecycle import validate_checkpointing_lifecycle


def model_contract(case):
    """Keep model boundaries explicit; a caller cannot supply its own scope."""
    if case == 'T02.sft':
        return dict(model_class='Gemma3ForConditionalGeneration', model_type='gemma3',
                    source_prefix='transformers.models.gemma3.', template='gemma3_vision',
                    max_length=512, gradient_checkpointing=False,
                    frozen_paths=['model.vision_tower', 'model.multi_modal_projector'],
                    language_path='model.language_model')
    if case == 'T01.sft':
        return dict(model_class='Qwen3VLForConditionalGeneration', model_type='qwen3_vl',
                    source_prefix='transformers.models.qwen3_vl.', template='qwen3_vl',
                    max_length=4096, gradient_checkpointing=True,
                    frozen_paths=['model.visual'], language_path='model.language_model')
    if case == 'T03.sft':
        return dict(model_class='InternVLForConditionalGeneration', model_type='internvl',
                    source_prefix=('transformers.models.internvl.', 'transformers.models.qwen3.'),
                    template='internvl_hf', max_length=4096, gradient_checkpointing=True,
                    frozen_paths=['model.vision_tower', 'model.multi_modal_projector'],
                    language_path='model.language_model')
    raise ValueError('native update recipe has no qualified model contract')


def _option(command, name, default=None):
    count = command.count(name)
    if count == 0 and default is not None:
        return default
    if count != 1:
        raise ValueError('missing or ambiguous original option: ' + name)
    return command[command.index(name) + 1]


def native_recipe_spec(launch):
    contract = model_contract(launch['case'])
    command = launch['command']
    integers = {'max_steps': '--max_steps', 'gradient_accumulation_steps': '--gradient_accumulation_steps',
                'per_device_train_batch_size': '--per_device_train_batch_size', 'save_steps': '--save_steps',
                'save_total_limit': '--save_total_limit'}
    floats = {'learning_rate': '--learning_rate', 'weight_decay': '--weight_decay',
              'adam_beta1': '--adam_beta1', 'adam_beta2': '--adam_beta2', 'adam_epsilon': '--adam_epsilon',
              'warmup_ratio': '--warmup_ratio'}
    expected = {key: int(_option(command, option)) for key, option in integers.items()}
    expected.update({key: float(_option(command, option)) for key, option in floats.items()})
    expected.update(bf16=True, fp16=False,
                    gradient_checkpointing=contract['gradient_checkpointing'], max_grad_norm=1.0,
                    optim=_option(command, '--optim'), lr_scheduler_type=_option(command, '--lr_scheduler_type'))
    if launch['case'] == 'T03.sft':
        if _option(command, '--model_type') != 'internvl' or '--template' in command:
            raise ValueError('original InternVL HF model/default template differs')
    elif _option(command, '--template') != contract['template']:
        raise ValueError('original native VLM template differs')
    if (_option(command, '--torch_dtype') != 'bfloat16'
            or _option(command, '--gradient_checkpointing') != str(contract['gradient_checkpointing']).lower()
            or _option(command, '--freeze_vit') != 'true'
            or _option(command, '--target_modules') != 'all-linear'
            or int(_option(command, '--lora_rank')) != 16
            or int(_option(command, '--lora_alpha')) != 32
            or float(_option(command, '--lora_dropout')) != .05
            or _option(command, '--attn_impl') != 'sdpa'
            or int(_option(command, '--max_length')) != contract['max_length']
            or expected['max_steps'] != 25 or expected['gradient_accumulation_steps'] != 4
            or expected['per_device_train_batch_size'] != 1):
        raise ValueError('original native VLM recipe differs')
    return dict(case=launch['case'], original_launch_SHA256=launch['a100_launch_sha256'],
                training=expected, effective_warmup_steps=math.ceil(expected['max_steps'] * expected['warmup_ratio']),
                model_class=contract['model_class'], attention='sdpa',
                frozen_paths=contract['frozen_paths'], language_path=contract['language_path'],
                lora=dict(r=16, lora_alpha=32, lora_dropout=.05, bias='none'))



def target_signature(value):
    """Ignore only order in literal-identifier regex alternatives.

    PEFT uses target regexes for a boolean fullmatch. Reordering an alternative
    of plain identifiers preserves that language; all other syntax stays exact.
    Complete trainable manifests and frozen boundaries are checked separately.
    """
    if isinstance(value, str):
        return ('regex', re.sub(r'\(([A-Za-z_][A-Za-z_0-9]*(?:\|[A-Za-z_][A-Za-z_0-9]*)+)\)',
            lambda match: '(' + '|'.join(sorted(match.group(1).split('|'))) + ')', value))
    if isinstance(value, (set, list, tuple)) and all(isinstance(x, str) for x in value):
        return ('suffixes', tuple(sorted(value)))
    raise ValueError('invalid native target specification')


def parameter_manifest(named):
    return [dict(name=n, shape=list(p.shape), dtype=str(p.dtype), numel=p.numel()) for n, p in named]


def validate_native_recipe(model, args, binding, *, checkpointing_method=None):
    spec = binding['spec']
    contract = model_contract(spec['case'])
    if (any(spec[key] != contract[key] for key in ('model_class', 'frozen_paths', 'language_path'))
            or spec['attention'] != 'sdpa'
            or spec['training']['gradient_checkpointing'] != contract['gradient_checkpointing']):
        raise ValueError('native model binding scope differs')
    core = model.get_base_model()
    if type(core).__name__ != spec['model_class'] or core.config.model_type != contract['model_type']:
        raise ValueError('native base architecture differs')
    if spec['case'] == 'T03.sft':
        if (binding.get('resolved_template_type') != 'internvl_hf'
                or core.config.text_config.model_type != 'qwen3'
                or type(core.get_submodule(spec['language_path'])).__name__ != 'Qwen3Model'):
            raise ValueError('native InternVL HF template/text architecture differs')
    lifecycle = binding.get('native_checkpointing_lifecycle')
    if lifecycle is not None:
        validate_checkpointing_lifecycle(core, args, binding, checkpointing_method)
    for name, expected in spec['training'].items():
        if name == 'gradient_checkpointing' and lifecycle is not None:
            expected = False
        actual = getattr(args, name)
        if getattr(actual, 'value', actual) != expected:
            raise ValueError('native Trainer argument differs: ' + name)
    if args.get_warmup_steps(args.max_steps) != spec['effective_warmup_steps']:
        raise ValueError('native effective warmup differs')
    active = model.active_adapter
    if not isinstance(active, str) or active not in model.peft_config:
        raise ValueError('expected one native PEFT adapter')
    config = model.peft_config[active]
    for name, expected in spec['lora'].items():
        if getattr(config, name) != expected:
            raise ValueError('native PEFT configuration differs: ' + name)
    if target_signature(config.target_modules) != target_signature(binding['target_modules']):
        raise ValueError('expanded native LoRA targets differ')
    if config.modules_to_save or config.use_dora or config.use_rslora:
        raise ValueError('unexpected trainable adapter scope')
    if bool(core.is_gradient_checkpointing) != contract['gradient_checkpointing']:
        raise ValueError('native gradient checkpointing differs')
    if core.get_submodule(spec['language_path']).config._attn_implementation != spec['attention']:
        raise ValueError('native attention differs')
    named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    if (not named or any('.lora_A.' not in n and '.lora_B.' not in n for n, _ in named)
            or parameter_manifest(named) != binding['trainable_manifest']):
        raise ValueError('complete native trainable names,shapes,dtypes or counts differ')
    frozen = {}
    for path in spec['frozen_paths']:
        parameters = list(core.get_submodule(path).parameters())
        if not parameters or any(p.requires_grad for p in parameters):
            raise ValueError('native frozen boundary differs: ' + path)
        frozen[path] = dict(tensors=len(parameters), parameters=sum(p.numel() for p in parameters))
    return named, frozen
