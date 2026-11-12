"""Structural OneVision contract tests; no native model or numerical qualification."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from vlm.runtime.native_recipe_contract import model_contract, native_recipe_spec, validate_native_recipe
from vlm.tests.test_native_recipe_contract import launch
from vlm.tests.test_qwen_native_recipe_contract import structural_fixture


def llava_launch():
    value = launch()
    value['case'] = 'T04.sft'
    i = value['command'].index('--template')
    del value['command'][i:i+2]
    value['command'] += ['--model_type', 'llava_onevision1_5']
    for key, changed in (('--attn_impl', 'eager'), ('--max_length', '4096')):
        value['command'][value['command'].index(key)+1] = changed
    return value


def fixture():
    model, _, binding, _, frozen = structural_fixture()
    spec = native_recipe_spec(llava_launch())
    module = 'transformers_modules.frozen_onevision.modeling_llavaonevision1_5'
    def instance(name):
        return type(name, (), {'__module__': module})()
    core = instance('LLaVAOneVision1_5_ForConditionalGeneration')
    text = NS(model_type='LLaVAOneVision1_5_text', _attn_implementation='eager')
    vision_config = NS(model_type='rice_vit', _attn_implementation='eager')
    core.config = NS(model_type='llavaonevision1_5', text_config=text,
                     vision_config=vision_config, _attn_implementation='eager')
    core.is_gradient_checkpointing = False
    language = instance('LLaVAOneVision1_5_TextModel')
    language.config, language.gradient_checkpointing = text, False
    language.layers = [NS(self_attn=instance('LLaVAOneVision1_5_Attention')) for _ in range(36)]
    vision = instance('RiceTransformerPretrainedModel')
    vision.config, vision.gradient_checkpointing = vision_config, False
    vision.blocks = [NS(attn=instance('RiceAttention')) for _ in range(24)]
    vision.parameters = lambda: iter([frozen])
    core.get_submodule = {'model.language_model': language, 'model.visual': vision}.__getitem__
    model.get_base_model = lambda: core
    binding.update(spec=spec, resolved_template_type='llava_onevision1_5',
                   model_source_SHA256=model_contract('T04.sft')['model_source_SHA256'])
    args = NS(**spec['training'], get_warmup_steps=lambda steps: 1)
    return model, args, binding, core, language, vision, frozen


class OneVisionNativeContractTest(unittest.TestCase):
    def test_original_implicit_template_eager_no_checkpointing_and_full_context(self):
        value = llava_launch();before = deepcopy(value)
        spec = native_recipe_spec(value)
        self.assertEqual(value, before)
        self.assertEqual(spec['attention'], 'eager')
        self.assertFalse(spec['training']['gradient_checkpointing'])
        self.assertEqual(spec['effective_warmup_steps'], 1)
        self.assertEqual(spec['frozen_paths'], ['model.visual'])
        self.assertNotIn('--template', value['command'])

    def test_recipe_drift_and_explicit_template_are_rejected(self):
        for key, changed in (('--attn_impl', 'sdpa'), ('--gradient_checkpointing', 'true'),
                             ('--max_length', '512'), ('--model_type', 'llava_hf'),
                             ('--gradient_accumulation_steps', '1')):
            value = llava_launch();value['command'][value['command'].index(key)+1] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):native_recipe_spec(value)
        value = llava_launch();value['command'] += ['--template', 'llava_onevision1_5']
        with self.assertRaises(ValueError):native_recipe_spec(value)

    def test_complete_structural_model_and_frozen_boundary(self):
        model, args, binding, *_ = fixture()
        named, frozen = validate_native_recipe(model, args, binding)
        self.assertEqual(len(named), 1)
        self.assertEqual(frozen, {'model.visual': dict(tensors=1, parameters=16)})

    def test_foreign_source_template_nested_model_dispatch_and_layer_are_rejected(self):
        mutations = ('source', 'template', 'text_type', 'vision_type', 'dynamic_module',
                     'root_attention', 'vision_attention', 'language_attention', 'vision_class',
                     'language_class', 'last_text_layer', 'last_vision_layer', 'text_count',
                     'vision_count', 'language_checkpointing', 'vision_checkpointing',
                     'root_checkpointing', 'frozen', 'manifest')
        for kind in mutations:
            model, args, binding, core, language, vision, frozen = fixture()
            if kind == 'source':binding['model_source_SHA256'] = '0'*64
            elif kind == 'template':binding['resolved_template_type'] = 'qwen2_vl'
            elif kind == 'text_type':core.config.text_config.model_type = 'qwen3'
            elif kind == 'vision_type':core.config.vision_config.model_type = 'siglip'
            elif kind == 'dynamic_module':core.__class__ = type(type(core).__name__, (), {})
            elif kind == 'root_attention':core.config.attn_implementation = 'sdpa'
            elif kind == 'vision_attention':vision.config._attn_implementation = 'sdpa'
            elif kind == 'language_attention':language.config._attn_implementation = 'sdpa'
            elif kind == 'vision_class':vision.__class__ = type('OtherVision', (), {})
            elif kind == 'language_class':language.__class__ = type('OtherLanguage', (), {})
            elif kind == 'last_text_layer':language.layers[-1].self_attn = object()
            elif kind == 'last_vision_layer':vision.blocks[-1].attn = object()
            elif kind == 'text_count':language.layers.pop()
            elif kind == 'vision_count':vision.blocks.pop()
            elif kind == 'language_checkpointing':language.gradient_checkpointing = True
            elif kind == 'vision_checkpointing':vision.gradient_checkpointing = True
            elif kind == 'root_checkpointing':core.is_gradient_checkpointing = True
            elif kind == 'frozen':frozen.requires_grad = True
            else:binding['trainable_manifest'][0]['shape'] = [2, 4]
            with self.subTest(kind=kind), self.assertRaises(ValueError):validate_native_recipe(model, args, binding)


if __name__ == '__main__':unittest.main()
