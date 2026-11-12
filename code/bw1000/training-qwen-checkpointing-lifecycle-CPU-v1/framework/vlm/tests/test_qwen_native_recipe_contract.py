"""Exercise Qwen's checkpointing/frozen-boundary contract without Torch."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.native_recipe_contract import model_contract, native_recipe_spec, validate_native_recipe
from vlm.tests.test_native_recipe_contract import launch


def qwen_launch():
    value = launch()
    value['case'] = 'T01.sft'
    for option, changed in (('--gradient_checkpointing', 'true'), ('--max_length', '4096'),
                            ('--template', 'qwen3_vl')):
        value['command'][value['command'].index(option) + 1] = changed
    return value


def structural_fixture():
    spec = native_recipe_spec(qwen_launch())
    parameter = SimpleNamespace(requires_grad=True, shape=(4, 2), dtype='torch.float32', numel=lambda: 8)
    frozen = SimpleNamespace(requires_grad=False, numel=lambda: 16)
    vision = SimpleNamespace(parameters=lambda: iter([frozen]))
    language = SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa'))
    core = type('Qwen3VLForConditionalGeneration', (), {})()
    core.config = SimpleNamespace(model_type='qwen3_vl')
    core.is_gradient_checkpointing = True
    modules = {'model.visual': vision, 'model.language_model': language}
    core.get_submodule = modules.__getitem__
    named = [('base_model.model.model.language_model.layers.0.q_proj.lora_A.default.weight', parameter)]
    config = SimpleNamespace(**spec['lora'], target_modules=['q_proj'], modules_to_save=None,
                             use_dora=False, use_rslora=False)
    model = SimpleNamespace(active_adapter='default', peft_config={'default': config},
                            get_base_model=lambda: core, named_parameters=lambda: iter(named))
    args = SimpleNamespace(**spec['training'], get_warmup_steps=lambda steps: 1)
    binding = dict(spec=spec, target_modules=['q_proj'], trainable_manifest=[
        dict(name=named[0][0], shape=[4, 2], dtype='torch.float32', numel=8)])
    return model, args, binding, core, frozen


class QwenNativeRecipeTest(unittest.TestCase):
    def test_original_qwen_preserves_checkpointing_context_and_warmup(self):
        value = native_recipe_spec(qwen_launch())
        self.assertTrue(value['training']['gradient_checkpointing'])
        self.assertEqual(value['model_class'], 'Qwen3VLForConditionalGeneration')
        self.assertEqual(value['frozen_paths'], ['model.visual'])
        self.assertEqual(value['effective_warmup_steps'], 1)
        self.assertEqual(model_contract('T01.sft')['max_length'], 4096)
        self.assertEqual(model_contract('T01.sft')['source_prefix'], 'transformers.models.qwen3_vl.')

    def test_gemma_recipe_interpretation_is_unchanged(self):
        value = native_recipe_spec(launch())
        self.assertFalse(value['training']['gradient_checkpointing'])
        self.assertEqual(value['model_class'], 'Gemma3ForConditionalGeneration')
        self.assertEqual(value['frozen_paths'], ['model.vision_tower', 'model.multi_modal_projector'])

    def test_wrong_context_checkpointing_template_and_accumulation_are_rejected(self):
        for option, changed in (('--gradient_checkpointing', 'false'), ('--max_length', '512'),
                                ('--template', 'gemma3_vision'), ('--gradient_accumulation_steps', '1'),
                                ('--attn_impl', 'eager'), ('--max_steps', '2')):
            value = qwen_launch()
            value['command'][value['command'].index(option) + 1] = changed
            with self.subTest(option=option), self.assertRaises(ValueError):
                native_recipe_spec(value)

    def test_structural_double_checks_boundary_and_manifest_not_numerics(self):
        model, args, binding, core, frozen = structural_fixture()
        named, boundaries = validate_native_recipe(model, args, binding)
        self.assertEqual(len(named), 1)
        self.assertEqual(boundaries, {'model.visual': dict(tensors=1, parameters=16)})
        for mutation in ('checkpointing', 'vision', 'architecture', 'attention', 'manifest', 'scope', 'case'):
            model, args, binding, core, frozen = structural_fixture()
            if mutation == 'checkpointing':
                core.is_gradient_checkpointing = False
            elif mutation == 'vision':
                frozen.requires_grad = True
            elif mutation == 'architecture':
                core.config.model_type = 'gemma3'
            elif mutation == 'attention':
                core.get_submodule('model.language_model').config._attn_implementation = 'eager'
            elif mutation == 'manifest':
                binding['trainable_manifest'][0]['shape'] = [2, 4]
            elif mutation == 'scope':
                binding['spec']['frozen_paths'] = []
            else:
                binding['spec']['case'] = 'T03.sft'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_native_recipe(model, args, binding)


if __name__ == '__main__':
    unittest.main()
