"""Structural doubles exercise the frozen HF InternVL scope, not real numerics."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from vlm.runtime.native_recipe_contract import model_contract, native_recipe_spec, validate_native_recipe
from vlm.tests import test_checkpointing_lifecycle as lifecycle_fixtures
from vlm.tests.test_qwen_native_recipe_contract import qwen_launch


def internvl_launch():
    value = qwen_launch()
    value['case'] = 'T03.sft'
    i = value['command'].index('--template')
    del value['command'][i:i+2]
    value['command'] += ['--model_type', 'internvl']
    return value


class InternVLNativeRecipeTest(unittest.TestCase):
    def setUp(self):
        provider = lifecycle_fixtures.CheckpointingLifecycleTest()
        provider.setUp()
        self.addCleanup(provider.doCleanups)
        self.provider = provider

    def fixture(self):
        model, args, binding, oldcore, frozen = self.provider.fixture()
        spec = native_recipe_spec(internvl_launch())
        core = type('InternVLForConditionalGeneration', (), {})()
        core.config = SimpleNamespace(model_type='internvl', text_config=SimpleNamespace(model_type='qwen3'))
        core.is_gradient_checkpointing = True
        previous = oldcore.get_submodule('model.language_model')
        language = type('Qwen3Model', (), {})()
        language.__dict__.update(previous.__dict__)
        vision = oldcore.get_submodule('model.visual')
        aligner = SimpleNamespace(parameters=lambda: iter([frozen]))
        modules = {'model.language_model': language, 'model.vision_tower': vision,
                   'model.multi_modal_projector': aligner}
        core.get_submodule = modules.__getitem__
        model.get_base_model = lambda: core
        binding['spec'] = spec
        binding['resolved_template_type'] = 'internvl_hf'
        binding['native_checkpointing_lifecycle']['vision_path'] = 'model.vision_tower'
        return model, args, binding, core, frozen

    def test_original_native_hf_template_is_resolved_without_injecting_option(self):
        launch = internvl_launch()
        before = deepcopy(launch)
        spec = native_recipe_spec(launch)
        self.assertEqual(launch, before)
        self.assertTrue(spec['training']['gradient_checkpointing'])
        self.assertEqual(spec['effective_warmup_steps'], 1)
        self.assertNotIn('--template', launch['command'])
        self.assertEqual(spec['frozen_paths'], ['model.vision_tower', 'model.multi_modal_projector'])
        self.assertEqual(model_contract('T03.sft')['source_prefix'],
                         ('transformers.models.internvl.', 'transformers.models.qwen3.'))

    def test_changed_model_template_context_and_training_are_rejected(self):
        for key, value in (('--model_type', 'internvl3'), ('--max_length', '512'),
                           ('--gradient_checkpointing', 'false'), ('--attn_impl', 'eager')):
            launch = internvl_launch();launch['command'][launch['command'].index(key)+1] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_recipe_spec(launch)
        launch = internvl_launch();launch['command'] += ['--template', 'internvl_hf']
        with self.assertRaises(ValueError):
            native_recipe_spec(launch)

    def test_actual_lifecycle_and_default_frozen_aligner_are_required(self):
        model, args, binding, core, frozen = self.fixture()
        named, boundaries = validate_native_recipe(model, args, binding, checkpointing_method=self.provider.method)
        self.assertEqual(len(named), 1)
        self.assertEqual(set(boundaries), {'model.vision_tower', 'model.multi_modal_projector'})
        frozen.requires_grad = True
        with self.assertRaises(ValueError):
            validate_native_recipe(model, args, binding, checkpointing_method=self.provider.method)

    def test_wrong_hf_resolution_text_decoder_and_vision_mapping_are_rejected(self):
        for kind in ('template', 'text', 'decoder_class', 'vision_path', 'language_layer'):
            model, args, binding, core, _ = self.fixture()
            if kind == 'template':binding['resolved_template_type'] = 'internvl3_5'
            elif kind == 'text':core.config.text_config.model_type = 'qwen2'
            elif kind == 'decoder_class':core.get_submodule('model.language_model').__class__ = type('OtherDecoder', (), {})
            elif kind == 'vision_path':binding['native_checkpointing_lifecycle']['vision_path'] = 'vision_model'
            else:core.get_submodule('model.language_model').layers[4].gradient_checkpointing = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_native_recipe(model, args, binding, checkpointing_method=self.provider.method)


if __name__ == '__main__':
    unittest.main()
