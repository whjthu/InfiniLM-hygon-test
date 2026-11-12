"""Lowered Trainer flags cannot substitute for real language checkpointing."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.checkpointing_lifecycle import provider_identity, validate_checkpointing_lifecycle
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.tests.test_qwen_native_recipe_contract import structural_fixture


class CheckpointingLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name)/'mixin.py'
        path.write_text('class SwiftMixin:\n    def _prepare_gradient_checkpointing(self, model):\n        pass\n')
        spec = importlib.util.spec_from_file_location('swift.trainers.mixin', path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.method = module.SwiftMixin._prepare_gradient_checkpointing
        self.provider = provider_identity(self.method)

    def fixture(self):
        model, args, binding, core, frozen = structural_fixture()
        language = core.get_submodule('model.language_model')
        language.is_gradient_checkpointing = True
        language.layers = [SimpleNamespace(gradient_checkpointing=True) for _ in range(36)]
        language.config.use_cache = False
        vision = core.get_submodule('model.visual')
        vision.gradient_checkpointing = False
        vision.named_modules = lambda: iter([('', vision)])
        args.gradient_checkpointing = False
        binding['native_checkpointing_lifecycle'] = dict(original_requested=True,native_Trainer_flag=False,
            language_path='model.language_model',vision_path='model.visual',language_layer_count=36,
            provider=deepcopy(self.provider))
        return model,args,binding,core,frozen

    def test_lowered_flag_requires_effective_model_state_and_original_recipe_stays_on(self):
        model,args,binding,core,_ = self.fixture()
        state = validate_checkpointing_lifecycle(core,args,binding,self.method)
        self.assertTrue(binding['spec']['training']['gradient_checkpointing'])
        self.assertFalse(args.gradient_checkpointing)
        self.assertEqual(state['language_checkpointed_layers'],36)
        self.assertEqual(len(validate_native_recipe(model,args,binding,checkpointing_method=self.method)[0]),1)

    def test_missing_binding_or_method_does_not_allow_a_false_flag(self):
        model,args,binding,_,_ = self.fixture()
        with self.assertRaises(ValueError):
            validate_native_recipe(model,args,binding)
        binding.pop('native_checkpointing_lifecycle')
        with self.assertRaisesRegex(ValueError,'gradient_checkpointing'):
            validate_native_recipe(model,args,binding)

    def test_layer_vision_cache_and_trainer_state_regressions_are_rejected(self):
        for kind in ('language','layer','layer_count','vision','cache','Trainer_flag'):
            model,args,binding,core,_ = self.fixture()
            language=core.get_submodule('model.language_model')
            if kind=='language':language.is_gradient_checkpointing=False
            elif kind=='layer':language.layers[17].gradient_checkpointing=False
            elif kind=='layer_count':language.layers.pop()
            elif kind=='vision':core.get_submodule('model.visual').gradient_checkpointing=True
            elif kind=='cache':language.config.use_cache=True
            else:args.gradient_checkpointing=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                validate_native_recipe(model,args,binding,checkpointing_method=self.method)

    def test_changed_provider_and_recipe_bindings_are_rejected(self):
        for kind in ('source','method_source','recipe','case','path'):
            model,args,binding,_,_ = self.fixture()
            if kind=='source':binding['native_checkpointing_lifecycle']['provider']['source_file_SHA256']='0'*64
            elif kind=='method_source':binding['native_checkpointing_lifecycle']['provider']['method_source_SHA256']='0'*64
            elif kind=='recipe':binding['spec']['training']['gradient_checkpointing']=False
            elif kind=='case':binding['spec']['case']='T02.sft'
            else:binding['native_checkpointing_lifecycle']['language_path']='model.visual'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                validate_native_recipe(model,args,binding,checkpointing_method=self.method)

    def test_ordinary_noncheckpointing_argument_validation_is_preserved(self):
        model,args,binding,_,_ = self.fixture();args.learning_rate=.5
        with self.assertRaisesRegex(ValueError,'learning_rate'):
            validate_native_recipe(model,args,binding,checkpointing_method=self.method)


if __name__ == '__main__':
    unittest.main()
