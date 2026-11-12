"""A vision wrapper alone cannot qualify an unverified embedded text model."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from vlm.runtime.provider_sources import model_provider_sources


class ProviderSourcesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rows = []
        self.pins = {}
        for family in ('internvl', 'qwen3', 'qwen3_vl'):
            name = 'transformers.models.'+family+'.unit_source'
            path = self.root/(family+'.py');path.write_text('class Module:\n    family='+repr(family)+'\n')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            previous = sys.modules.get(name);sys.modules[name] = module
            def restore(name=name, previous=previous):
                if previous is None:sys.modules.pop(name, None)
                else:sys.modules[name] = previous
            self.addCleanup(restore)
            spec.loader.exec_module(module)
            self.rows.append((family, module.Module()))
            self.pins['transformers.models.'+family+'.'] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_composite_model_requires_both_family_pins_and_actual_class_graphs(self):
        model = SimpleNamespace(named_modules=lambda: iter(self.rows[:2]))
        pins = {k:v for k,v in self.pins.items() if 'qwen3_vl' not in k}
        result = model_provider_sources(model, pins, 'T03.sft')
        self.assertEqual(len(result['files']), 2)
        self.assertEqual(len(result['classes']), 2)
        for bad in ({}, {'transformers.models.internvl.':pins['transformers.models.internvl.']},
                    {**pins, 'other.':'0'*64}, {**pins, 'transformers.models.qwen3.':'0'*64}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_provider_sources(model, bad, 'T03.sft')

    def test_missing_embedded_decoder_provider_fails_even_when_vision_is_pinned(self):
        model = SimpleNamespace(named_modules=lambda: iter(self.rows[:1]))
        pins = {k:v for k,v in self.pins.items() if 'qwen3_vl' not in k}
        with self.assertRaises(ValueError):
            model_provider_sources(model, pins, 'T03.sft')

    def test_single_family_qwen_binding_remains_a_single_SHA(self):
        model = SimpleNamespace(named_modules=lambda: iter(self.rows[2:]))
        result = model_provider_sources(model, self.pins['transformers.models.qwen3_vl.'], 'T01.sft')
        self.assertEqual(len(result['files']), 1)
        with self.assertRaises(ValueError):
            model_provider_sources(model, {'transformers.models.qwen3_vl.':self.pins['transformers.models.qwen3_vl.']}, 'T01.sft')


if __name__ == '__main__':
    unittest.main()
