"""Bound callable identity checks, without importing Torch or GPU extensions."""
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.training_providers import provider_inventory


class ProviderInventoryTest(unittest.TestCase):
    def fixture(self, directory):
        name = 'provider_fixture.model'
        path = Path(directory)/'model.py'
        raw = b'''def conv(*args): return None
def chunk(*args): return None
def recurrent(*args): return None
class Delta:
    causal_conv1d_fn = None
    causal_conv1d_update = staticmethod(conv)
    chunk_gated_delta_rule = staticmethod(chunk)
    recurrent_gated_delta_rule = staticmethod(recurrent)
'''
        path.write_bytes(raw)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        old = sys.modules.get(name)
        sys.modules[name] = module
        self.addCleanup(lambda: sys.modules.pop(name, None) if old is None else sys.modules.__setitem__(name, old))
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        delta = module.Delta()
        model = types.SimpleNamespace(named_modules=lambda: [('block.linear_attn', delta)])
        return model, delta, path, dict(model_module_prefix='provider_fixture.',
            expected_model_source_SHA256=hashlib.sha256(raw).hexdigest())

    def test_actual_bound_fallback_functions_and_no_optional_import(self):
        with tempfile.TemporaryDirectory() as d:
            model, _, path, options = self.fixture(d)
            before = set(sys.modules)
            row = provider_inventory(model, **options)
            self.assertEqual(before, set(sys.modules))
            self.assertEqual(len(row['bindings']), 1)
            calls = row['bindings'][0]['callables']
            self.assertFalse(calls['causal_conv1d_fn']['selected'])
            self.assertEqual(calls['chunk_gated_delta_rule']['qualname'], 'chunk')
            self.assertEqual(calls['chunk_gated_delta_rule']['file']['SHA256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_foreign_source_absent_binding_and_noncallable_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model, delta, _, options = self.fixture(d)
            with self.assertRaises(ValueError):
                provider_inventory(model, **{**options, 'expected_model_source_SHA256': '0'*64})
            with self.assertRaises(ValueError):
                provider_inventory(model, **{**options, 'model_module_prefix': 'foreign.'})
            for value in (None, 42):
                delta.chunk_gated_delta_rule = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    provider_inventory(model, **options)


if __name__ == '__main__':
    unittest.main()
