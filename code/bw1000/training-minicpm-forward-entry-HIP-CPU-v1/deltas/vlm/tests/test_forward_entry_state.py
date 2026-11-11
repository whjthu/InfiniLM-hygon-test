"""Real CPU state hashes expose RNG, frozen-weight and buffer differences."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.forward_entry_state import capture_entry_state, tensor_digest


def rng():
    return dict(cpu_SHA256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4),
                                       requires_grad=False)
        self.tied = self.base
        self.adapter = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('persistent', torch.arange(4, dtype=torch.bfloat16))
        self.register_buffer('rotary', torch.arange(4, dtype=torch.float32), persistent=False)
        self.register_buffer('empty', None, persistent=False)
        self.gradient_checkpointing = False
        self.config = SimpleNamespace(use_cache=False, model_type='tiny')


class EntryStateTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = Tiny().train()
        self.x = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, 1::2]

    def capture(self, **kwargs):
        return capture_entry_state(self.model, (self.x,), {}, check=lambda: None,
                                   rng_snapshot=rng, device_type='cpu', **kwargs)

    def test_complete_frozen_trainable_and_nonpersistent_inventory_no_mutation(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        random_before = rng()
        value = self.capture()
        rows = {(r['kind'], r['name']): r for r in value['registered_state']}
        self.assertFalse(rows['parameter', 'base']['requires_grad'])
        self.assertTrue(rows['parameter', 'adapter']['requires_grad'])
        self.assertEqual(rows['parameter', 'tied']['alias_of'], 'parameter:base')
        self.assertFalse(rows['buffer', 'rotary']['persistent'])
        self.assertFalse(rows['buffer', 'empty']['present'])
        self.assertEqual(random_before, rng())
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(self.model.training)
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))
        self.assertFalse(value['recovery_qualified'])
        self.assertFalse(value['formal_performance'])
        self.assertFalse(value['unregistered_python_state_observed'])
        self.assertEqual(value['selected_module_runtime']['']['config']['use_cache'], False)

    def test_frozen_base_change_is_visible_without_adapter_change(self):
        before = self.capture()
        self.model.base[0, 0] += 1
        after = self.capture()
        self.assertNotEqual(before['registered_state_SHA256'], after['registered_state_SHA256'])
        def adapter(value):
            return next(r['SHA256'] for r in value['registered_state'] if r['name'] == 'adapter')
        self.assertEqual(adapter(before), adapter(after))

    def test_nonpersistent_rotary_change_is_visible(self):
        before = self.capture()
        self.model.rotary[0] += 1
        after = self.capture()
        self.assertNotEqual(before['registered_state_SHA256'], after['registered_state_SHA256'])

    def test_same_values_different_layout_and_alignment_are_distinguished(self):
        alignment = self.x.data_ptr() % 256
        before = self.capture()
        self.x = self.x.contiguous()
        after = self.capture()
        self.assertNotEqual(before['input_layouts'], after['input_layouts'])
        self.assertEqual(before['input_alignments']['args/0']['data_mod_256'], alignment)

    def test_strided_hash_equals_logical_bytes_with_bounded_chunks(self):
        value = torch.arange(131072, dtype=torch.float32).reshape(256, 512).t()[::2]
        checks = []
        result = tensor_digest(value, check=lambda: checks.append(1), chunk_bytes=1024)
        self.assertEqual(result['SHA256'], hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
        self.assertEqual(result['logical_bytes'], value.numel() * value.element_size())
        self.assertLessEqual(result['peak_chunk_bytes'], 1024)
        self.assertGreater(len(checks), 100)

    def test_rng_consumption_inside_observation_is_rejected(self):
        def consuming():
            torch.rand(1)
        with self.assertRaisesRegex(ValueError, 'RNG changed'):
            capture_entry_state(self.model, (), {}, check=consuming, rng_snapshot=rng, device_type='cpu')

    def test_deadline_expiry_stops_before_state_copy(self):
        def expired():
            raise TimeoutError('expired')
        with self.assertRaises(TimeoutError):
            capture_entry_state(self.model, (), {}, check=expired, rng_snapshot=rng, device_type='cpu')

    def test_nonfinite_state_and_small_total_bound_rejected(self):
        with self.assertRaisesRegex(ValueError, 'total byte bound'):
            self.capture(max_logical_bytes=1)
        self.model.rotary[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.capture()

    def test_sparse_complex_and_invalid_copy_bounds_rejected(self):
        for value in (torch.ones(3, dtype=torch.complex64), torch.eye(3).to_sparse()):
            with self.assertRaises(ValueError):
                tensor_digest(value, check=lambda: None)
        for bound in (True, 0, 1023, 16 * 1024**2 + 1):
            with self.assertRaises(ValueError):
                tensor_digest(torch.ones(1), check=lambda: None, chunk_bytes=bound)


if __name__ == '__main__':
    unittest.main()
