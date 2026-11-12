"""Tiny real PyTorch CPU hooks; never a native MiniCPM or GPU qualification."""
import copy
import unittest

import torch

from vlm.runtime.forward_boundary_trace import ForwardBoundaryTrace


class BoundaryHookTest(unittest.TestCase):
    def start(self, modules, **kwargs):
        self.progress = []
        return ForwardBoundaryTrace(modules, check=lambda: None,
            rng_snapshot=lambda: torch.random.get_rng_state().tolist(),
            progress=lambda row: self.progress.append(copy.deepcopy(row)), **kwargs)

    def test_real_output_gradients_rng_preserved_and_handles_removed(self):
        torch.manual_seed(7)
        first, second = torch.nn.Linear(3, 3), torch.nn.Linear(3, 2)
        model = torch.nn.Sequential(first, second)
        value = torch.tensor([[.2, -.3, .5]], requires_grad=True)
        expected = model(value); expected.sum().backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(); value.grad = None
        rng = torch.random.get_rng_state().clone()
        trace = self.start([('first', first), ('second', second)])
        actual = model(value); actual.sum().backward(); result = trace.finish()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(torch.equal(a, p.grad) for a, p in zip(gradients, model.parameters())))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(result['completed_boundaries'], 2)
        self.assertEqual(result['copied_logical_bytes'], (3 + 2) * 4)
        self.assertFalse(first._forward_hooks or second._forward_hooks)
        self.assertFalse(result['numerical_acceptance'])

    def test_duplicate_call_fails_and_preserves_partial_denominator(self):
        layer = torch.nn.Identity(); trace = self.start([('only', layer)])
        value = torch.ones(2); layer(value)
        with self.assertRaisesRegex(ValueError, 'repeated'):
            layer(value)
        self.assertEqual(self.progress[-1]['completed_boundaries'], 1)
        self.assertEqual(self.progress[-1]['planned_count'], 1)
        self.assertTrue(self.progress[-1]['failed'])
        self.assertFalse(layer._forward_hooks)

    def test_limit_checked_before_any_value_copy(self):
        layer = torch.nn.Identity(); trace = self.start([('only', layer)], max_logical_bytes=4)
        with self.assertRaisesRegex(ValueError, 'logical-byte limit'):
            layer(torch.ones(2))
        self.assertEqual(self.progress[-1]['copied_logical_bytes'], 0)
        self.assertFalse(layer._forward_hooks)

    def test_nested_noncontiguous_and_bfloat16_values(self):
        class Nested(torch.nn.Module):
            def forward(self, x):
                return {'last_hidden_state': x.T, 'pooler_output': [x[:, :1].to(torch.bfloat16)]}
        layer = Nested(); trace = self.start([('nested', layer)])
        layer(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        result = trace.finish(); values = result['records'][0]['tensors']
        self.assertEqual(len(values), 2)
        self.assertEqual([v['logical_bytes'] for v in values], [48, 6])

    def test_missing_boundary_fails_and_detaches_all(self):
        a, b = torch.nn.Identity(), torch.nn.Identity(); trace = self.start([('a', a), ('b', b)])
        a(torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            trace.finish()
        self.assertEqual(self.progress[-1]['planned_count'], 2)
        self.assertEqual(self.progress[-1]['completed_boundaries'], 1)
        self.assertFalse(a._forward_hooks or b._forward_hooks)

    def test_nonfinite_output_removes_hooks(self):
        layer = torch.nn.Identity(); trace = self.start([('only', layer)])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            layer(torch.tensor([float('nan')]))
        self.assertTrue(self.progress[-1]['failed'])
        self.assertFalse(layer._forward_hooks)

    def test_changed_rng_is_not_accepted(self):
        layer = torch.nn.Identity()
        calls = iter([{'state': 1}, {'state': 2}]); events = []
        trace = ForwardBoundaryTrace([('only', layer)], check=lambda: None,
            rng_snapshot=lambda: next(calls), progress=events.append)
        with self.assertRaisesRegex(ValueError, 'changed RNG'):
            layer(torch.ones(1))
        self.assertFalse(layer._forward_hooks)

    def test_same_module_cannot_masquerade_as_two_boundaries(self):
        layer = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.start([('a', layer), ('b', layer)])
