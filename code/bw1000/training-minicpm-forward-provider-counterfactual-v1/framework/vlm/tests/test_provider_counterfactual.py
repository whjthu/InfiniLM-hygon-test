"""Safety checks for instance-only provider arm selection and restoration."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from provider_counterfactual import selected_callables

class ProviderArmTest(unittest.TestCase):
    def test_exact_one_attribute_and_restore_after_failure(self):
        native = lambda x: x + 1
        other = lambda x: x - 1
        owners = [SimpleNamespace(chunk_gated_delta_rule=native, recurrent_gated_delta_rule=other) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, 'body failed'):
            with selected_callables([(str(i), o) for i, o in enumerate(owners)], lambda x: x * 3) as rows:
                self.assertEqual([o.chunk_gated_delta_rule(5) for o in owners], [15, 15])
                self.assertEqual([r['calls'] for r in rows], [1, 1])
                self.assertTrue(all(o.recurrent_gated_delta_rule is other for o in owners))
                raise RuntimeError('body failed')
        self.assertTrue(all(o.chunk_gated_delta_rule is native for o in owners))

    def test_native_arm_counts_without_replacement(self):
        f = lambda x: x + 2
        o = SimpleNamespace(chunk_gated_delta_rule=f)
        with selected_callables([('one', o)]) as rows:
            self.assertEqual(o.chunk_gated_delta_rule(7), 9)
            self.assertEqual(rows[0]['calls'], 1)
        self.assertIs(o.chunk_gated_delta_rule, f)

    def test_nested_duplicate_and_invalid_rejected_without_mutation(self):
        f = lambda x: x
        o = SimpleNamespace(chunk_gated_delta_rule=f)
        for rows, replacement in [([], None), ([('x', o), ('y', o)], None), ([('x', o)], 1)]:
            with self.assertRaises(ValueError):
                with selected_callables(rows, replacement):
                    self.fail('invalid arm accepted')
            self.assertIs(o.chunk_gated_delta_rule, f)
        with selected_callables([('x', o)]):
            with self.assertRaises(ValueError):
                with selected_callables([('x', o)]):
                    self.fail('nested arm accepted')
        self.assertIs(o.chunk_gated_delta_rule, f)

    def test_external_mutation_is_reported_after_restoration(self):
        f = lambda x: x
        o = SimpleNamespace(chunk_gated_delta_rule=f)
        with self.assertRaisesRegex(RuntimeError, 'changed externally'):
            with selected_callables([('x', o)]):
                o.chunk_gated_delta_rule = lambda x: 0
        self.assertIs(o.chunk_gated_delta_rule, f)

if __name__ == '__main__':
    unittest.main()
