"""Pure evidence/opt-in checks; fixtures are not executed model evidence."""
import copy
import unittest

from vlm.runtime.forward_boundary_trace import compare_boundary_traces, validate_boundary_opt_in


def trace():
    names = ['embedding', 'mixer', 'decoder']
    return dict(schema=1, complete=True, failed=False, raw_values_exported=False,
                planned_boundaries=names, planned_count=3, completed_boundaries=3,
                copied_logical_bytes=24, max_logical_bytes=1024,
                records=[dict(name=name, ordinal=i + 1, value_SHA256=str(i) * 64,
                              observer_RNG_unchanged=True) for i, name in enumerate(names)])


class BoundaryEvidenceTest(unittest.TestCase):
    def test_explicit_opt_in_requires_one_complete_tail_update(self):
        self.assertFalse(validate_boundary_opt_in(False, None, None))
        self.assertTrue(validate_boundary_opt_in(True, 20, 21))
        for enabled, entry, stop in [(None, 20, 21), (1, 20, 21), ('true', 20, 21),
                                     (True, None, None), (True, 19, 20), (True, 20, 25)]:
            with self.subTest(enabled=enabled, entry=entry, stop=stop), self.assertRaises(ValueError):
                validate_boundary_opt_in(enabled, entry, stop)

    def test_first_difference_keeps_full_denominator(self):
        a, b = trace(), trace()
        b['records'][1]['value_SHA256'] = 'f' * 64
        b['records'][2]['value_SHA256'] = 'e' * 64
        v = compare_boundary_traces(a, b)
        self.assertEqual(v['first_value_difference'], dict(ordinal=2, name='mixer'))
        self.assertEqual((v['equal_boundaries'], v['boundary_denominator']), (1, 3))
        self.assertFalse(v['operator_cause_proven'])

    def test_equal_hashes_do_not_accept_numerics_or_recovery(self):
        v = compare_boundary_traces(trace(), trace())
        self.assertEqual(v['state'], 'OBSERVED_BOUNDARY_VALUES_EQUAL')
        self.assertIsNone(v['first_value_difference'])
        self.assertFalse(v['numerical_acceptance'])

    def test_incomplete_duplicate_bad_hash_and_over_budget_rejected(self):
        mutations = [lambda v: v.update(complete=False),
                     lambda v: v['records'].pop(),
                     lambda v: v['records'][1].update(name='embedding'),
                     lambda v: v['records'][0].update(value_SHA256='z' * 64),
                     lambda v: v['records'][0].update(observer_RNG_unchanged=False),
                     lambda v: v.update(copied_logical_bytes=1025),
                     lambda v: v.update(raw_values_exported=True)]
        for mutate in mutations:
            bad = copy.deepcopy(trace()); mutate(bad)
            with self.assertRaises(ValueError):
                compare_boundary_traces(trace(), bad)

    def test_changed_order_does_not_invent_a_value_boundary(self):
        b = trace(); b['records'].reverse()
        for i, row in enumerate(b['records']):
            row['ordinal'] = i + 1
        v = compare_boundary_traces(trace(), b)
        self.assertEqual(v['state'], 'OBSERVED_ORDER_DIFFERS')
        self.assertIsNone(v['first_value_difference'])
