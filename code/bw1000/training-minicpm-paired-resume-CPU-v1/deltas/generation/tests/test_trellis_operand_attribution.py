import unittest

import torch

from gencuda.trellis_operand_attribution import attribute_site, attribute_sites


def row(value, upstream):
    value = torch.tensor(value, dtype=torch.bfloat16)
    upstream = torch.tensor(upstream, dtype=torch.bfloat16)
    return dict(input=value, upstream=upstream,
                gradient=(upstream.double().T @ value.double()).float())


class OperandAttributionTest(unittest.TestCase):
    def test_two_operand_decomposition_and_single_operand_controls(self):
        a = row([[1, 2], [3, 4]], [[2, 3], [5, 7]])
        h = row([[2, 2], [3, 6]], [[1, 4], [6, 7]])
        result = attribute_site(a, h)
        self.assertLess(result['decomposition']['residual_max_abs'], 1e-12)
        self.assertEqual(attribute_site(a, dict(h, input=a['input']))[
            'hygon_input_a100_upstream_vs_aa']['squared_error'], 0)
        self.assertEqual(attribute_site(a, dict(h, upstream=a['upstream']))[
            'a100_input_hygon_upstream_vs_aa']['squared_error'], 0)

    def test_identical_operands_and_site_denominator(self):
        a = row([[1, 2]], [[3, 4]])
        result = attribute_sites({'site': a}, {'site': a})
        self.assertEqual(result['four_site_aggregate']['oracle_hh_vs_aa']['relative_l2'], 0)
        with self.assertRaises(ValueError):
            attribute_sites({'site': a}, {'other': a})

    def test_invalid_original_shapes_and_values_are_rejected(self):
        a = row([[1, 2]], [[3, 4]])
        for field, value in [('input', torch.ones(2, 3, dtype=torch.bfloat16)),
                             ('upstream', torch.full((1, 2), float('nan'), dtype=torch.bfloat16)),
                             ('gradient', a['gradient'].to(torch.bfloat16))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                attribute_site(a, dict(a, **{field: value}))


if __name__ == '__main__':
    unittest.main()
