"""Rank-four BF16 saved-operand algebra, with an independent NumPy oracle."""
import math
import unittest
import numpy as np
import torch
from gencuda.trellis_operand_attribution import attribute_site


def row(x, dy):
    x = torch.tensor(x, dtype=torch.bfloat16)
    dy = torch.tensor(dy, dtype=torch.bfloat16)
    return dict(input=x, upstream=dy, gradient=(dy.double().T @ x.double()).to(torch.bfloat16))


class RankFourAttributionTests(unittest.TestCase):
    def setUp(self):
        self.a = row([[1, 2, 3, 4], [-1, 2, -3, 4]], [[2, -3, 1], [4, 5, -2]])
        self.h = row([[2, 2, 2, 4], [-2, 2, -3, 5]], [[1, -4, 2], [6, 5, -3]])

    def compute(self, a, h):
        return attribute_site(a, h, rank=4, gradient_dtype=torch.bfloat16)

    def test_independent_oracle_and_gram_closure(self):
        actual = self.compute(self.a, self.h)
        xa, xh = [v['input'].float().numpy().astype(np.float64) for v in (self.a, self.h)]
        ya, yh = [v['upstream'].float().numpy().astype(np.float64) for v in (self.a, self.h)]
        aa = np.einsum('ri,rj->ji', xa, ya, optimize=False)
        hh = np.einsum('ri,rj->ji', xh, yh, optimize=False)
        expected = math.fsum(float(v) ** 2 for v in (hh-aa).reshape(-1))
        self.assertEqual(actual['oracle_hh_vs_aa']['squared_error'], expected)
        self.assertAlmostEqual(actual['decomposition']['summed_squared_error'], expected)
        self.assertLess(actual['decomposition']['residual_max_abs'], 1e-12)

    def test_input_only_control(self):
        h = row(self.h['input'].tolist(), self.a['upstream'].tolist())
        result = self.compute(self.a, h)
        self.assertEqual(result['a100_input_hygon_upstream_vs_aa']['squared_error'], 0)
        self.assertEqual(result['decomposition']['gram'][1][1], 0)
        self.assertEqual(result['decomposition']['gram'][2][2], 0)

    def test_upstream_only_control(self):
        h = row(self.a['input'].tolist(), self.h['upstream'].tolist())
        result = self.compute(self.a, h)
        self.assertEqual(result['hygon_input_a100_upstream_vs_aa']['squared_error'], 0)
        self.assertEqual(result['decomposition']['gram'][0][0], 0)
        self.assertEqual(result['decomposition']['gram'][2][2], 0)

    def test_explicit_rank_and_dtype_are_not_inferred(self):
        with self.assertRaises(ValueError):
            attribute_site(self.a, self.h)
        for rank in (0, True, 65):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                attribute_site(self.a, self.h, rank=rank, gradient_dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            self.compute(self.a, dict(self.h, gradient=self.h['gradient'].float()))

    def test_actual_three_dimensional_layout_and_nonfinite_rejection(self):
        a = {**self.a, 'input': self.a['input'].reshape(1, 2, 4),
             'upstream': self.a['upstream'].reshape(1, 2, 3)}
        self.assertEqual(self.compute(a, a)['oracle_hh_vs_aa']['relative_l2'], 0)
        with self.assertRaises(ValueError):
            self.compute(a, dict(a, upstream=torch.full((1, 2, 3), float('nan'), dtype=torch.bfloat16)))


if __name__ == '__main__':
    unittest.main()
