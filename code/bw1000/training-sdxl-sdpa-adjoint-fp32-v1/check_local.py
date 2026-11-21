"""Check the shared production formula against an independent numerical CPU reference."""

import json
from math import sqrt
from pathlib import Path
import unittest

import numpy as np

from adjoint import vjp
from intervention import PROJECTIONS, REGIONS, trainable_attention_sites


def softmax(x):
    scores = np.exp(x - x.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def backward(q, k, v, dy):
    return vjp(q, k, v, dy, 1 / sqrt(q.shape[-1]), softmax=softmax,
               row_sum=lambda x: x.sum(axis=-1, keepdims=True), concatenate=lambda x: np.concatenate(x, axis=-2))


class AttentionAdjointTests(unittest.TestCase):
    def test_all_operands_match_central_difference(self):
        rng = np.random.default_rng(174)
        q = rng.normal(size=(1, 2, 5, 4))
        k, v = (rng.normal(size=(1, 2, 3, 4)) for _ in range(2))
        dy = rng.normal(size=q.shape)
        scale = 1 / np.sqrt(q.shape[-1])

        def objective(items):
            scores = np.einsum('bhld,bhsd->bhls', items[0], items[1]) * scale
            return np.einsum('bhls,bhsd,bhld->', softmax(scores), items[2], dy)

        originals = [q, k, v]
        for array, gradient in zip(originals, backward(q, k, v, dy), strict=True):
            for index in np.ndindex(array.shape):
                old = array[index]
                array[index] = old + 1e-5
                high = objective(originals)
                array[index] = old - 1e-5
                low = objective(originals)
                array[index] = old
                self.assertAlmostEqual(gradient[index], (high-low)/2e-5, places=8)

    def test_fp32_multiple_blocks_and_final_partial_block(self):
        rng = np.random.default_rng(42)
        q = rng.normal(size=(1, 3, 133, 64)).astype(np.float32)
        k, v = (rng.normal(size=(1, 3, 77, 64)).astype(np.float32) for _ in range(2))
        direction = rng.normal(size=q.shape).astype(np.float32)
        gradients = backward(q, k, v, direction)

        scale = 1 / np.sqrt(q.shape[-1])
        probs = softmax(np.einsum('bhld,bhsd->bhls', q, k, dtype=np.float64) * scale)
        dp = np.einsum('bhld,bhsd->bhls', direction, v, dtype=np.float64)
        ds = probs * (dp - (dp * probs).sum(axis=-1, keepdims=True))
        references = (np.einsum('bhls,bhsd->bhld', ds, k) * scale,
                      np.einsum('bhls,bhld->bhsd', ds, q) * scale,
                      np.einsum('bhls,bhld->bhsd', probs, direction))
        for actual, want in zip(gradients, references, strict=True):
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, want, rtol=2e-5, atol=2e-6)

    def test_scope_matches_the_actual_parameter_count_by_block(self):
        repo = Path(__file__).resolve().parents[3]
        saved = repo/'results/bw1000/2026-11-20/training-sdxl-original528-update-error-by-block.json'
        observed = json.loads(saved.read_bytes())['result']['blocks']
        self.assertEqual({region: count*8 for region, count in REGIONS.items()},
                         {region: details['tensors'] for region, details in observed.items()})
        names = [f'{region}.attentions.0.transformer_blocks.{i}.attn1.{projection}.lora_{operand}.default.weight'
                 for region, count in REGIONS.items() for i in range(count)
                 for projection in PROJECTIONS for operand in ('A', 'B')]
        self.assertEqual(len(trainable_attention_sites(names)), 140)
        for drift in (names[:-1], names + names[:1], names[:-1] + ['conv_out.weight']):
            with self.assertRaises(ValueError):
                trainable_attention_sites(drift)


if __name__ == '__main__':
    unittest.main()
