from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import compare_deep_c4_direct_history as c4


class DirectHistoryComparisonTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.binding = json.loads(c4.INPUT.read_bytes())
        self.weight_keys = sorted(json.loads(c4.METADATA.read_bytes())['weights'])
        self.range_pin = 'a' * 64

    @staticmethod
    def write(path, data):
        path.write_bytes(data)
        return sha256(data).hexdigest()

    def arm(self, name, *, ties=False, score_changed=False, truncated=False, range_pin=None):
        root = self.root / name
        root.mkdir()
        close = [c4.FLOOR + 1, c4.FLOOR + (name == 'candidate'), 1, 1]
        counts = [7, 7, c4.WEIGHT_BYTES + 7 * 512]

        def guards():
            return dict(zip(c4.GUARD_KEYS, counts), input_guard_bytes_each_side=256)

        initial = guards()
        records, cache, samples = [], [], []
        for index, plan in enumerate(self.binding['steps']):
            label = plan['label']
            row = dict(step=index, label=label)
            if plan['action'] == 'reset':
                snapshot = dict(state={'all_zero': True}, raw_cache={'SHA256': '1'*64})
                row.update(before=snapshot, after=snapshot)
                records.append(row)
                continue
            counts[0] += 1
            counts[1] += 3
            counts[2] += sum(pin['bytes'] for pin in plan['inputs'].values()) + 3 * 512
            row.update(past_before=plan['past'], tokens=plan['tokens'], input_pins=plan['inputs'], prepare_input_audit=guards())
            if label in c4.OUTPUT_LABELS:
                width = plan['compressed_after']
                scores = [0.0] * width
                if score_changed and label == 'full_boundary':
                    scores[0] = 1.0
                indices = list(range(min(width, 512)))
                if ties and label == 'primary':
                    indices[0], indices[1] = indices[1], indices[0]
                row['outputs'] = {}
                for key, values, fmt, dtype in (('scores', scores, 'f', 10), ('indices', indices, 'q', 3)):
                    raw = struct.pack(f'<{len(values)}{fmt}', *values)
                    row['outputs'][key] = dict(metadata=dict(shape=[1, len(values)], dtype=dtype, bytes=len(raw), contiguous=True),
                        SHA256=self.write(root / f'{label}-{key}.bin', raw))
            if (label, plan['past']) in c4.SAMPLED:
                samples.append({key: plan[key] for key in ('label', 'past', 'tokens', 'variant')})
                added = plan['compressed_after'] - plan['past'] // 4
                cache.append(dict(step=index, report=dict(pending_slots=added, committed_slots=added if plan['commit'] else 0,
                        checked_cache_bytes=c4.CACHE_BYTES, after_SHA256='2'*64)))
            records.append(row)
        if truncated:
            records.pop()
        diagnostic = dict(performance_eligible=False, counts={key: self.binding[key] for key in c4.COUNT_KEYS},
                    final_past=self.binding['final_past'], records=records, cache_audit=cache, numeric_audit=samples,
                    input_audit=dict(postload=initial, final=guards()))
        outcome = dict(checkpoint=dict(bytes=c4.WEIGHT_BYTES, loaded=self.weight_keys), closed=close, diagnostic=diagnostic)
        progress = [dict(event='admitted', planned_states=71, planned_outputs=6, original_receipt_SHA256=range_pin or self.range_pin)]
        progress.extend(dict(event='step_complete', record=row) for row in records)
        progress.extend([dict(event='closed', driver_free_before_after_close=close[:2]), dict(event='weighted_complete', complete=True)])
        return dict(directory=name, Native_SHA256=c4.BASELINE_NATIVE_SHA if name == 'baseline' else '3'*64,
                    outcome_SHA256=self.write(root/'weighted-outcome.json', json.dumps(outcome).encode()),
                    progress_SHA256=self.write(root/'progress.jsonl', b'\n'.join(json.dumps(row).encode() for row in progress)))

    def compare(self, **candidate):
        arms = dict(baseline=self.arm('baseline'), candidate=self.arm('candidate', **candidate))
        data = json.dumps(dict(arms=arms, original_range_receipt_SHA256=self.range_pin)).encode()
        path = self.root / 'pair.json'
        return c4.compare(path, self.write(path, data))

    def test_all_pairs_and_topk_tie_with_equal_scores(self):
        result = self.compare(ties=True)
        self.assertTrue(result['exact_scores_all_samples'] and result['sample_results']['primary']['same_scores_topk_ties'])
        self.assertEqual(result['full_steps_per_arm'], 71)
        self.assertFalse(result['performance_eligible'] or result['operator_branch_execution_verified'])

    def test_score_mismatch_at_full_context_remains_visible(self):
        result = self.compare(score_changed=True)
        self.assertTrue(result['sample_results']['primary']['scores_byte_exact'])
        self.assertFalse(result['sample_results']['full_boundary']['scores_byte_exact'] or result['exact_scores_all_samples'])

    def test_short_diagnostic_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, 'step denominator'):
            self.compare(truncated=True)

    def test_weight_range_identity_must_agree(self):
        with self.assertRaisesRegex(ValueError, 'weighted input or completion'):
            self.compare(range_pin='4' * 64)


if __name__ == '__main__':
    unittest.main()
