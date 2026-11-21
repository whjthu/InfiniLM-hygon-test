import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from training_formal_metrics import effect
from vggt_co3d_effect import ARMS, CASE, POLICY, THRESHOLDS, UPSTREAM, credit, generate, planned_pairs, scorer_sha256


class Co3DEffectTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.identity = dict(case=CASE, framework_commit='1' * 40, dataset_SHA256='2' * 64,
                             scorer_SHA256=scorer_sha256())
        self.plan = dict(self.identity, policy=POLICY, split='test', pose_evaluator_commit=UPSTREAM,
                         sequences=[dict(category='apple', sequence='seq_1', frame_ids=[7, 3, 1]),
                                    dict(category='banana', sequence='seq_1', frame_ids=[12, 10])])
        self.pairs = list(planned_pairs(self.plan))
        self.plan_path, self.plan_sha = self.put('plan.json', self.plan)
        self.manifest = dict(evaluation_plan_SHA256=self.plan_sha, arms={})
        for index, arm in enumerate(ARMS, 3):
            checkpoint = str(index) * 64
            self.manifest['arms'][arm] = dict(checkpoint_SHA256=checkpoint)
            self.receipt(arm, [0, 5, 15, None] if arm == 'Hygon' else [0, 5, 15, 0])

    def put(self, name, obj):
        path = self.root / name
        raw = (json.dumps(obj) + '\n').encode('utf-8')
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def receipt(self, arm, angles, **header_overrides):
        binding = self.manifest['arms'][arm]
        header = dict(arm=arm, checkpoint_SHA256=binding['checkpoint_SHA256'],
                      evaluation_plan_SHA256=self.plan_sha, pose_evaluator_commit=UPSTREAM)
        header.update(header_overrides)
        records = [header]
        for pair, error in reversed(list(zip(self.pairs, angles))):
            row = dict(sample_id=pair)
            if error is None:
                row.update(status='failed', reason='missing frame output')
            else:
                row.update(status='scored', rotation_deg=0, translation_deg=error)
            records.append(row)
        raw = ''.join(json.dumps(row) + '\n' for row in records).encode('utf-8')
        path = self.root / f'{arm}.jsonl'
        path.write_bytes(raw)
        binding.update(path=path.name, SHA256=hashlib.sha256(raw).hexdigest())

    def run_scoring(self):
        path, digest = self.put('predictions.json', self.manifest)
        return generate(self.root, self.plan_path, self.plan_sha, path, digest, self.root / 'effects')

    def checked_read(self, ref):
        raw = (self.root / ref['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ref['SHA256'])
        return json.loads(raw)

    def test_complete_receipts_use_category_macro_and_pass_existing_reader(self):
        rows = self.checked_read(self.run_scoring())
        row = rows['AUC@5']
        self.assertEqual((row['Hygon_scored'], row['Hygon_failed'], row['planned_samples']), (3, 1, 4))
        self.assertAlmostEqual(row['A100'], 0.7)
        self.assertAlmostEqual(row['Hygon'], 0.2)
        for threshold in THRESHOLDS:
            self.assertAlmostEqual(rows[f'AUC@{threshold}']['A100'] - rows[f'AUC@{threshold}']['Hygon'], 0.5)
        inventory = self.checked_read(row['sample_inventory'])
        self.assertEqual([item['sample_id'] for item in inventory['arms']['Hygon']], self.pairs)
        self.assertEqual(inventory['category_auc']['Hygon']['banana'], 0)
        self.assertEqual(self.pairs[0], 'apple/seq_1/7/3')
        regressions = effect(rows, {key: 'higher' for key in rows}, self.identity, self.checked_read)
        self.assertEqual(set(regressions), set(rows))

    def test_right_endpoint_and_failed_pairs(self):
        for threshold in THRESHOLDS:
            self.assertEqual(credit(0, threshold), threshold)
            self.assertEqual(credit(math.nextafter(threshold, 0), threshold), 1)
            self.assertEqual(credit(threshold, threshold), 1)
            self.assertEqual(credit(math.nextafter(threshold, math.inf), threshold), 0)
            self.assertEqual(credit(None, threshold), 0)

    def replace_report(self, row, changed_inventory, **new_scores):
        inventory, sha = self.put('changed-inventory.json', changed_inventory)
        scores = self.checked_read(row['raw_scores'])
        replacement = dict(path=inventory.name, SHA256=sha)
        scores.update(new_scores, sample_inventory=replacement)
        path, sha = self.put('changed-scores.json', scores)
        row.update(new_scores, sample_inventory=replacement, raw_scores=dict(path=path.name, SHA256=sha))

    def test_forged_aggregate_is_rejected_even_with_matching_receipts(self):
        rows = self.checked_read(self.run_scoring())
        row = rows['AUC@5']
        inventory = self.checked_read(row['sample_inventory'])
        inventory['category_auc']['Hygon']['banana'] = 1
        self.replace_report(row, inventory, Hygon=0.7)
        with self.assertRaisesRegex(ValueError, 'AUC differs from the full per-category pair denominator'):
            effect(rows, {key: 'higher' for key in rows}, self.identity, self.checked_read)

    def test_thresholds_cannot_use_different_pair_errors_with_same_aggregate(self):
        rows = self.checked_read(self.run_scoring())
        row = rows['AUC@15']
        inventory = self.checked_read(row['sample_inventory'])
        sample = inventory['arms']['Hygon'][0]
        self.assertEqual(sample['maximum_error_deg'], 0.0)
        sample['maximum_error_deg'] = 0.25
        self.replace_report(row, inventory)
        with self.assertRaisesRegex(ValueError, 'AUC thresholds use different plans, predictions or pair errors'):
            effect(rows, {key: 'higher' for key in rows}, self.identity, self.checked_read)

    def test_missing_pair_does_not_disappear_from_denominator(self):
        self.receipt('Hygon', [0, 5, 15])
        with self.assertRaisesRegex(ValueError, 'explicit failure missing'):
            self.run_scoring()
        self.assertFalse((self.root / 'effects').exists())

    def test_checkpoint_swap_is_rejected(self):
        self.receipt('Hygon', [0, 5, 15, 0], checkpoint_SHA256='9' * 64)
        with self.assertRaisesRegex(ValueError, 'header differs'):
            self.run_scoring()


if __name__ == '__main__':
    unittest.main()
