import copy
import hashlib
import json
from pathlib import Path
import unittest

from training_formal_metrics import effect, verify_hunyuan_dp2_a100_windows


class Dp2OriginalWindowTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads((root / 'coordination/training/training-case-acceptance-matrix-v1.json').read_bytes())
        self.case = next(case for case in matrix['cases'] if case['case'] == 'T09.dp2')
        self.audit = json.loads((root / self.case['reference_evidence']).read_bytes())
        self.original = json.loads((root / self.audit['reference']['path']).read_bytes())
        results = self.audit['audit']['runs']
        self.baselines = [run for run in results.values() if run['start'] == 0]

    def verify(self):
        return verify_hunyuan_dp2_a100_windows(self.case, self.audit, lambda ref: self.original)

    def test_real_original_three_repeats(self):
        self.assertAlmostEqual(self.verify(), 0.19871607975371872)
        self.assertLess(self.baselines[1]['windows'][0]['seconds'], self.baselines[1]['windows'][1]['seconds'])

    def test_rejects_fast_rank_aggregate(self):
        first, second = self.baselines[1]['windows']
        self.assertLess(first['seconds'], second['seconds'])
        self.baselines[1]['measurement_updates_per_second'] = 20 / first['seconds']
        with self.assertRaisesRegex(ValueError, 'differs from the slowest rank'):
            self.verify()

    def test_rejects_checkpoints_outside_original_timing_boundary(self):
        self.baselines[0]['windows'][0]['boundary'] = 'excludes checkpoint IO'
        with self.assertRaisesRegex(ValueError, 'original boundary'):
            self.verify()

    def test_rejects_short_resume_as_reference_repeat(self):
        audited = self.audit['audit']['runs']
        del audited[self.baselines[1]['run']]
        resumed = next(run for run in audited.values() if run['start'] == 10)
        resumed['start'] = 0
        with self.assertRaisesRegex(ValueError, 'reference run identity'):
            self.verify()

    def test_rejects_copied_timing_summary(self):
        first, second, _ = self.baselines
        path = 'rank-0/summary.json'
        second['original_artifacts'][path]['sha256'] = first['original_artifacts'][path]['sha256']
        with self.assertRaisesRegex(ValueError, 'duplicated a timed summary'):
            self.verify()

    def test_rejects_unproven_matrix_median(self):
        self.case['a100_median_updates_per_second'] = 0.2
        with self.assertRaisesRegex(ValueError, 'matrix median differs'):
            self.verify()


class HoldoutInventoryTest(unittest.TestCase):
    def setUp(self):
        self.attachments = {}
        self.identity = {'case': 'T06.sft', 'dataset_SHA256': '5' * 64}
        self.checkpoints = {platform: digit * 64 for platform, digit in
                            zip(('A100', 'Hygon', 'Hygon_before'), '123')}

    def pin(self, value):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.attachments[digest] = value
        return {'SHA256': digest}

    def verify(self, ids, arms, expected_failure=1):
        metric = 'FID'
        plan = self.pin(dict(self.identity, metric=metric, sample_ids=ids))
        inventory = self.pin(dict(self.identity, metric=metric,
                                  sample_plan_SHA256=plan['SHA256'],
                                  checkpoint_SHA256=self.checkpoints, arms=arms))
        row = {'A100': 10, 'Hygon': 10.4, 'Hygon_before': 12, 'planned_samples': 2,
               'sample_plan': plan, 'sample_inventory': inventory}
        for platform in self.checkpoints:
            row[platform + '_scored'] = 2 - expected_failure
            row[platform + '_failed'] = expected_failure
        row['raw_scores'] = self.pin(dict(row, **self.identity, metric=metric,
                                         checkpoint_SHA256=self.checkpoints))
        row['paired_checkpoint_binding'] = self.pin(dict(self.identity, **self.checkpoints))
        return effect({metric: row}, {metric: 'lower'}, self.identity,
                      lambda item: self.attachments[item['SHA256']])

    def inventory(self):
        records = [
            {'sample_id': 'a', 'status': 'scored',
             'artifact': {'location': 'private/output', 'SHA256': 'a' * 64}},
            {'sample_id': 'b', 'status': 'failed',
             'artifact': {'location': 'private/failure', 'SHA256': 'b' * 64}},
        ]
        return {arm: copy.deepcopy(records) for arm in self.checkpoints}

    def test_accepts_same_ids_and_preserves_failed_sample(self):
        result = self.verify(['a', 'b'], self.inventory())
        self.assertAlmostEqual(result['FID']['relative_degradation'], .04)

    def test_rejects_same_count_with_foreign_id(self):
        arms = self.inventory()
        arms['Hygon'][1]['sample_id'] = 'c'
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            self.verify(['a', 'b'], arms)

    def test_rejects_duplicate_and_missing_ids(self):
        arms = self.inventory()
        arms['Hygon_before'][1]['sample_id'] = 'a'
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            self.verify(['a', 'b'], arms)

    def test_rejects_missing_failure_artifact(self):
        arms = self.inventory()
        arms['Hygon'][1]['artifact']['SHA256'] = ''
        with self.assertRaisesRegex(ValueError, 'output or failure evidence'):
            self.verify(['a', 'b'], arms)

    def test_rejects_aggregate_with_unreported_failure(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            self.verify(['a', 'b'], self.inventory(), expected_failure=0)


if __name__ == '__main__':
    unittest.main()
