import copy
import hashlib
import json
import unittest

from training_formal_metrics import effect


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
