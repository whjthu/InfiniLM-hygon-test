"""Check full-denominator TextVQA evidence using controlled private inputs."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import textvqa_effect as producer
from coco_cider_effect import (ORIGINAL_TRAINING_SHA256 as CIDER_TRAINING_SHA256, POLICY as CIDER_POLICY,
                               WHEEL_SHA256 as CIDER_WHEEL_SHA256)
from training_formal_metrics import effect, stage_three


class TextVQAEffectTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.identity = dict(case='T01.sft', framework_commit='1' * 40,
                             dataset_SHA256='2' * 64, scorer_SHA256='3' * 64)
        self.plan, self.plan_sha = self.store('plan.json', dict(self.identity, metric='TextVQA',
                                                               sample_ids=['11', '12', '13']))
        self.checkpoint_ids = {arm: digit * 64 for arm, digit in zip(producer.ARMS, '456')}
        self.checkpoint, self.checkpoint_sha = self.store('checkpoints.json', dict(self.identity, **self.checkpoint_ids))
        reference = [{'question_id': i, 'answers': ['yes', 'yes', 'yes'] + ['no'] * 7} for i in range(5000)]
        self.annotations, self.annotation_sha = self.store('annotations.json', dict(dataset_type='val',
                                                          dataset_version='0.5.1', data=reference))
        self.predictions = {}
        for arm in producer.ARMS:
            data = [{'question_id': 12, 'answer': 'yes'}, {'question_id': 11, 'answer': 'yes'},
                    {'question_id': 13, 'status': 'failed', 'reason': 'fixture device failure'}]
            path = self.root / f'{arm}.jsonl'
            path.write_bytes(('\n'.join(map(json.dumps, data)) + '\n').encode())
            self.predictions[arm] = dict(path=path.name, SHA256=sha256(path.read_bytes()).hexdigest(),
                                         checkpoint_SHA256=self.checkpoint_ids[arm])
        manifest, manifest_sha = self.store('predictions.json', dict(sample_plan_SHA256=self.plan_sha,
                                                                     arms=self.predictions))
        self.args = SimpleNamespace(ledger_root=self.root, output=self.root / 'proof', framework=self.root,
                                    plan=self.plan, plan_sha256=self.plan_sha, checkpoints=self.checkpoint,
                                    checkpoints_sha256=self.checkpoint_sha, predictions=manifest,
                                    predictions_sha256=manifest_sha, annotations=self.annotations)
        self.normalizer = SimpleNamespace(answer_processor=lambda value: value.lower().strip())

    def store(self, name, value):
        path = self.root / name
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return path, sha256(raw).hexdigest()

    def read(self, item):
        return json.loads(producer.read_exact(self.root / item['path'], item['SHA256']))

    def attach(self, name, value):
        path, digest = self.store(name, value)
        return {'path': path.name, 'SHA256': digest}

    def generate(self):
        with (patch.object(producer, 'ANNOTATIONS_SHA256', self.annotation_sha),
              patch.object(producer, 'official_evaluator', return_value=self.normalizer)):
            return self.read(producer.generate(self.args))

    def audit(self, row):
        return effect({'TextVQA': row}, {'TextVQA': 'higher'}, self.identity, self.read)

    def repin_score(self, row):
        metrics = {field: value for field, value in row.items()
                   if field not in ('raw_scores', 'paired_checkpoint_binding')}
        path, digest = self.store('new-score.json', dict(metrics, **self.identity, metric='TextVQA',
                                                        checkpoint_SHA256=self.checkpoint_ids))
        row['raw_scores'] = {'path': path.name, 'SHA256': digest}

    def test_generated_evidence_scores_failures_as_zero(self):
        row = self.generate()
        self.assertAlmostEqual(row['Hygon'], .6)
        self.assertEqual((row['planned_samples'], row['Hygon_scored'], row['Hygon_failed']), (3, 2, 1))
        self.assertEqual(self.audit(row)['TextVQA']['relative_degradation'], 0)
        inventory = self.read(row['sample_inventory'])
        self.assertEqual([s['sample_id'] for s in inventory['arms']['Hygon']], ['11', '12', '13'])
        self.assertEqual([s['textvqa_credit_x30'] for s in inventory['arms']['Hygon']], [27, 27, 0])

    def test_rejects_valid_only_accuracy_with_consistent_summary_pin(self):
        row = self.generate()
        row['Hygon'] = .9
        self.repin_score(row)
        with self.assertRaisesRegex(ValueError, 'full question denominator'):
            self.audit(row)

    def test_rejects_nonzero_credit_on_failed_row(self):
        row = self.generate()
        inventory = self.read(row['sample_inventory'])
        inventory['arms']['Hygon'][2]['textvqa_credit_x30'] = 9
        path, digest = self.store('tampered-inventory.json', inventory)
        row['sample_inventory'] = {'path': path.name, 'SHA256': digest}
        self.repin_score(row)
        with self.assertRaisesRegex(ValueError, 'per-question TextVQA credit'):
            self.audit(row)

    def test_does_not_silently_drop_missing_or_duplicate_question(self):
        truth = {'11': ['yes'] * 10, '12': ['no'] * 10}
        line = b'{"question_id":11,"answer":"yes"}\n'
        for raw, reason in ((line, 'missing for planned question'), (line + line, 'duplicate prediction')):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                producer.score_arm(raw, 'fixture.jsonl', list(truth), truth, self.normalizer, 'raw')

    def vlm_stage_three_fixture(self, *, bad_checkpoints=False):
        scorer = self.attach('coco-scorer.json', dict(policy=CIDER_POLICY, wheel_SHA256=CIDER_WHEEL_SHA256,
                             implementation_SHA256='1' * 64, java_SHA256='2' * 64,
                             java_version='fixture', numpy_version='fixture'))
        caption_binding = dict(dataset_SHA256='a' * 64, scorer_SHA256=scorer['SHA256'])
        case = {'case': self.identity['case'], 'original_cards': 8, 'launch_SHA256': '8' * 64}
        identity = {**self.identity, **caption_binding}
        checkpoints = dict(self.checkpoint_ids)
        if bad_checkpoints:
            checkpoints['Hygon'] = '7' * 64
        image_ids = ['54', '55']
        plan = self.attach('coco-plan.json', dict(identity, metric='COCO CIDEr', sample_ids=image_ids))
        media_records = [dict(image_id=key, bytes=100, SHA256=key[0] * 64) for key in image_ids]
        media = self.attach('coco-image-inventory.json', dict(identity, metric='COCO CIDEr',
                            sample_plan_SHA256=plan['SHA256'], checkpoint_SHA256=checkpoints,
                            original_training_SHA256=CIDER_TRAINING_SHA256, images=media_records))
        case_scores = dict(A100=1, Hygon=.97, Hygon_before=.88)
        sampled = {arm: [{'sample_id': item['image_id'], 'status': 'scored', 'cider_score': value,
                          'image_SHA256': item['SHA256'],
                          'artifact': {'location': 'private/caption', 'SHA256': '9' * 64}} for item in media_records]
                   for arm, value in case_scores.items()}
        inventory = self.attach('coco-inventory.json', dict(identity, metric='COCO CIDEr',
                                sample_plan_SHA256=plan['SHA256'], checkpoint_SHA256=checkpoints,
                                cider_scoring=CIDER_POLICY, scorer_binding=scorer, image_inventory=media,
                                prediction_file_SHA256={arm: '3' * 64 for arm in producer.ARMS}, arms=sampled))
        row = dict(case_scores, planned_samples=len(image_ids), sample_plan=plan, sample_inventory=inventory)
        for arm in producer.ARMS:
            row.update({f'{arm}_scored': len(image_ids), f'{arm}_failed': 0})
        row['raw_scores'] = self.attach('coco-scores.json', dict(row, **identity, metric='COCO CIDEr',
                                                   checkpoint_SHA256=checkpoints))
        row['paired_checkpoint_binding'] = self.attach('coco-checkpoints.json', dict(identity, **checkpoints))
        holdout = {'metric_bindings': {'COCO CIDEr': caption_binding,
                       'TextVQA': {key: self.identity[key] for key in ('dataset_SHA256', 'scorer_SHA256')}},
                   'metrics': {'TextVQA': self.generate(), 'COCO CIDEr': row}}
        training = dict(training_recipe_SHA256='a'*64,schedule_SHA256='b'*64,input_order_SHA256='c'*64)
        effect_training = {}
        for platform,commit,method,handle in (
                ('A100','9'*40,'launcher-process','a100-original-run'),
                ('Hygon',self.identity['framework_commit'],'slurm-step','1103606.888')):
            run = dict(training,case=case['case'],framework_commit=commit,platform=platform,
                      training_progress=dict(start_global_step=0,final_global_step=100,executed_optimizer_updates=100),
                      terminal=dict(method=method,handle=handle,state='COMPLETED',exit_code='0:0'),
                      final_checkpoint_SHA256=self.checkpoint_ids[platform])
            if platform=='A100':
                run['original_launch_SHA256']=case['launch_SHA256']
            else:
                run['start_checkpoint_SHA256']=self.checkpoint_ids['Hygon_before']
            effect_training[platform]=self.attach(platform+'-training.json',run)
        holdout['training_protocol']=self.attach('training-protocol.json',dict(
            training,case=case['case'],framework_commit=self.identity['framework_commit'],A100_framework_commit='9'*40,
            A100_original_launch_SHA256=case['launch_SHA256'],start_global_step=0,final_global_step=100,runs=effect_training))
        timing = {arm: [dict(run_id=arm + str(i), raw_receipt={'SHA256': str(i)*64}) for i in indices]
                  for arm, indices in (('A100', range(1, 4)), ('Hygon', range(4, 7)))}
        evidence = dict(case=case['case'], framework_commit=self.identity['framework_commit'], cards=8,
                        recipe_SHA256='1' * 64, boundary_SHA256='2' * 64,
                        A100_original_launch_SHA256=case['launch_SHA256'], A100_framework_commit='9' * 40,
                        timing=timing, holdout=holdout)
        return case, evidence

    def audit_vlm_stage_three(self, case, evidence):
        with patch('training_formal_metrics.timing', side_effect=[1.0, 0.91]):
            return stage_three(case, self.identity['framework_commit'], evidence, self.read)

    def test_stage_three_joins_independent_textvqa_and_coco_bindings(self):
        case, evidence = self.vlm_stage_three_fixture()
        result = self.audit_vlm_stage_three(case, evidence)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['effect_regressions']['COCO CIDEr']['relative_degradation'], .03)
        self.assertEqual(result['effect_regressions']['TextVQA']['relative_degradation'], 0)

    def test_stage_three_rejects_swapped_or_missing_metric_bindings(self):
        case, evidence = self.vlm_stage_three_fixture()
        holdout = evidence['holdout']
        bindings = holdout['metric_bindings']
        holdout['metric_bindings'] = {'COCO CIDEr': bindings['TextVQA'], 'TextVQA': bindings['COCO CIDEr']}
        with self.assertRaisesRegex(ValueError, 'different held-out evidence'):
            self.audit_vlm_stage_three(case, evidence)
        holdout['metric_bindings'] = {'TextVQA': bindings['TextVQA']}
        with self.assertRaisesRegex(ValueError, 'invalid per-metric'):
            self.audit_vlm_stage_three(case, evidence)
        del holdout['metric_bindings']
        holdout.update(bindings['TextVQA'])
        with self.assertRaisesRegex(ValueError, 'requires independent'):
            self.audit_vlm_stage_three(case, evidence)

    def test_stage_three_rejects_different_checkpoints_across_metrics(self):
        case, evidence = self.vlm_stage_three_fixture(bad_checkpoints=True)
        with self.assertRaisesRegex(ValueError, 'different paired checkpoints'):
            self.audit_vlm_stage_three(case, evidence)

    def test_stage_three_rejects_effect_from_a_different_training_window_or_checkpoint(self):
        case,evidence=self.vlm_stage_three_fixture()
        holdout=evidence['holdout']
        original=holdout['training_protocol']
        for changes,expected in (
                ({'training_progress':dict(start_global_step=0,final_global_step=99,executed_optimizer_updates=99)},'frozen update window'),
                ({'schedule_SHA256':'d'*64},'recipe or identity differs'),
                ({'final_checkpoint_SHA256':'e'*64},'different paired checkpoints'),
                ({'terminal':dict(method='slurm-step',handle='1103606.888',state='RUNNING',exit_code='0:0')},'successful terminal')):
            with self.subTest(expected=expected):
                protocol=self.read(original)
                run=self.read(protocol['runs']['Hygon'])
                run.update(changes)
                protocol['runs']['Hygon']=self.attach('changed-effect-training.json',run)
                holdout['training_protocol']=self.attach('changed-training-protocol.json',protocol)
                with self.assertRaisesRegex(ValueError,expected):
                    self.audit_vlm_stage_three(case,evidence)


if __name__ == '__main__':
    unittest.main()
