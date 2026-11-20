"""COCO scoring integration and full-denominator acceptance without GPU or network."""
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import coco_cider_effect as producer
from training_formal_metrics import effect


class CiderEffectTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def put(self, name, content):
        path = self.root / name
        raw = content if isinstance(content, bytes) else json.dumps(content).encode()
        path.write_bytes(raw)
        return {'path': name, 'SHA256': sha256(raw).hexdigest()}

    def read(self, pin):
        return json.loads(producer.read_exact(self.root / pin['path'], pin['SHA256']))

    def fixture(self):
        files = {key: self.put(f'{key:012d}.jpg', f'fixture image {key}'.encode()) for key in (4, 6, 9)}
        annotations = self.put('annotations.json', {'images': [{'id': key, 'file_name': info['path']}
                                                              for key, info in files.items()],
                               'annotations': [{'id': 1, 'image_id': 4, 'caption': 'red ball'},
                                               {'id': 2, 'image_id': 6, 'caption': 'bicycle'},
                                               {'id': 3, 'image_id': 9, 'caption': 'phone'}]})
        runtime = dict(policy=producer.POLICY, wheel_SHA256=producer.WHEEL_SHA256,
                       implementation_SHA256='1'*64, java_SHA256='2'*64, java_version='fixture', numpy_version='fixture')
        scorer = self.put('scorer.json', runtime)
        identity = dict(case='T01.sft', framework_commit='f'*40, dataset_SHA256=annotations['SHA256'],
                        scorer_SHA256=scorer['SHA256'])
        plan = self.put('plan.json', dict(identity, metric=producer.METRIC, sample_ids=['9', '4', '6']))
        checkpoints = {arm: digit*64 for arm, digit in zip(producer.ARMS, '345')}
        binding = self.put('checkpoints.json', dict(identity, **checkpoints))
        predictions = [dict(image_id=4, caption='red ball', image_SHA256=files[4]['SHA256']),
                       dict(image_id=6, status='failed', reason='OOM'),
                       dict(image_id=9, caption='phone', image_SHA256=files[9]['SHA256'])]
        rows = b''.join(json.dumps(row).encode() + b'\n' for row in predictions)
        manifest = self.put('manifest.json', dict(sample_plan_SHA256=plan['SHA256'], arms={
            arm: dict(self.put(f'{arm}.jsonl', rows), checkpoint_SHA256=checkpoints[arm]) for arm in producer.ARMS}))
        java = self.root / 'java'
        java.touch()
        args = SimpleNamespace(ledger_root=self.root, output=self.root / 'proof', annotations=self.root / annotations['path'],
                               scorer_binding=self.root / scorer['path'], wheel=None, java=java, images_root=self.root,
                               plan=self.root / plan['path'], plan_sha256=plan['SHA256'],
                               checkpoints=self.root / binding['path'], checkpoints_sha256=binding['SHA256'],
                               predictions=self.root / manifest['path'], predictions_sha256=manifest['SHA256'])
        return args, identity, runtime

    def generated(self):
        args, identity, runtime = self.fixture()

        def score(wheel, java, truth, captions):
            self.assertEqual(list(truth), ['9', '4', '6'])
            self.assertEqual(truth['6'], ['bicycle'])
            self.assertEqual(captions['Hygon']['6'], '')
            return {arm: (1.0, [1.0, 2.0, 0.0]) for arm in producer.ARMS}

        with (patch.object(producer, 'scorer_binding', return_value=runtime),
              patch.object(producer, 'score_captions', side_effect=score)):
            row = self.read(producer.generate(args))
        return row, identity

    def verify(self, row, identity):
        return effect({producer.METRIC: row}, {producer.METRIC: 'higher'}, identity, self.read)

    def repin_score(self, row, identity):
        raw = self.read(row['raw_scores'])
        raw.update({key: value for key, value in row.items() if key not in ('raw_scores', 'paired_checkpoint_binding')})
        row['raw_scores'] = self.put('changed-scores.json', dict(raw, **identity))

    def test_generated_cider_row_keeps_complete_denominator(self):
        row, identity = self.generated()
        self.assertEqual((row['Hygon'], row['Hygon_scored'], row['Hygon_failed']), (1.0, 2, 1))
        self.assertEqual(self.verify(row, identity)[producer.METRIC]['relative_degradation'], 0.0)
        inventory = self.read(row['sample_inventory'])
        self.assertEqual([sample['sample_id'] for sample in inventory['arms']['Hygon']], ['9', '4', '6'])

    def test_preflight_cli_hashes_actual_files_and_preserves_order(self):
        args, _, _ = self.fixture()
        cli = ['coco_cider_effect.py', 'verify-media', '--plan', str(args.plan), '--plan-sha256',
               args.plan_sha256, '--annotations', str(args.annotations), '--images-root', str(args.images_root)]
        output = io.StringIO()
        with patch('sys.argv', cli), patch('sys.stdout', output):
            producer.main()
        media = json.loads(output.getvalue())
        self.assertEqual(media['sample_plan_SHA256'], args.plan_sha256)
        self.assertEqual([row['image_id'] for row in media['images']], ['9', '4', '6'])
        for row in media['images']:
            body = (args.images_root / f"{int(row['image_id']):012d}.jpg").read_bytes()
            self.assertEqual((row['SHA256'], row['bytes']), (sha256(body).hexdigest(), len(body)))

    def test_rejects_forged_aggregate_and_nonzero_failed_score(self):
        row, identity = self.generated()
        row['Hygon'] = 1.5
        self.repin_score(row, identity)
        with self.assertRaisesRegex(ValueError, 'full image denominator'):
            self.verify(row, identity)
        inventory = self.read(row['sample_inventory'])
        inventory['arms']['Hygon'][-1]['cider_score'] = 1.5
        row['sample_inventory'] = self.put('changed-inventory.json', inventory)
        self.repin_score(row, identity)
        with self.assertRaisesRegex(ValueError, 'invalid per-image CIDEr score'):
            self.verify(row, identity)

    def test_rejects_missing_duplicate_and_foreign_predictions(self):
        image_hashes = {'4': '1'*64, '6': '2'*64}
        raw = json.dumps(dict(image_id=4, caption='ball', image_SHA256=image_hashes['4'])).encode() + b'\n'
        for bad, reason in ((raw, 'missing a prediction'), (raw * 2, 'duplicate'),
                            (raw.replace(b'4', b'5'), 'unexpected')):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                producer.predictions(bad, 'fixture.jsonl', image_hashes)

    def test_rejects_the_same_training_bytes_renamed_with_a_new_image_id(self):
        row, identity = self.generated()
        inventory = self.read(row['sample_inventory'])
        media = self.read(inventory['image_inventory'])
        fixture_hash = media['images'][0]['SHA256']
        training = dict(producer.original_training_images())
        training[next(iter(training))] = fixture_hash
        with patch.object(producer, 'original_training_images', return_value=training):
            with self.assertRaisesRegex(ValueError, 'byte-identical original'):
                self.verify(row, identity)

    def test_rejects_caption_from_different_image_bytes(self):
        raw = json.dumps(dict(image_id=4, caption='ball', image_SHA256='2'*64)).encode() + b'\n'
        with self.assertRaisesRegex(ValueError, 'prediction image bytes differ'):
            producer.predictions(raw, 'fixture.jsonl', {'4': '1'*64})
        row, identity = self.generated()
        inventory = self.read(row['sample_inventory'])
        inventory['arms']['Hygon'][0]['image_SHA256'] = '7'*64
        row['sample_inventory'] = self.put('wrong-input-inventory.json', inventory)
        self.repin_score(row, identity)
        with self.assertRaisesRegex(ValueError, 'image differs from the frozen media'):
            self.verify(row, identity)

    def test_same_training_image_rejected_by_producer_and_formal_acceptance(self):
        original = producer.original_training_image_ids()
        self.assertEqual(len(original), 64)
        leaked = min(original, key=int)
        with self.assertRaisesRegex(ValueError, '1 original VLM training images'):
            producer.validate_holdout_image_ids(['9', leaked])
        row, identity = self.generated()
        plan = self.read(row['sample_plan'])
        plan['sample_ids'][-1] = leaked
        row['sample_plan'] = self.put('contaminated-plan.json', plan)
        inventory = self.read(row['sample_inventory'])
        inventory['sample_plan_SHA256'] = row['sample_plan']['SHA256']
        for samples in inventory['arms'].values():
            samples[-1]['sample_id'] = leaked
        row['sample_inventory'] = self.put('contaminated-inventory.json', inventory)
        self.repin_score(row, identity)
        with self.assertRaisesRegex(ValueError, '1 original VLM training images'):
            self.verify(row, identity)

    def test_calls_archive_cider_with_all_ids_and_validated_tokenizer_order(self):
        wheel = self.root / 'fixture.whl'
        cider = ('from .cider_scorer import compare\n'
                 'class Cider:\n    def compute_score(self, gold, hypotheses):\n'
                 '        assert list(gold) == list(hypotheses)\n'
                 '        values = [float(compare(hypotheses[k][0], gold[k][0])) for k in gold]\n'
                 '        return sum(values)/len(values), values\n')
        with ZipFile(wheel, 'w') as archive:
            archive.writestr('pycocoevalcap/cider/cider.py', cider)
            archive.writestr('pycocoevalcap/cider/cider_scorer.py', 'def compare(test, ref):\n    return test == ref\n')
            archive.writestr('pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar', b'fixture')
            archive.writestr('pycocoevalcap/tokenizer/ptbtokenizer.py', "PUNCTUATIONS = ['.']\n")

        def tokenize(command, **kwargs):
            self.assertIn('-preserveLines', command)
            self.assertTrue(kwargs['check'])
            inputs = Path(command[-1]).read_bytes()
            return SimpleNamespace(stdout=inputs.lower())

        gold = {'7': ['Red BALL .'], '4': ['Phone']}
        hypotheses = {'A100': {'7': '', '4': 'Phone'}, 'Hygon': {'7': 'Red\nball', '4': ''}}
        with (patch.object(producer, 'WHEEL_SHA256', sha256(wheel.read_bytes()).hexdigest()),
              patch.object(producer.subprocess, 'run', side_effect=tokenize)):
            result = producer.score_captions(wheel, Path('java'), gold, hypotheses)
        self.assertEqual(result, {'A100': (.5, [0.0, 1.0]), 'Hygon': (.5, [1.0, 0.0])})


if __name__ == '__main__':
    unittest.main()
