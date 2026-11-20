import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import vggt_co3d_effect as effect
from vggt_co3d_export_poses import read_media_plan
import vggt_co3d_freeze_holdout as freeze
from vggt_co3d_pose_receipts import generate as make_angles


class Co3DHoldoutTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.media = self.root / 'images'
        self.annotations = self.root / 'annotations'
        self.annotations.mkdir()
        self.training_path = self.root / 'training.json'
        self.training_bytes = b'original train image bytes'
        self.training = dict(case=effect.CASE, training_input_evidence_SHA256='b'*64, frames=[
            dict(category='apple', sequence='old', image='apple/old/images/frame_0.jpg',
                 SHA256=sha256(self.training_bytes).hexdigest())])
        train_sha = self.put(self.training_path, self.training)
        training_sequence = self.frames('old', 4, write_media=False)
        test_sequence = self.frames('new', 4)
        small_sequence = self.frames('short', 2, write_media=False)
        annotation_path = self.annotations / 'apple_test.jgz'
        raw = gzip.compress(json.dumps(dict(old=training_sequence, new=test_sequence, short=small_sequence)).encode())
        annotation_path.write_bytes(raw)
        inventory = dict(coordinates=freeze.ANNOTATION_COORDINATES, training_frames_SHA256=train_sha,
                         sampling=dict(seed=0, frames_per_sequence=3, minimum_sequence_frames=4),
                         test_annotations=[dict(category='apple', path=annotation_path.name,
                                                SHA256=sha256(raw).hexdigest())])
        path = self.root / 'inventory.json'
        inventory_sha = self.put(path, inventory)
        self.args = SimpleNamespace(inventory=path, inventory_sha256=inventory_sha, training_frames=self.training_path,
                                    annotations_root=self.annotations, media_root=self.media,
                                    framework_commit='c'*40, output=self.root/'frozen')

    @staticmethod
    def put(path, value):
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def frames(self, scene, count, write_media=True):
        frames = []
        for frame in range(count):
            name = f'apple/{scene}/images/frame_{frame}.jpg'
            frames.append(dict(filepath=name, R=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], T=[frame, 5, 4]))
            if write_media:
                path = self.media / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'{scene} fixture {frame}'.encode())
        return frames

    def read(self, name):
        return json.loads((self.args.output / name).read_bytes())

    def test_freeze_is_deterministic_and_excludes_sequences_before_media_open(self):
        first = freeze.generate(self.args)
        plan = self.read('plan.json')
        truth = self.read('truth.json')
        self.assertEqual(plan['sequences'], [dict(category='apple', sequence='new', frame_ids=[2, 3, 1])])
        self.assertEqual(truth['sequences'][0]['frames'][0]['extrinsic'], [[0, -1, 0, -2], [1, 0, 0, -5], [0, 0, 1, 4]])
        record = self.read('freeze.json')
        self.assertEqual(record['image_pairs'], 3)
        self.assertEqual({row['reason'] for row in record['excluded_sequences']}, {'training_sequence', 'too_few_frames'})
        self.args.output = self.root / 'frozen-again'
        self.assertEqual(freeze.generate(self.args), first)

    def test_copy_of_training_image_under_another_sequence_cannot_enter_holdout(self):
        (self.media/'apple/new/images/frame_2.jpg').write_bytes(self.training_bytes)
        with self.assertRaisesRegex(ValueError, 'matches an actual training image'):
            freeze.generate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_training_inventory_must_match_the_frozen_input_hash(self):
        self.training['frames'][0]['SHA256'] = 'd'*64
        self.put(self.training_path, self.training)
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            freeze.generate(self.args)

    def test_frozen_bundle_is_accepted_by_existing_pose_and_effect_tools(self):
        freeze.generate(self.args)
        plan, media, truth, record = [self.read(name+'.json') for name in ('plan','media','truth','freeze')]
        self.assertEqual(read_media_plan(plan, media), media['sequences'])
        arms = {}
        for index, arm in enumerate(effect.ARMS, 1):
            checkpoint = str(index)*64
            frame_data = [dict(frame, status='scored') for frame in truth['sequences'][0]['frames']]
            pose = dict(truth, arm=arm, checkpoint_SHA256=checkpoint, pose_evaluator_commit=effect.UPSTREAM,
                        sequences=[dict(truth['sequences'][0], frames=frame_data)])
            path = self.root / (arm + '.json')
            arms[arm] = dict(path=path.name, SHA256=self.put(path, pose), checkpoint_SHA256=checkpoint)
        manifest = self.root / 'poses.json'
        manifest_sha = self.put(manifest, dict(evaluation_plan_SHA256=record['plan']['SHA256'], arms=arms))
        plan_path = self.args.output / 'plan.json'
        angles = make_angles(plan_path, record['plan']['SHA256'], self.args.output/'truth.json', record['truth']['SHA256'],
                             manifest, manifest_sha, self.root/'angles')
        scores = effect.generate(self.root, plan_path, record['plan']['SHA256'], Path(angles['path']),
                                 angles['SHA256'], self.root/'scores')
        for row in json.loads((self.root / scores['path']).read_bytes()).values():
            self.assertEqual(row['planned_samples'], 3)
            self.assertTrue(all(row[arm] == 1 and row[arm+'_failed'] == 0 for arm in effect.ARMS))


if __name__ == '__main__':
    unittest.main()
