from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import vggt_co3d_export_poses as exporter
from vggt_co3d_effect import CASE, POLICY, UPSTREAM
from vggt_co3d_pose_receipts import read_poses


class FixtureRunner:
    runtime = {'fixture': True}

    def __init__(self, *_):
        self.head_installed = False

    def install_head(self, _filename, _digest):
        self.head_installed = True

    def predict(self, images):
        assert [image.read() for image in images] == [b'frame seven', b'frame three']
        good = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        bad = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        return [good, bad if self.head_installed else good]


class Co3DExportTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.scene = dict(category='apple', sequence='scene_12')
        frames = []
        for frame_id, data in ((7, b'frame seven'), (3, b'frame three')):
            path = self.root / f'{frame_id}.jpg'
            path.write_bytes(data)
            frames.append(dict(frame_id=frame_id, image=path.name, SHA256=sha256(data).hexdigest()))
        self.media = dict(dataset_SHA256='a'*64, sequences=[dict(self.scene, frames=frames)])
        self.plan = dict(case=CASE, dataset_SHA256='a'*64, split='test', pose_evaluator_commit=UPSTREAM,
                         policy=POLICY, sequences=[dict(self.scene, frame_ids=[7, 3])])

    def write(self, filename, obj):
        path = self.root / filename
        raw = json.dumps(obj).encode()
        path.write_bytes(raw)
        return path, sha256(raw).hexdigest()

    def test_export_two_hygon_arms_through_one_model_and_validate_receipt(self):
        media, self.plan['media_manifest_SHA256'] = self.write('media.json', self.media)
        plan, plan_sha = self.write('plan.json', self.plan)
        args = SimpleNamespace(plan=plan, plan_sha256=plan_sha, media=media, media_root=self.root,
                               source_root=self.root, source_manifest=self.root/'source.json',
                               base_weights=self.root/'base.pt', head=self.root/'head.pt', head_sha256='b'*64,
                               platform='Hygon', output=self.root/'out')
        with patch.object(exporter, 'verify_source'), patch.object(exporter, 'NativeVGGT', side_effect=FixtureRunner) as runner:
            ref = exporter.generate(args)
        self.assertEqual(runner.call_count, 1)
        manifest = json.loads((args.output / ref['path']).read_bytes())
        for arm, artifact in manifest['arms'].items():
            raw = (args.output / artifact['path']).read_bytes()
            self.assertEqual(artifact['SHA256'], sha256(raw).hexdigest())
            poses = read_poses(raw, self.plan, plan_sha, arm=arm, checkpoint=artifact['checkpoint_SHA256'])
            self.assertEqual(isinstance(poses[('apple', 'scene_12')][3], str), arm == 'Hygon')
        unbound = dict(self.plan, media_manifest_SHA256='c'*64)
        with self.assertRaisesRegex(ValueError, 'receipt identity differs'):
            read_poses(raw, unbound, plan_sha, arm=arm, checkpoint=artifact['checkpoint_SHA256'])

    def test_changed_frame_order_is_rejected(self):
        self.media['sequences'][0]['frames'].reverse()
        with self.assertRaisesRegex(ValueError, 'frame order differs'):
            exporter.read_media_plan(self.plan, self.media)

    def test_mutated_media_fails_before_prediction(self):
        (self.root / '7.jpg').write_bytes(b'substituted')
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            list(exporter.export_sequences(self.media['sequences'], self.root, None))

    def test_media_path_cannot_escape_root(self):
        for path in ('../data.jpg', 'C:/data.jpg', '/data.jpg'):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                exporter.resolved_member(self.root, path)


if __name__ == '__main__':
    unittest.main()
