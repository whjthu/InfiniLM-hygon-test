from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import vggt_co3d_effect as effect
from vggt_co3d_pose_receipts import COORDINATES, checked_pose, generate, pose_errors


def pose(translation, degrees=0):
    cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.column_stack(([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], translation))


class PoseReceiptTest(unittest.TestCase):
    def test_relative_angles_and_translation_ambiguity(self):
        origin = pose([0, 0, 0])
        truth = pose([1, 0, 0])
        rotation, translation = pose_errors(truth, origin, pose([-2, 0, 0], 26), origin)
        self.assertAlmostEqual(rotation, 26)
        self.assertLess(translation, 1e-5)
        self.assertEqual(pose_errors(truth, origin, origin, origin)[1], 90)
        self.assertEqual(pose_errors(truth, origin, pose([0, 1, 0]), origin)[1], 90)

    def test_independent_world_gauges_leave_pair_errors_unchanged(self):
        first, second = pose([1, 2, 3], 17), pose([2, 3, 4], 95)
        def shift(value, gauge):
            return value @ np.vstack((gauge, [0, 0, 0, 1]))
        transform = pose([1, 7, -40], 45)
        rotation, translation = pose_errors(first, second, shift(first, transform), shift(second, transform))
        self.assertLess(rotation, 1e-10)
        self.assertLess(translation, 1e-5)

    def test_rejects_reflection_and_nonfinite_extrinsic(self):
        valid = pose([0, 0, 0])
        reflected = valid.copy()
        reflected[0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'not proper'):
            checked_pose(reflected.tolist())
        valid[0, 3] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            checked_pose(valid.tolist())

    def test_three_arms_flow_into_aggregator_and_failed_frame_counts(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            def write(name, obj):
                raw = json.dumps(obj).encode('utf-8')
                path = root / name
                path.write_bytes(raw)
                return path, sha256(raw).hexdigest()

            source = dict(category='apple', sequence='scene_1')
            plan = dict(case=effect.CASE, framework_commit='a' * 40, dataset_SHA256='b' * 64,
                        scorer_SHA256=effect.scorer_sha256(), split='test', pose_evaluator_commit=effect.UPSTREAM,
                        policy=effect.POLICY, sequences=[dict(source, frame_ids=[2, 9, 7])])
            plan_path, plan_sha = write('plan.json', plan)
            common = dict(evaluation_plan_SHA256=plan_sha, coordinates=COORDINATES)
            truth_frames = [dict(frame_id=i, extrinsic=pose([j, 0, 0]).tolist()) for j, i in enumerate([2, 9, 7])]
            truth_path, truth_sha = write('truth.json', dict(common, sequences=[dict(source, frames=truth_frames)]))
            manifest = dict(evaluation_plan_SHA256=plan_sha, arms={})
            for n, arm in enumerate(effect.ARMS, start=1):
                checkpoint = str(n) * 64
                frames = [dict(frame, status='scored') for frame in truth_frames]
                if arm == 'Hygon':
                    frames[-1] = dict(frame_id=7, status='failed', reason='forward error')
                receipt = dict(common, arm=arm, checkpoint_SHA256=checkpoint, pose_evaluator_commit=effect.UPSTREAM,
                               sequences=[dict(source, frames=frames)])
                path, digest = write(arm + '.json', receipt)
                manifest['arms'][arm] = dict(path=path.name, SHA256=digest, checkpoint_SHA256=checkpoint)
            manifest_path, manifest_sha = write('manifest.json', manifest)
            angles = generate(plan_path, plan_sha, truth_path, truth_sha, manifest_path, manifest_sha, root / 'angles')
            rows = effect.generate(root, plan_path, plan_sha, Path(angles['path']), angles['SHA256'], root / 'effects')
            metrics = json.loads((root / rows['path']).read_bytes())
            for row in metrics.values():
                self.assertEqual(row['planned_samples'], 3)
                self.assertEqual(row['A100'], 1)
                self.assertAlmostEqual(row['Hygon'], 1/3)
                self.assertEqual(row['Hygon_failed'], 2)


if __name__ == '__main__':
    unittest.main()
