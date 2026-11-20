"""Build CO3D pair-angle receipts from pinned OpenCV world-to-camera poses."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from vggt_co3d_effect import ARMS, CASE, UPSTREAM, planned_pairs, read_exact, require, sha


COORDINATES = 'opencv-world-to-camera-3x4'
TRANSLATION_EPSILON = 1e-15
ROTATION_TOLERANCE = 1e-4


def checked_pose(value):
    require(isinstance(value, list) and len(value) == 3 and all(isinstance(row, list) and len(row) == 4
            and all(type(element) in (int, float) for element in row) for row in value),
            'expected a numeric OpenCV 3x4 extrinsic')
    pose = np.array(value, dtype=np.float64)
    require(np.isfinite(pose).all(), 'nonfinite CO3D extrinsic')
    rotation = pose[:, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=ROTATION_TOLERANCE) and
            abs(np.linalg.det(rotation) - 1) <= ROTATION_TOLERANCE, 'extrinsic rotation is not proper')
    return pose


def relative_pose(first, second):
    rotation = first[:, :3] @ second[:, :3].T
    translation = first[:, 3] - rotation @ second[:, 3]
    require(np.isfinite(translation).all(), 'nonfinite relative translation')
    return rotation, translation


def pose_errors(first_truth, second_truth, first_prediction, second_prediction):
    truth_rotation, truth_translation = relative_pose(first_truth, second_truth)
    pred_rotation, pred_translation = relative_pose(first_prediction, second_prediction)
    delta = truth_rotation.T @ pred_rotation
    cosine = (np.trace(delta) - 1) / 2
    skew = (delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1])
    rotation_deg = math.degrees(math.atan2(math.hypot(*skew) / 2, cosine))

    truth_direction = truth_translation / (math.hypot(*truth_translation) + TRANSLATION_EPSILON)
    pred_direction = pred_translation / (math.hypot(*pred_translation) + TRANSLATION_EPSILON)
    translation_deg = math.degrees(math.acos(min(1.0, abs(truth_direction @ pred_direction))))
    return rotation_deg, translation_deg


def read_poses(raw, plan, plan_sha, *, arm=None, checkpoint=None):
    receipt = json.loads(raw)
    identity = dict(evaluation_plan_SHA256=plan_sha, coordinates=COORDINATES)
    if 'media_manifest_SHA256' in plan:
        require(sha(plan['media_manifest_SHA256']), 'invalid CO3D media identity')
        identity['media_manifest_SHA256'] = plan['media_manifest_SHA256']
    if arm is not None:
        identity.update(arm=arm, checkpoint_SHA256=checkpoint, pose_evaluator_commit=UPSTREAM)
    require(all(receipt.get(key) == value for key, value in identity.items()), 'pose receipt identity differs')
    expected = {(row['category'], row['sequence']): row['frame_ids'] for row in plan['sequences']}
    poses = {}
    for row in receipt['sequences']:
        key = (row['category'], row['sequence'])
        require(key in expected and key not in poses, 'outside-plan or duplicate CO3D pose sequence')
        frames = row['frames']
        require([frame['frame_id'] for frame in frames] == expected[key] and
                all(type(frame['frame_id']) is int for frame in frames), 'CO3D frame identities or order differ')
        poses[key] = {}
        for frame in frames:
            if arm is not None and frame['status'] == 'failed':
                reason = frame.get('reason')
                require(isinstance(reason, str) and reason.strip() and 'extrinsic' not in frame,
                        'failed pose requires a reason and no extrinsic')
                poses[key][frame['frame_id']] = reason
            else:
                require(arm is None or frame['status'] == 'scored', 'unexpected CO3D frame status')
                poses[key][frame['frame_id']] = checked_pose(frame['extrinsic'])
    require(poses.keys() == expected.keys(), 'frozen CO3D pose sequence missing')
    return poses


def pair_records(plan, truth, predictions):
    for sequence in plan['sequences']:
        key = (sequence['category'], sequence['sequence'])
        for first, second in combinations(sequence['frame_ids'], 2):
            pair = (predictions[key][first], predictions[key][second])
            row = dict(sample_id=f'{key[0]}/{key[1]}/{first}/{second}')
            failures = [f'frame {frame}: {pose}' for frame, pose in zip((first, second), pair) if isinstance(pose, str)]
            if failures:
                row.update(status='failed', reason=' | '.join(failures))
            else:
                rotation, translation = pose_errors(truth[key][first], truth[key][second], *pair)
                row.update(status='scored', rotation_deg=rotation, translation_deg=translation)
            yield row


def generate(plan_path, plan_sha, truth_path, truth_sha, poses_path, poses_sha, output):
    plan = json.loads(read_exact(plan_path, plan_sha))
    require(plan.get('case') == CASE, 'expected the VGGT CO3D case')
    planned_pairs(plan)
    truth = read_poses(read_exact(truth_path, truth_sha), plan, plan_sha)
    manifest = json.loads(read_exact(poses_path, poses_sha))
    require(manifest['evaluation_plan_SHA256'] == plan_sha and manifest['arms'].keys() == set(ARMS),
            'CO3D pose input manifest differs from the common plan')
    predictions = {}
    for arm in ARMS:
        item = manifest['arms'][arm]
        checkpoint = item['checkpoint_SHA256']
        require(sha(checkpoint), 'expected a pinned checkpoint identity')
        raw = read_exact(poses_path.parent / Path(item['path']), item['SHA256'])
        poses = read_poses(raw, plan, plan_sha, arm=arm, checkpoint=checkpoint)
        header = dict(arm=arm, checkpoint_SHA256=checkpoint, evaluation_plan_SHA256=plan_sha,
                      pose_evaluator_commit=UPSTREAM)
        rows = [header, *pair_records(plan, truth, poses)]
        predictions[arm] = (checkpoint, ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode())

    output.mkdir(mode=0o700)
    for arm, (checkpoint, raw) in predictions.items():
        filename = f'{arm}.jsonl'
        with (output / filename).open('xb') as stream:
            stream.write(raw)
        predictions[arm] = dict(path=filename, SHA256=sha256(raw).hexdigest(), checkpoint_SHA256=checkpoint)
    prediction_manifest = dict(evaluation_plan_SHA256=plan_sha, arms=predictions,
                               truth_pose_receipt_SHA256=truth_sha, model_pose_manifest_SHA256=poses_sha,
                               converter_SHA256=sha256(Path(__file__).read_bytes()).hexdigest())
    path = output / 'pair-predictions.json'
    raw = (json.dumps(prediction_manifest, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), SHA256=sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'truth', 'poses', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('plan', 'truth', 'poses'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.plan, args.plan_sha256, args.truth, args.truth_sha256,
                              args.poses, args.poses_sha256, args.output)))


if __name__ == '__main__':
    main()
