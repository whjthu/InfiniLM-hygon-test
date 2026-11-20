"""Freeze in-place CO3D test media and camera truth before evaluating VGGT checkpoints."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from vggt_co3d_effect import CASE, POLICY, UPSTREAM, planned_pairs, read_exact, require, scorer_sha256, sha
from vggt_co3d_export_poses import resolved_member, write_receipt
from vggt_co3d_pose_receipts import COORDINATES, checked_pose


ANNOTATION_COORDINATES = 'pytorch3d-world-to-view'
DEFAULT_SAMPLING = dict(seed=0, frames_per_sequence=10, minimum_sequence_frames=50)


def frame_path(category, sequence, name):
    require(isinstance(category, str) and re.fullmatch(r'[a-z0-9_]+', category) and
            isinstance(sequence, str) and re.fullmatch(r'[A-Za-z0-9_]+', sequence) and isinstance(name, str)
            and '\\' not in name, 'invalid CO3D media identity')
    path = PurePosixPath(name)
    require(len(path.parts) == 4 and path.parts[:3] == (category, sequence, 'images') and path.as_posix() == name,
            'CO3D image path differs from its sequence identity')
    require(path.suffix.lower() in ('.jpg', '.jpeg', '.png') and '..' not in path.parts,
            'expected a CO3D image path')
    return name


def training_identity(raw):
    training = json.loads(raw)
    require(training.get('case') == CASE and sha(training.get('training_input_evidence_SHA256')),
            'missing actual T12 training input identity')
    records = training['frames']
    require(isinstance(records, list) and bool(records), 'missing actual training frame inventory')
    paths, digests, sequences = set(), set(), set()
    for frame in records:
        category, sequence = frame['category'], frame['sequence']
        name = frame_path(category, sequence, frame['image'])
        require(sha(frame['SHA256']) and name not in paths, 'invalid or duplicate training frame')
        paths.add(name)
        digests.add(frame['SHA256'])
        sequences.add((category, sequence))
    return paths, digests, sequences


def camera_truth(frame):
    rotation, translation = frame['R'], frame['T']
    require(isinstance(rotation, list) and len(rotation) == 3 and all(isinstance(row, list) and len(row) == 3
            and all(type(value) in (int, float) for value in row) for row in rotation) and
            isinstance(translation, list) and len(translation) == 3 and
            all(type(value) in (int, float) for value in translation), 'invalid PyTorch3D camera annotation')
    pose = np.column_stack((np.asarray(rotation).T, translation))
    pose[:2] *= -1
    result = pose.tolist()
    checked_pose(result)
    return result


def freeze_inputs(inventory, inventory_sha, training, annotations_root, media_root, framework):
    require(inventory.get('coordinates') == ANNOTATION_COORDINATES and
            re.fullmatch(r'[a-f0-9]{40}', framework), 'expected a pinned frame annotation convention and framework')
    sampling = inventory.get('sampling', DEFAULT_SAMPLING)
    count, minimum, seed = (sampling[key] for key in ('frames_per_sequence', 'minimum_sequence_frames', 'seed'))
    require(all(type(value) is int for value in (count, minimum, seed)) and
            2 <= count <= minimum and 0 <= seed < 2**32, 'invalid CO3D frame sampling')
    train_paths, train_digests, train_sequences = training
    rng = np.random.RandomState(seed)
    annotations = inventory['test_annotations']
    categories = [row['category'] for row in annotations]
    require(categories and all(isinstance(category, str) and re.fullmatch(r'[a-z0-9_]+', category)
            for category in categories) and categories == sorted(set(categories)), 'require a sorted, unique category set')

    exclusions = Counter()
    plan_sequences, media_sequences, truth_sequences = [], [], []
    for category_record in annotations:
        category = category_record['category']
        path = resolved_member(annotations_root, category_record['path'])
        require(path.name == f'{category}_test.jgz', 'expected the publisher test annotation filename')
        data = json.loads(gzip.decompress(read_exact(path, category_record['SHA256'])))
        require(isinstance(data, dict) and bool(data), 'missing test sequence annotations')
        for sequence, frames in sorted(data.items()):
            require(isinstance(frames, list), 'invalid test frame inventory')
            if (category, sequence) in train_sequences:
                exclusions[category, 'training_sequence'] += 1
                continue
            if len(frames) < minimum:
                exclusions[category, 'too_few_frames'] += 1
                continue
            names = [frame_path(category, sequence, row['filepath']) for row in frames]
            require(len(set(names)) == len(names) and not train_paths.intersection(names),
                    'duplicate or training frame in considered test sequence')
            poses = [camera_truth(row) for row in frames]
            if any(sum(row['T']) > 1e5 for row in frames):
                exclusions[category, 'translation_outside_publisher_filter'] += 1
                continue
            frame_ids = rng.choice(len(frames), count, replace=False).tolist()
            image_rows, truth_rows = [], []
            for frame_id in frame_ids:
                name = names[frame_id]
                image = resolved_member(media_root, name)
                raw = image.read_bytes()
                require(bool(raw), 'empty CO3D image')
                digest = sha256(raw).hexdigest()
                require(digest not in train_digests, 'test image matches an actual training image byte-for-byte')
                image_rows.append(dict(frame_id=frame_id, image=name, SHA256=digest))
                truth_rows.append(dict(frame_id=frame_id, extrinsic=poses[frame_id]))
            identity = dict(category=category, sequence=sequence)
            plan_sequences.append(dict(identity, frame_ids=frame_ids))
            media_sequences.append(dict(identity, frames=image_rows))
            truth_sequences.append(dict(identity, frames=truth_rows))
        require(any(row['category'] == category for row in plan_sequences), 'a declared category has no eligible sequence')
    plan = dict(case=CASE, split='test', dataset_SHA256=inventory_sha, policy=POLICY, pose_evaluator_commit=UPSTREAM,
                framework_commit=framework, scorer_SHA256=scorer_sha256(), sampling=sampling, sequences=plan_sequences)
    media = dict(dataset_SHA256=inventory_sha, sequences=media_sequences)
    truth = dict(coordinates=COORDINATES, sequences=truth_sequences)
    excluded = [dict(category=category, reason=reason, sequences=count) for (category, reason), count in sorted(exclusions.items())]
    return plan, media, truth, excluded


def generate(args):
    inventory = json.loads(read_exact(args.inventory, args.inventory_sha256))
    training_sha = inventory['training_frames_SHA256']
    training = training_identity(read_exact(args.training_frames, training_sha))
    plan, media, truth, excluded = freeze_inputs(inventory, args.inventory_sha256, training,
        args.annotations_root.resolve(strict=True), args.media_root.resolve(strict=True), args.framework_commit)
    pair_count = len(planned_pairs(plan))
    args.output.mkdir(mode=0o700)
    media_ref = write_receipt(args.output, 'media.json', media)
    plan['media_manifest_SHA256'] = media_ref['SHA256']
    plan_ref = write_receipt(args.output, 'plan.json', plan)
    truth.update(evaluation_plan_SHA256=plan_ref['SHA256'], media_manifest_SHA256=media_ref['SHA256'])
    truth_ref = write_receipt(args.output, 'truth.json', truth)
    return write_receipt(args.output, 'freeze.json', dict(plan=plan_ref, media=media_ref, truth=truth_ref,
        training_frames_SHA256=training_sha, checked_training_frames=len(training[0]), eligible_sequences=len(plan['sequences']),
        image_pairs=pair_count, excluded_sequences=excluded, cross_split_check='sequence, path and exact image bytes'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'training-frames', 'annotations-root', 'media-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--framework-commit', required=True)
    print(json.dumps(generate(parser.parse_args())))


if __name__ == '__main__':
    main()
