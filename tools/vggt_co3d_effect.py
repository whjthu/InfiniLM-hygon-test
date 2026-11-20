"""Aggregate pinned VGGT CO3D pair-angle receipts for the three T12 checkpoints."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import re

from textvqa_effect import ARMS, read_exact, require, sha


CASE = 'T12.camera-head'
THRESHOLDS = (5, 15, 30)
UPSTREAM = '4b8be14b574b58c91ecd699122daf3d8004901d4'
POLICY = 'vggt-co3d-frozen-categories-and-explicit-zero-credit-failures-v1'


def scorer_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def planned_pairs(plan):
    require(plan.get('policy') == POLICY and plan.get('split') == 'test' and
            plan.get('pose_evaluator_commit') == UPSTREAM, 'unexpected CO3D evaluation protocol')
    sequences = plan['sequences']
    require(isinstance(sequences, list) and bool(sequences), 'no frozen CO3D sequences')
    pairs = {}
    seen = set()
    for sequence in sequences:
        category, name, frames = (sequence[key] for key in ('category', 'sequence', 'frame_ids'))
        require(isinstance(category, str) and re.fullmatch(r'[a-z0-9_]+', category) and
                isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_]+', name) and
                (category, name) not in seen, 'invalid or duplicate CO3D sequence')
        require(isinstance(frames, list) and len(frames) >= 2 and
                all(type(frame) is int and frame >= 0 for frame in frames) and
                len(set(frames)) == len(frames), 'invalid ordered CO3D frame inventory')
        seen.add((category, name))
        for first, second in combinations(frames, 2):
            pairs[f'{category}/{name}/{first}/{second}'] = category
    return pairs


def maximum_error(record):
    if record['status'] == 'failed':
        require(isinstance(record.get('reason'), str) and record['reason'].strip() and
                'rotation_deg' not in record and 'translation_deg' not in record,
                'failed CO3D pair requires a reason and no angles')
        return None
    require(record['status'] == 'scored', 'unexpected CO3D pair status')
    rotation, translation = (record[key] for key in ('rotation_deg', 'translation_deg'))
    require(type(rotation) in (int, float) and 0 <= rotation <= 180 and
            type(translation) in (int, float) and 0 <= translation <= 90,
            'invalid rotation or ambiguity-aware translation angle')
    return max(rotation, translation)


def credit(error, limit):
    """Return the number of cumulative histogram bins credited to one pair."""
    if error is None or error > limit:
        return 0
    # The final histogram bin includes its right edge, unlike earlier bins.
    return limit - min(math.floor(error), limit - 1)


def score_arm(raw, path, planned, binding):
    samples, categories = {}, {}
    lines = raw.splitlines(keepends=True)
    require(lines and json.loads(lines[0]) == binding, 'CO3D angle receipt header differs from the planned checkpoint')
    for line_number, line in enumerate(lines[1:], 2):
        record = json.loads(line)
        sample = record['sample_id']
        require(isinstance(sample, str) and sample in planned and sample not in samples,
                'duplicate or outside-plan CO3D pair')
        error = maximum_error(record)
        category = planned[sample]
        categories.setdefault(category, []).append(error)
        samples[sample] = dict(sample_id=sample, status=record['status'], category=category,
                               maximum_error_deg=error, artifact=dict(location=f'{path}#line={line_number}',
                               SHA256=sha256(line).hexdigest()))
    require(samples.keys() == planned.keys(), 'prediction or explicit failure missing for planned CO3D pair')
    ordered = [samples[sample] for sample in planned]
    category_auc = {category: {limit: sum(credit(error, limit) for error in errors) / (len(errors) * limit)
                               for limit in THRESHOLDS} for category, errors in sorted(categories.items())}
    return ordered, category_auc


def generate(root, plan_path, plan_sha, predictions_path, predictions_sha, output):
    root = root.resolve(strict=True)
    output = output.resolve()
    require(output.is_relative_to(root), 'CO3D evidence output must be within the ledger root')
    plan = json.loads(read_exact(plan_path, plan_sha))
    predictions = json.loads(read_exact(predictions_path, predictions_sha))
    identity = {key: plan[key] for key in ('case', 'framework_commit', 'dataset_SHA256', 'scorer_SHA256')}
    require(identity['case'] == CASE and sha(identity['dataset_SHA256']) and
            isinstance(identity['framework_commit'], str) and
            re.fullmatch(r'[a-f0-9]{40}', identity['framework_commit']) and
            identity['scorer_SHA256'] == scorer_sha256(), 'invalid CO3D case, dataset, framework or scorer identity')
    require(predictions['evaluation_plan_SHA256'] == plan_sha and predictions['arms'].keys() == set(ARMS),
            'prediction plan binding or checkpoint arms differ')
    planned = planned_pairs(plan)
    checkpoints, pins, samples, by_category = {}, {}, {}, {}
    for arm in ARMS:
        item = predictions['arms'][arm]
        checkpoints[arm], pins[arm] = item['checkpoint_SHA256'], item['SHA256']
        require(sha(checkpoints[arm]), 'missing pinned CO3D checkpoint')
        path = predictions_path.parent / Path(item['path'])
        binding = dict(arm=arm, checkpoint_SHA256=checkpoints[arm], evaluation_plan_SHA256=plan_sha,
                       pose_evaluator_commit=UPSTREAM)
        samples[arm], by_category[arm] = score_arm(read_exact(path, pins[arm]), str(path.resolve()), planned, binding)

    def write(name, value):
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        path = output / name
        with path.open('xb') as stream:
            stream.write(data)
        return dict(path=path.relative_to(root).as_posix(), SHA256=sha256(data).hexdigest())

    output.mkdir(mode=0o700)
    rows = {}
    for limit in THRESHOLDS:
        metric = f'AUC@{limit}'
        prefix = f'auc{limit}'
        provenance = dict(identity, metric=metric)
        sample_plan = write(f'{prefix}-plan.json', dict(provenance, sample_ids=list(planned),
                            evaluation_plan_SHA256=plan_sha))
        category_auc = {arm: {category: auc[limit] for category, auc in by_category[arm].items()} for arm in ARMS}
        inventory = write(f'{prefix}-inventory.json', dict(provenance, sample_plan_SHA256=sample_plan['SHA256'],
                          checkpoint_SHA256=checkpoints, arms=samples, category_auc=category_auc,
                          prediction_file_SHA256=pins, prediction_manifest_SHA256=predictions_sha,
                          policy=POLICY, pose_evaluator_commit=UPSTREAM))
        row = dict(planned_samples=len(planned), sample_plan=sample_plan, sample_inventory=inventory)
        for arm in ARMS:
            failed = sum(sample['status'] == 'failed' for sample in samples[arm])
            scores = category_auc[arm].values()
            row.update({arm: math.fsum(scores) / len(scores), f'{arm}_failed': failed,
                        f'{arm}_scored': len(planned) - failed})
        row['raw_scores'] = write(f'{prefix}-scores.json', dict(row, **provenance, checkpoint_SHA256=checkpoints))
        row['paired_checkpoint_binding'] = write(f'{prefix}-checkpoints.json', dict(provenance, **checkpoints))
        rows[metric] = row
    return write('co3d-metric-rows.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer-sha256', action='store_true', help='print the score implementation identity')
    for name in ('ledger-root', 'plan', 'predictions', 'output'):
        parser.add_argument('--' + name, type=Path)
    for name in ('plan-sha256', 'predictions-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.scorer_sha256:
        print(scorer_sha256())
        return
    if not all((args.ledger_root, args.plan, args.plan_sha256, args.predictions, args.predictions_sha256, args.output)):
        parser.error('scoring requires a ledger root, two exact input files and an output directory')
    print(json.dumps(generate(args.ledger_root, args.plan, args.plan_sha256,
                              args.predictions, args.predictions_sha256, args.output)))


if __name__ == '__main__':
    main()
