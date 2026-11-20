"""Produce formal TextVQA sample credits from pinned private prediction files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


ANNOTATIONS_SHA256 = '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'
MMF_SHA256 = 'e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a'
LICENSE_SHA256 = 'dbb40c7bb9a3de99f13fe5643fbdaa0ae90a1591c232f1b316e0528a835776c9'
SCORING = {'annotations_SHA256': ANNOTATIONS_SHA256, 'MMF_SHA256': MMF_SHA256,
           'score_unit': '1/30', 'failed_credit': 0}
ARMS = ('A100', 'Hygon', 'Hygon_before')
VALID_CREDITS = (0, 9, 18, 27, 30)
EMPTY_THINK_PREFIX = '<think>\n\n</think>\n\n'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def read_exact(path, digest):
    path = path.resolve(strict=True)
    require(path.is_file() and sha(digest), 'regular input with an exact SHA-256 required')
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, f'input SHA-256 differs: {path.name}')
    return raw


def official_evaluator(framework):
    runtime = framework / 'vlm/runtime'
    source = runtime / 'mmf_m4c_evaluators.py'
    read_exact(runtime / 'MMF-LICENSE.txt', LICENSE_SHA256)
    raw = read_exact(source, MMF_SHA256)
    scope = {}
    exec(compile(raw, str(source), 'exec'), scope)
    return scope['TextVQAAccuracyEvaluator']()


def annotations(raw):
    data = json.loads(raw)
    require(data.get('dataset_type') == 'val' and data.get('dataset_version') == '0.5.1' and
            isinstance(data.get('data'), list) and len(data['data']) == 5000,
            'expected the official 5,000-question TextVQA v0.5.1 validation annotations')
    by_id = {}
    for row in data['data']:
        question = str(row['question_id'])
        answers = row['answers']
        require(type(row['question_id']) is int and row['question_id'] >= 0 and question not in by_id and
                isinstance(answers, list) and len(answers) == 10 and all(isinstance(a, str) for a in answers),
                'invalid or duplicate official annotation')
        by_id[question] = answers
    return by_id


def score_answer(evaluator, answer, expected, policy):
    if policy == 'empty-think-prefix-v1':
        require(answer.startswith(EMPTY_THINK_PREFIX), 'missing the frozen MiniCPM empty thinking prefix')
        answer = answer[len(EMPTY_THINK_PREFIX):]
        require('<think>' not in answer and '</think>' not in answer, 'malformed MiniCPM response prefix')
    require(answer.strip(), 'empty response must be recorded as a failure')
    predicted = evaluator.answer_processor(answer)
    matches = sum(evaluator.answer_processor(gold) == predicted for gold in expected)
    return VALID_CREDITS[min(matches, 4)]


def score_arm(raw, source, ids, gold, evaluator, policy):
    planned = set(ids)
    rows = {}
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        record = json.loads(line)
        key = str(record['question_id'])
        require(type(record['question_id']) is int and key in planned and key not in rows,
                'outside-plan or duplicate prediction question')
        status = record.get('status', 'scored')
        require(status in ('scored', 'failed'), 'unsupported prediction status')
        if status == 'failed':
            require(isinstance(record.get('reason'), str) and record['reason'].strip(), 'missing failure reason')
            credit = 0
        else:
            answer = record.get('answer')
            require(isinstance(answer, str), 'missing original prediction string')
            credit = score_answer(evaluator, answer, gold[key], policy)
        rows[key] = {'sample_id': key, 'status': status, 'textvqa_credit_x30': credit,
                     'artifact': {'location': f'{source}#line={number}', 'SHA256': sha256(line).hexdigest()}}
    require(rows.keys() == planned, 'prediction or explicit failure missing for planned question')
    return [rows[key] for key in ids]


def manifest_payload(plan, plan_sha, checkpoints, prediction_rows, truth, evaluator):
    identity = {key: plan[key] for key in ('case', 'framework_commit', 'dataset_SHA256', 'scorer_SHA256')}
    require(identity['case'] in {f'T{i:02d}.sft' for i in range(1, 6)} and plan['metric'] == 'TextVQA' and
            re.fullmatch(r'[a-f0-9]{40}', identity['framework_commit']) and
            all(sha(identity[key]) for key in ('dataset_SHA256', 'scorer_SHA256')) and
            all(checkpoints[key] == value for key, value in identity.items()) and
            all(sha(checkpoints[arm]) for arm in ARMS), 'case, scorer, dataset or checkpoint identity differs')
    ids = plan['sample_ids']
    require(isinstance(ids, list) and bool(ids) and all(isinstance(key, str) and key in truth for key in ids) and
            len(set(ids)) == len(ids), 'invalid fixed official question set')
    policy = plan.get('answer_policy', 'raw')
    require(policy == 'raw' or (policy == 'empty-think-prefix-v1' and identity['case'] == 'T05.sft'),
            'unknown or incompatible frozen answer policy')
    require(prediction_rows.keys() == set(ARMS), 'all three checkpoint prediction files are required')
    checkpoint_ids = {arm: checkpoints[arm] for arm in ARMS}
    arms, file_sha = {}, {}
    scores = {'planned_samples': len(ids)}
    for arm, (source, raw, digest, checkpoint) in prediction_rows.items():
        require(checkpoint == checkpoint_ids[arm] and sha256(raw).hexdigest() == digest,
                'prediction bytes or checkpoint differ from their binding')
        samples = score_arm(raw, source, ids, truth, evaluator, policy)
        failed = sum(row['status'] == 'failed' for row in samples)
        scores.update({arm: sum(row['textvqa_credit_x30'] for row in samples) / (30 * len(ids)),
                       f'{arm}_failed': failed, f'{arm}_scored': len(ids) - failed})
        arms[arm], file_sha[arm] = samples, digest
    inventory = dict(identity, metric='TextVQA', sample_plan_SHA256=plan_sha, checkpoint_SHA256=checkpoint_ids,
                     textvqa_scoring=SCORING, prediction_file_SHA256=file_sha, arms=arms)
    return scores, inventory


def generate(args):
    root = args.ledger_root.resolve(strict=True)

    def reference(path, digest):
        return {'path': path.resolve().relative_to(root).as_posix(), 'SHA256': digest}

    def write(name, value):
        path = args.output / name
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        with path.open('xb') as output:
            output.write(data)
        return reference(path, sha256(data).hexdigest())

    require(args.output.resolve().is_relative_to(root), 'evidence output must be within the ledger root')
    plan_ref = reference(args.plan, args.plan_sha256)
    checkpoint_ref = reference(args.checkpoints, args.checkpoints_sha256)
    plan = json.loads(read_exact(args.plan, args.plan_sha256))
    checkpoints = json.loads(read_exact(args.checkpoints, args.checkpoints_sha256))
    manifest = json.loads(read_exact(args.predictions, args.predictions_sha256))
    require(manifest['sample_plan_SHA256'] == plan_ref['SHA256'] and manifest['arms'].keys() == set(ARMS),
            'prediction plan or three-arm identity differs')
    inputs = {}
    for arm, item in manifest['arms'].items():
        path = Path(item['path'])
        if not path.is_absolute():
            path = args.predictions.parent / path
        inputs[arm] = (str(path.resolve()), read_exact(path, item['SHA256']),
                       item['SHA256'], item['checkpoint_SHA256'])
    scores, inventory = manifest_payload(plan, args.plan_sha256, checkpoints, inputs,
                                         annotations(read_exact(args.annotations, ANNOTATIONS_SHA256)),
                                         official_evaluator(args.framework))
    args.output.mkdir(mode=0o700)
    row = dict(scores, sample_plan=plan_ref, sample_inventory=write('sample-inventory.json', inventory))
    provenance = {key: inventory[key] for key in ('case', 'metric', 'framework_commit', 'dataset_SHA256',
                                                  'scorer_SHA256', 'checkpoint_SHA256')}
    row['raw_scores'] = write('scores.json', dict(row, **provenance))
    row['paired_checkpoint_binding'] = checkpoint_ref
    return write('textvqa-metric-row.json', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ledger-root', 'framework', 'plan', 'checkpoints', 'predictions', 'annotations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('plan', 'checkpoints', 'predictions'):
        parser.add_argument('--' + name + '-sha256', required=True)
    print(json.dumps(generate(parser.parse_args()), ensure_ascii=False))
