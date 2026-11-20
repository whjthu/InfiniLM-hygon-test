"""Score pinned private COCO captions with the COCO evaluation implementation."""
import argparse
import ast
from hashlib import file_digest, sha256
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile

from textvqa_effect import ARMS, read_exact, require, sha


METRIC = 'COCO CIDEr'
WHEEL_SHA256 = '083ed7910f1aec000b0a237ef6665f74edf19954204d0b1cbdb8399ed132228d'
POLICY = 'pycocoevalcap-1.2/PTB-3.4.1/full-frozen-IDF/failed-empty-v1'
SCORE_TOLERANCE = 1e-12
ORIGINAL_TRAINING_MANIFEST = (Path(__file__).resolve().parents[1] /
                              'code/bw1000/training-vlm-original-assets-v1/original-bindings.json')
ORIGINAL_TRAINING_SHA256 = '5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'


def original_training_images():
    manifest = json.loads(read_exact(ORIGINAL_TRAINING_MANIFEST, ORIGINAL_TRAINING_SHA256))
    common = manifest['common']
    images = common['image_identity']
    identities = {}
    for index, row in enumerate(images):
        match = re.fullmatch(r'coco-val2017/images/([0-9]{12})\.jpg', row['relative_role'])
        require(match is not None and type(row['ordered_index']) is int and row['ordered_index'] == index,
                'original COCO training image identity or order differs')
        number = int(match[1])
        require(number > 0 and str(number) not in identities and sha(row['sha256']),
                'invalid or duplicate original COCO training image')
        identities[str(number)] = row['sha256']
    require(len(identities) == len(images) == common['image_count'] == 64,
            'expected all 64 original VLM training image identities')
    return identities


def original_training_image_ids():
    return set(original_training_images())


def validate_holdout_image_ids(ids):
    require(isinstance(ids, list) and len(ids) >= 2 and all(isinstance(key, str) and re.fullmatch(r'[1-9][0-9]*', key)
            for key in ids) and len(set(ids)) == len(ids), 'invalid fixed COCO image set')
    overlap = len(original_training_image_ids().intersection(ids))
    require(not overlap, f'COCO holdout includes {overlap} original VLM training images')


def verify_image_bytes(records, ids):
    require(isinstance(records, list) and all(isinstance(row, dict) for row in records)
            and [row.get('image_id') for row in records] == ids,
            'COCO media inventory does not match the full ordered plan')
    require(all(sha(row.get('SHA256')) and type(row.get('bytes')) is int and row['bytes'] > 0 for row in records),
            'invalid COCO media byte evidence')
    originals = set(original_training_images().values())
    overlap = sum(row['SHA256'] in originals for row in records)
    require(not overlap, f'COCO holdout contains {overlap} byte-identical original VLM training images')
    return {row['image_id']: row['SHA256'] for row in records}


def hash_holdout_media(directory, image_files, ids):
    directory = directory.resolve(strict=True)
    records = []
    for key in ids:
        filename = image_files[key]
        require(isinstance(filename, str) and re.fullmatch(r'(?:COCO_[a-z0-9]+_)?[0-9]{12}\.jpg', filename)
                and int(filename[-16:-4]) == int(key), 'invalid publisher COCO image filename')
        path = (directory / filename).resolve(strict=True)
        require(path.is_relative_to(directory) and path.is_file(), 'expected a COCO image in the declared media root')
        with path.open('rb') as stream:
            digest = file_digest(stream, 'sha256').hexdigest()
            size = stream.tell()
        records.append(dict(image_id=key, SHA256=digest, bytes=size))
    return records


def scorer_binding(wheel, java):
    import numpy

    with wheel.open('rb') as stream:
        require(file_digest(stream, 'sha256').hexdigest() == WHEEL_SHA256, 'expected the pinned PyPI wheel')
    with java.open('rb') as stream:
        java_sha = file_digest(stream, 'sha256').hexdigest()
    version = subprocess.run([str(java), '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=True, timeout=15).stdout
    require(bool(version.strip()), 'Java version is unavailable')
    return dict(policy=POLICY, wheel_SHA256=WHEEL_SHA256,
                implementation_SHA256=sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=numpy.__version__,
                java_SHA256=java_sha, java_version=version.decode('utf-8', errors='replace').strip())


def annotations(raw, ids):
    document = json.loads(raw)
    planned = set(ids)
    images = [row['id'] for row in document['images']]
    require(all(type(key) is int and key > 0 for key in images) and len(set(images)) == len(images)
            and planned <= {str(key) for key in images}, 'invalid COCO image inventory or missing planned image')
    truth = {key: [] for key in ids}
    seen = set()
    for row in document['annotations']:
        key = str(row['image_id'])
        if key in truth:
            number, caption = row['id'], row['caption']
            require(type(row['image_id']) is int and type(number) is int and number not in seen
                    and isinstance(caption, str) and caption.strip(), 'invalid or duplicate COCO caption annotation')
            seen.add(number)
            truth[key].append(caption)
    require(all(truth.values()), 'a planned image has no official caption references')
    filenames = {str(row['id']): row['file_name'] for row in document['images'] if str(row['id']) in planned}
    return truth, filenames


def plan_media(plan, annotations_path, images_root):
    require(plan.get('metric') == METRIC, 'expected a COCO CIDEr plansheet')
    ids = plan['sample_ids']
    validate_holdout_image_ids(ids)
    truth, image_files = annotations(read_exact(annotations_path, plan['dataset_SHA256']), ids)
    image_records = hash_holdout_media(images_root, image_files, ids)
    return truth, image_records, verify_image_bytes(image_records, ids)


def predictions(raw, source, image_hashes):
    records = {}
    planned = image_hashes.keys()
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        row = json.loads(line)
        key = str(row['image_id'])
        require(type(row['image_id']) is int and key in planned and key not in records,
                'unexpected or duplicate COCO prediction image')
        status = row.get('status', 'scored')
        require(status in ('scored', 'failed'), 'unsupported prediction status')
        media_sha = image_hashes[key]
        require((status == 'failed' and 'image_SHA256' not in row) or row.get('image_SHA256') == media_sha,
                'prediction image bytes differ from frozen COCO media')
        field = 'caption' if status == 'scored' else 'reason'
        require(isinstance(row.get(field), str) and row[field].strip(), f'missing prediction {field}')
        caption = row['caption'] if status == 'scored' else ''
        records[key] = caption, dict(sample_id=key, status=status, image_SHA256=media_sha,
                                     artifact=dict(location=f'{source}#line={number}', SHA256=sha256(line).hexdigest()))
    require(records.keys() == planned, 'planned COCO image is missing a prediction or explicit failure')
    return records


def load_cider(directory):
    name = '_hygon_coco_cider_' + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, directory / 'cider.py',
                                                  submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module.Cider
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(name + '.cider_scorer', None)


def score_captions(wheel, java, truth, all_captions):
    with TemporaryDirectory(prefix='hygon-coco-cider-') as temporary:
        directory = Path(temporary)
        jar = directory / 'stanford-corenlp-3.4.1.jar'
        with wheel.open('rb') as stream:
            require(file_digest(stream, 'sha256').hexdigest() == WHEEL_SHA256, 'COCO scorer wheel changed')
            stream.seek(0)
            with ZipFile(stream) as archive:
                for source in ('cider.py', 'cider_scorer.py'):
                    (directory / source).write_bytes(archive.read('pycocoevalcap/cider/' + source))
                jar.write_bytes(archive.read('pycocoevalcap/tokenizer/' + jar.name))
                source = ast.parse(archive.read('pycocoevalcap/tokenizer/ptbtokenizer.py'))
        punctuation = set(next(ast.literal_eval(node.value) for node in source.body
                               if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                               and target.id == 'PUNCTUATIONS' for target in node.targets)))
        cider = load_cider(directory)
        input_file = directory / 'captions.txt'

        def tokenize(captions):
            if not captions:
                return []
            text = '\n'.join(caption.replace('\r', ' ').replace('\n', ' ') for caption in captions)
            input_file.write_bytes((text + '\n').encode('utf-8'))
            result = subprocess.run([str(java), '-Dfile.encoding=UTF-8', '-cp', str(jar),
                                     'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines', '-lowerCase',
                                     str(input_file)], capture_output=True, check=True, timeout=300)
            lines = result.stdout.decode('utf-8').split('\n')
            if lines[-1] == '':
                lines.pop()
            require(len(lines) == len(captions), 'Stanford tokenizer returned the wrong caption count')
            return [' '.join(word for word in line.rstrip().split(' ') if word not in punctuation) for line in lines]

        refs = iter(tokenize([caption for captions in truth.values() for caption in captions]))
        gold = {key: [next(refs) for _ in captions] for key, captions in truth.items()}
        result = {}
        for arm, captions in all_captions.items():
            accepted = iter(tokenize([captions[key] for key in truth if captions[key]]))
            hypotheses = {key: [next(accepted) if captions[key] else ''] for key in truth}
            mean, scores = cider().compute_score(gold, hypotheses)
            result[arm] = float(mean), list(map(float, scores))
        return result


def generate(args):
    root = args.ledger_root.resolve(strict=True)

    def reference(path, digest):
        return dict(path=path.resolve().relative_to(root).as_posix(), SHA256=digest)

    def write(name, value):
        path = args.output / name
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        with path.open('xb') as output:
            output.write(data)
        return reference(path, sha256(data).hexdigest())

    plan_ref = reference(args.plan, args.plan_sha256)
    checkpoint_ref = reference(args.checkpoints, args.checkpoints_sha256)
    require(args.output.resolve().is_relative_to(root), 'evidence output must be within the ledger root')
    plan = json.loads(read_exact(args.plan, args.plan_sha256))
    binding = json.loads(read_exact(args.checkpoints, args.checkpoints_sha256))
    manifest = json.loads(read_exact(args.predictions, args.predictions_sha256))
    identity = {key: plan[key] for key in ('case', 'framework_commit', 'dataset_SHA256', 'scorer_SHA256')}
    require(plan['metric'] == METRIC and re.fullmatch(r'T0[1-5]\.sft', identity['case'])
            and re.fullmatch(r'[a-f0-9]{40}', identity['framework_commit'])
            and all(binding.get(key) == value for key, value in identity.items())
            and all(sha(binding.get(arm)) for arm in ARMS), 'invalid COCO case or checkpoint binding')
    require(manifest['sample_plan_SHA256'] == plan_ref['SHA256'] and manifest['arms'].keys() == set(ARMS),
            'prediction plan or three-arm identity differs')
    ids = plan['sample_ids']
    truth, image_records, image_hashes = plan_media(plan, args.annotations, args.images_root)
    java = args.java.resolve(strict=True)
    scorer_ref = reference(args.scorer_binding, identity['scorer_SHA256'])
    runtime = scorer_binding(args.wheel, java)
    require(runtime == json.loads(read_exact(args.scorer_binding, identity['scorer_SHA256'])),
            'COCO scorer does not match the frozen binding')
    checkpoints = {arm: binding[arm] for arm in ARMS}
    arms, file_pins, captions = {}, {}, {}
    for arm in ARMS:
        item = manifest['arms'][arm]
        require(item['checkpoint_SHA256'] == checkpoints[arm], 'COCO prediction checkpoint differs')
        path = args.predictions.parent / item['path']
        records = predictions(read_exact(path, item['SHA256']), str(path.resolve()), image_hashes)
        file_pins[arm] = item['SHA256']
        captions[arm] = {key: records[key][0] for key in ids}
        arms[arm] = [records[key][1] for key in ids]
    scores = score_captions(args.wheel, java, truth, captions)
    row = dict(planned_samples=len(ids), sample_plan=plan_ref)
    for arm, (mean, values) in scores.items():
        require(len(values) == len(ids) and all(0 <= value <= 10 + SCORE_TOLERANCE for value in values)
                and math.isclose(mean, math.fsum(values) / len(ids), rel_tol=0, abs_tol=SCORE_TOLERANCE),
                'CIDEr scorer returned invalid per-image values or mean')
        failed = 0
        for value, sample in zip(values, arms[arm]):
            if sample['status'] == 'failed':
                require(value == 0, 'a failed caption received nonzero CIDEr')
                failed += 1
            sample['cider_score'] = value
        row.update({arm: mean, f'{arm}_failed': failed, f'{arm}_scored': len(ids) - failed})
    args.output.mkdir(mode=0o700)
    receipt = dict(identity, metric=METRIC, checkpoint_SHA256=checkpoints)
    media_ref = write('image-inventory.json', dict(receipt, sample_plan_SHA256=plan_ref['SHA256'],
                      original_training_SHA256=ORIGINAL_TRAINING_SHA256, images=image_records))
    row['sample_inventory'] = write('sample-inventory.json', dict(receipt, sample_plan_SHA256=plan_ref['SHA256'],
                                    cider_scoring=POLICY,
                                    scorer_binding=scorer_ref,
                                    image_inventory=media_ref,
                                    prediction_file_SHA256=file_pins, arms=arms))
    row['raw_scores'] = write('scores.json', dict(row, **receipt))
    row['paired_checkpoint_binding'] = checkpoint_ref
    return write('coco-cider-metric-row.json', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    describe = commands.add_parser('describe-scorer')
    media = commands.add_parser('verify-media')
    score = commands.add_parser('score')
    for command in (describe, score):
        command.add_argument('--wheel', type=Path, required=True)
        command.add_argument('--java', type=Path, required=True)
    for command in (media, score):
        for name in ('plan', 'annotations', 'images-root'):
            command.add_argument('--' + name, type=Path, required=True)
        command.add_argument('--plan-sha256', required=True)
    for name in ('ledger-root', 'checkpoints', 'predictions', 'scorer-binding', 'output'):
        score.add_argument('--' + name, type=Path, required=True)
    for name in ('checkpoints', 'predictions'):
        score.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'describe-scorer':
        result = scorer_binding(args.wheel, args.java.resolve(strict=True))
    elif args.command == 'verify-media':
        plan = json.loads(read_exact(args.plan, args.plan_sha256))
        _, image_records, _ = plan_media(plan, args.annotations, args.images_root)
        result = dict(metric=METRIC, dataset_SHA256=plan['dataset_SHA256'], sample_plan_SHA256=args.plan_sha256,
                      original_training_SHA256=ORIGINAL_TRAINING_SHA256, images=image_records)
    else:
        result = generate(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
