"""Project exactly the ordered TextVQA validation question text for inference."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from project_textvqa_val_identities import ANNOTATION_BYTES, ANNOTATION_SHA256


IDENTITIES_SHA256 = 'c3e42949ca5d35b655995e4991d8f73a12f12f8da7d1428881f1258d3e0b1886'


def pinned(path, digest, cap):
    with path.open('rb') as source:
        raw = source.read(cap + 1)
    if not 0 < len(raw) <= cap or sha256(raw).hexdigest() != digest:
        raise ValueError('Frozen TextVQA source bytes differ')
    return raw


def project(annotation, identity):
    def select_question_fields(record):
        if 'question_id' not in record:
            return record
        return {key: record[key] for key in ('question_id', 'image_id', 'question', 'set_name')}

    original = json.loads(pinned(annotation, ANNOTATION_SHA256, ANNOTATION_BYTES), object_hook=select_question_fields)
    roster = json.loads(pinned(identity, IDENTITIES_SHA256, 300_000))
    if (original['dataset_type'], original['dataset_version'], roster['annotation_SHA256'], roster['schema']) != (
            'val', '0.5.1', ANNOTATION_SHA256, 'TEXTVQA_V051_VAL_IDENTITY_ONLY_V1'):
        raise ValueError('Wrong validation annotation and identity pair')
    rows = []
    for original_row, expected in zip(original['data'], roster['rows'], strict=True):
        question = original_row['question']
        question_id = original_row['question_id']
        image_id = original_row['image_id']
        if ({'question_id': question_id, 'image_id': image_id} != expected or type(question_id) is not int
                or original_row['set_name'] != 'val' or type(question) is not str or not 0 < len(question) <= 4096
                or not question.strip() or '<image>' in question):
            raise ValueError('Wrong ordered question or prompt-unsafe source text')
        rows.append({'question_id': question_id, 'image_id': image_id, 'question': question})
    if len(rows) != 5000 or len({row['question_id'] for row in rows}) != 5000 or len({row['image_id'] for row in rows}) != 3166:
        raise ValueError('Wrong validation denominator')
    return dict(schema='TEXTVQA_V051_VAL_QUESTIONS_WITHOUT_ANSWERS_V1', annotation_SHA256=ANNOTATION_SHA256,
                identity_SHA256=IDENTITIES_SHA256, rows=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('annotation', type=Path)
    cli.add_argument('identity', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    projection = project(args.annotation, args.identity)
    raw = (json.dumps(projection, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(raw)
    print(json.dumps(dict(questions=len(projection['rows']), unique_images=3166, answer_fields=0,
                          SHA256=sha256(raw).hexdigest(), bytes=len(raw))))


if __name__ == '__main__':
    main()
