"""Freeze a train-only 56/8 split of the original native InternVL GQA64 scope."""
import argparse
from hashlib import sha256
from io import BytesIO
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
ORIGINAL = ROOT / 'training-internvl-gqa-native-64-cpu-cfea7ada75b741449ce786c20974b61b'
FINGERPRINT_REQUEST = 'caac5aebc8e04fef9fd88f12ab69d952'
FINGERPRINT = ROOT / ('training-internvl-gqa-logit64-cpu-' + FINGERPRINT_REQUEST) / 'result.json'
EVIDENCE_PINS = {
    'training-lineage.json': (23649, '3df010f9f4ba6fb446ddd68ae05b5b002c1cef8f31db9feeb6dec411e8aee918'),
    'media-receipt.json': (22042, '8f692dab37c976bb819fd56c405721882b20258b008138430ae38c6f34d7b5b7'),
}
TRAIN_PIN = (19607, '78de19307549affa877be85756e7077bbb70d8d9feea09e1a8214157ee2fa000')
FINGERPRINT_PIN = (53728, '3b24a2b1f9f22408cc779129e335197de4eb513561369b297341608675a4e599')
ROSTER_SHA = '3bd9a38cbd6c1aae99a6365cdb8899bfe1c73e436359ee14da13cc55323163f1'
SPLIT_SEED = b'internvl-gqa64-train56-internal8-v1'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def unique_object(items):
    result = dict(items)
    need(len(items) == len(result), 'DUPLICATE_JSON_KEY')
    return result


def decode(raw):
    def reject_constant(_):
        raise ValueError('NONFINITE_JSON')
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_regular(path, cap, *, wanted=None):
    resolved = path.resolve(strict=True)
    need(path == resolved and stat.S_ISREG(path.lstat().st_mode), 'CANONICAL_REGULAR_INPUT_REQUIRED')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'BOUNDED_REGULAR_INPUT_REQUIRED')
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
        need(os.path.samestat(before, path.lstat()) and
             (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns) and len(raw) == before.st_size,
             'INPUT_CHANGED_DURING_READ')
    if wanted is not None:
        need((len(raw), digest(raw)) == wanted, 'EXACT_INPUT_PIN')
    return raw


def load_roster(source, fingerprint, train):
    code = read_regular(source, 12000)
    need(digest(code) == ROSTER_SHA, 'EXACT_NATIVE_ROSTER_SOURCE')
    scope = {'__name__': 'frozen_internvl_roster'}
    exec(compile(code, 'frozen_internvl_roster.py', 'exec'), scope)
    receipt = decode(fingerprint)
    need(receipt['accounting'][0] == '1103606.4370', 'EXACT_ORIGINAL_4370_ACCOUNTING')
    roster = scope['build_roster'](receipt, train, FINGERPRINT_REQUEST)
    need(len(roster) == 64, 'ORIGINAL_NATIVE_ROSTER_DENOMINATOR')
    return roster


def jpeg_identity(raw):
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        width, height = image.size
        need(image.format == 'JPEG' and 0 < width * height <= 25_000_000, 'BOUNDED_ORIGINAL_JPEG')
        image.load()
        pixels = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    return dhash, width / height


def near_duplicate(first, second):
    return abs(first[1] / second[1] - 1) <= 0.02 and (first[0] ^ second[0]).bit_count() <= 4


def split_records(train, lineage, media, *, target_data, read_media, identity=jpeg_identity, staged_images=None):
    lines = train.splitlines(keepends=True)
    need(len(lines) == len(lineage) == 64 and all(line.endswith(b'\n') for line in lines),
         'EXACT_64_SOURCE_ROWS')
    seen = {key: set() for key in ('image_id', 'question_id', 'source_group', 'SHA256', 'row_sha')}
    records = []
    for ordinal, (line, item) in enumerate(zip(lines, lineage, strict=True), 1):
        row = decode(line)
        need(set(row) == {'images', 'messages'} and isinstance(item, dict), 'EXACT_SOURCE_SCHEMA')
        name = item['path']
        need(isinstance(item['image_id'], str) and isinstance(item['question_id'], str) and
             bool(item['question_id']) and isinstance(name, str) and name == item['image_id'] + '.jpg'
             and re.fullmatch(r'[0-9]+\.jpg', name) and re.fullmatch(r'flickr:[0-9]+', item['source_group'])
             and re.fullmatch(r'[0-9a-f]{64}', item['SHA256']), 'EXACT_SOURCE_IDENTITIES')
        old = media[item['image_id']]
        need(set(old) == {'path', 'source_group', 'license', 'source_url', 'SHA256'}
             and all(item.get(key) == value for key, value in old.items()), 'LINEAGE_MEDIA_RECEIPT_DRIFT')
        need(row['images'] == [str(target_data / name)], 'LINEAGE_STAGED_ROW_IMAGE_DRIFT')
        if staged_images is not None:
            need(staged_images.get(name) == item['SHA256'], 'ORIGINAL_TARGET_BINDING_IMAGE_DRIFT')
        messages = row['messages']
        need(len(messages) == 2 and [m['role'] for m in messages] == ['user', 'assistant']
             and all(set(m) == {'role', 'content'} and isinstance(m['content'], str) and m['content'].strip()
                     for m in messages) and messages[0]['content'].startswith('<image>\n'), 'EXACT_SUPERVISION_ROLES')
        row_sha = digest(line[:-1])
        for key, values in seen.items():
            value = row_sha if key == 'row_sha' else item[key]
            need(value not in values, 'DUPLICATE_' + key.upper())
            values.add(value)
        raw = read_media(name)
        need(digest(raw) == item['SHA256'], 'EXACT_CURRENT_JPEG_SHA')
        records.append(dict(ordinal=ordinal, line=line, row_sha=row_sha, group=item['source_group'], photo=identity(raw)))
    if staged_images is not None:
        need(len(staged_images) == len(records), 'EXACT_ORIGINAL_TARGET_BINDING_DENOMINATOR')
    for a, b in itertools.combinations(records, 2):
        need(not near_duplicate(a['photo'], b['photo']), 'KNOWN_NEAR_DUPLICATE_WITHIN_GQA64')
    return records


def exposure_plan(records):
    holdout = {row['ordinal'] for row in sorted(records, key=lambda row:
        (sha256(SPLIT_SEED + b'\0internal\0' + row['group'].encode()).digest(), row['group']))[:8]}
    train = [row for row in records if row['ordinal'] not in holdout]
    internal = [row for row in records if row['ordinal'] in holdout]
    repeats = sorted(train[:32], key=lambda row:
        (sha256(SPLIT_SEED + b'\0repeat\0' + row['group'].encode()).digest(), row['group']))[:8]
    exposures = train + repeats
    slots = [dict(update=flat//32+1, rank=flat%8, accumulation=flat%32//8+1,
                  source_ordinal=row['ordinal'], row_SHA256=row['row_sha'], repeat=(flat>=56))
             for flat, row in enumerate(exposures)]
    need(len(train) == 56 and len(internal) == 8 and len(slots) == 64
         and len({slot['source_ordinal'] for slot in slots[:32]}) == 32
         and len({slot['source_ordinal'] for slot in slots[32:]}) == 32
         and all(slot['source_ordinal'] not in holdout for slot in slots), 'EXACT_PROPOSED_64_SLOT_PLAN')
    return train, internal, exposures, slots


def assemble(train_bytes, evidence, fingerprint, roster_source, media_reader, *, target_data, staged_images=None):
    load_roster(roster_source, fingerprint, train_bytes)
    records = split_records(train_bytes, evidence['training-lineage.json']['lineage'],
                            evidence['media-receipt.json']['admitted'], target_data=target_data,
                            read_media=media_reader, staged_images=staged_images)
    train, internal, exposures, slots = exposure_plan(records)
    files = {'train56.jsonl': b''.join(row['line'] for row in train),
             'internal8.jsonl': b''.join(row['line'] for row in internal),
             'proposed-exposures64.jsonl': b''.join(row['line'] for row in exposures),
             'proposed-slots.json': encoded(slots)}
    summary = dict(planned_original_rows=64, original_rows_verified=64, original_jpegs_verified=64,
                   disjoint_by_question_image_group_SHA=True, near_duplicate_pairs_checked=2016,
                   train_rows=56, internal_rows=8, planned_exposures=64, repeated_train_rows=8,
                   proposed_slots_per_update=[32,32], slots_per_rank_per_update=4,
                   scheduler_nominal_updates=25, original_expected_warmup_updates=1,
                   update1_requires_zero_LR=True, update2_requires_positive_LR_and_parameter_change=True,
                   prior_native_fingerprints_verified=64, split_seed=SPLIT_SEED.decode(),
                   original_native_processor_step='1103606.4369', original_native_fingerprint_step='1103606.4370',
                   untouched_previous_4370_native_source=True, val96_opened=False, predictions_opened=False,
                   model_or_adapter_loaded=False, optimizer_updates=0, actual_training_sampler_verified=False,
                   target_original_binding_verified=staged_images is not None,
                   training_ready=False, original_staged_train_SHA256=TRAIN_PIN[1],
                   original_fingerprint_SHA256=FINGERPRINT_PIN[1],
                   files={name: dict(bytes=len(raw), SHA256=digest(raw)) for name, raw in files.items()})
    return summary, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True, type=Path, help='Contains only the two frozen train-identity receipts')
    parser.add_argument('--roster-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New private output directory on the target')
    args = parser.parse_args()
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent and
         not os.path.lexists(args.output), 'FRESH_CANONICAL_PRIVATE_OUTPUT')
    data = ORIGINAL / 'data'
    train = read_regular(data/'train.jsonl', TRAIN_PIN[0], wanted=TRAIN_PIN)
    fingerprint = read_regular(FINGERPRINT, FINGERPRINT_PIN[0], wanted=FINGERPRINT_PIN)
    binding = decode(read_regular(ORIGINAL/'BINDING.json', 32768))
    need(binding['source_train_SHA256'] == '0ccbe304ad1d7e686bcfcf8b81bf17ef43bab4c4fcaeef5410fc7010bf170752'
         and binding['staged_train_SHA256'] == TRAIN_PIN[1], 'EXACT_TARGET_TRAIN_BINDING')
    evidence = {name: decode(read_regular(args.evidence_dir/name, pin[0], wanted=pin))
                for name, pin in EVIDENCE_PINS.items()}
    summary, files = assemble(train, evidence, fingerprint, args.roster_source,
                              lambda name: read_regular(data/name, 4*1024*1024),
                              target_data=PurePosixPath(data), staged_images=binding['images'])
    args.output.mkdir(mode=0o700)
    os.umask(0o077)
    for name, raw in [*files.items(), ('SPLIT.json', encoded(summary))]:
        with (args.output/name).open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    print(encoded(summary).decode(), end='')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        message = str(error)
        print(encoded(dict(passed=False, planned_original_rows=64, planned_train_rows=56,
                           planned_internal_rows=8, training_ready=False,
                           error_type=type(error).__name__,
                           error_code=message if re.fullmatch(r'[A-Z0-9_]{1,100}', message) else None)).decode(),
              end='', file=sys.stderr)
        raise SystemExit(2)
