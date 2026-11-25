"""Exact answer-free material for the fixed three-arm pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

from effect_contract import ROOT, need, pin, read

TARGET = str(ROOT / 'inputs-only')
INPUT_MANIFEST_SHA = '831a01dc1c047d8446847e1e93d1d1a3fabba068064066e8b6d291a2b72e2434'
PLAN_SHA = '0993b5b287d5b0a90d450310d6b6e0105b6ebf99589cf6e14a0e04d1556770ef'
DATASET = dict(bytes=14864, SHA256='4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc')
INPUT = 'blind/inputs.jsonl'
MEDIA = {f'blind/media/{index:03d}.png' for index in range(18, 24)}
EVENTS = [f'E{index:02d}' for index in range(1, 7)]
KINDS = {'depth_change', 'maximum_depth', 'maximum_reported_number', 'minimum_depth',
         'minimum_reported_number', 'reported_number_change'}
FIELDS = {'id', 'chart_id', 'group', 'kind', 'event_ids', 'images', 'messages', 'labels'}


def exact_member(root, name, expected=None, cap=131072):
    target = root / name
    need(root.is_absolute() and target.resolve(strict=True) == target,
         'input member or ancestor is a link')
    return read(target, expected, cap)


def check_bundle(root=Path(TARGET)):
    raw = exact_member(root, 'INPUT_FILES.json')
    manifest_pin = pin(raw)
    need(manifest_pin['SHA256'] == INPUT_MANIFEST_SHA, 'exact input-only manifest differs')
    manifest = json.loads(raw)
    need(set(manifest['objects']) == MEDIA | {INPUT} and manifest['gold_files_opened'] == 0
         and manifest['questions'] == 36 and manifest['charts'] == 6 and manifest['event_groups'] == 2
         and manifest['protocol_SHA256'] == PLAN_SHA
         and manifest['dataset_manifest_SHA256'] == DATASET['SHA256'],
         'input-only manifest fields differ')
    need({p.relative_to(root).as_posix() for p in root.rglob('*')} ==
         MEDIA | {INPUT, 'INPUT_FILES.json', 'blind', 'blind/media'},
         'input-only directory has a missing or additional member')
    rows = [json.loads(line) for line in exact_member(root, INPUT, manifest['objects'][INPUT]).splitlines()]
    charts = {}
    counts = Counter()
    for row in rows:
        need(set(row) == FIELDS and row['event_ids'] == EVENTS and row['labels'] is None
             and len(row['images']) == 1 and row['images'][0] in MEDIA and row['kind'] in KINDS
             and len(row['messages']) == 1 and set(row['messages'][0]) == {'role', 'content'}
             and row['messages'][0]['role'] == 'user'
             and row['messages'][0]['content'].startswith('<image>\n'), 'unexpected answer-free row')
        chart = row['chart_id']
        identity = (row['group'], row['images'][0])
        need(charts.setdefault(chart, identity) == identity, 'chart group/media changed')
        counts[chart, row['kind']] += 1
    need(len(rows) == len({row['id'] for row in rows}) == 36
         and len(charts) == 6 and len({group for group, _ in charts.values()}) == 2
         and {image for _, image in charts.values()} == MEDIA
         and counts == Counter((chart, kind) for chart in charts for kind in KINDS),
         'original 36-row, six-chart, two-group denominator differs')
    for name in sorted(MEDIA):
        raw = exact_member(root, name, manifest['objects'][name])
        need(raw[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR'
             and struct.unpack('>II', raw[16:24]) == (512, 512), 'exact native input shape differs')
    return rows, manifest_pin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', type=Path, default=Path(TARGET))
    args = parser.parse_args()
    rows, manifest = check_bundle(args.input_directory)
    print(json.dumps(dict(rows=len(rows), charts=6, event_groups=2, gold_opened=0,
                          input_manifest=manifest), sort_keys=True))
