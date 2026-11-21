"""Combine independently verified training admissions for one dispatch cycle."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from report_training80_backlog import BACKLOG, REPO, file_bytes, invocation, need, pin_file, verify_launch
from tuning_queue import canonical


MUTABLE_FIELDS = {'state', 'launch', 'next_action'}
RECEIPTS = ('publication_receipt', 'prerequisite_receipt', 'original_request_receipt')


def frozen_row(row):
    return {key: value for key, value in row.items() if key not in MUTABLE_FIELDS}


def read_admission(repo, base, path):
    raw = file_bytes(repo, path, ('local-only/',))
    overlay = json.loads(raw)
    need(canonical({k: v for k, v in overlay.items() if k != 'items'}) ==
         canonical({k: v for k, v in base.items() if k != 'items'}), 'overlay changes backlog policy')
    rows = overlay['items']
    need([row['id'] for row in rows] == [row['id'] for row in base['items']], 'overlay changes inventory or order')
    changes = []
    for original, row in zip(base['items'], rows):
        if canonical(original) == canonical(row):
            continue
        need(original['kind'] == row['kind'] == 'training' and original['state'] == 'blocked'
             and 'launch' not in original and row['state'] == 'ready'
             and canonical(frozen_row(original)) == canonical(frozen_row(row)), 'overlay changes frozen candidate scope')
        need(isinstance(row.get('next_action'), str) and row['next_action'].strip(), 'overlay lacks next action')
        if 'attempt_receipts' in original:
            need(row['launch']['attempt_receipts'] == original['attempt_receipts'], 'overlay changes original intent paths')
        changes.append(row)
    need(changes, 'overlay contains no admissions')
    return changes, sha256(raw).hexdigest()


def check_admission(repo, row):
    state = verify_launch(repo, row)
    need(invocation(repo, row) is not None, 'launcher arguments are not bound')
    pins = [json.loads(pin_file(repo, row['launch'][name], ('local-only/',)))['target_preflight']
            for name in RECEIPTS]
    need(all(pin == pins[0] for pin in pins), 'admission receipts do not bind the same target preflight')
    pin_file(repo, pins[0], ('local-only/',))
    return state


def merge(repo, base, overlays):
    repo = repo.resolve()
    candidates = {}
    accepted = {}
    inputs = {}
    rejected = {}
    for path in overlays:
        try:
            changes, checksum = read_admission(repo, base, path)
            inputs[path] = checksum
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            rejected[path] = str(error)
            continue
        for row in changes:
            candidates.setdefault(row['id'], {})[canonical(row)] = row

    valid = {}
    for ident, variants in sorted(candidates.items()):
        try:
            need(len(variants) == 1, 'conflicting local admissions for the same candidate')
            [row] = variants.values()
            valid[ident] = (check_admission(repo, row), row)
        except (AttributeError, KeyError, OSError, SyntaxError, TypeError, ValueError) as error:
            rejected[ident] = str(error)

    request_counts = Counter()
    workload_counts = Counter()
    for row in [*base['items'], *(row for _, row in valid.values())]:
        if 'cpu_request' in row:
            request_counts[row['cpu_request']] += 1
        if 'launch' in row:
            request_counts[row['launch']['request']] += 1
            workload_counts[row['launch']['workload_SHA256']] += 1

    replacements = {}
    for ident, (state, row) in valid.items():
        launch = row['launch']
        if request_counts[launch['request']] > 1 or workload_counts[launch['workload_SHA256']] > 1:
            rejected[ident] = 'duplicate original request or frozen workload'
        else:
            replacements[ident] = row
            accepted[ident] = state
    return (dict(base, items=[replacements.get(row['id'], row) for row in base['items']]),
            dict(inputs=inputs, accepted=accepted, rejected=rejected, actual_slurm_submissions=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--overlay', action='append', required=True, help='repository-relative local-only backlog')
    args = parser.parse_args()
    output = args.output.resolve()
    private = (REPO / 'local-only').resolve()
    need(private == REPO / 'local-only' and output.is_relative_to(private) and output != private,
         'write only a new private backlog file')
    backlog, report = merge(REPO, json.loads(BACKLOG.read_bytes()), args.overlay)
    raw = (json.dumps(backlog, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(report, backlog=dict(path=output.relative_to(REPO).as_posix(),
                                               SHA256=sha256(raw).hexdigest())), ensure_ascii=False))


if __name__ == '__main__':
    main()
