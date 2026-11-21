"""Independently verify the one original Gemma input-audit CPU step and its artifacts."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import signal
import subprocess

from build_gemma_train_exposure import TRAIN_FILE, TRAIN_SHA256
from control_gemma_render72_native_cpu import (
    COLUMNS, DATA, LABEL, MEMBERSHIP, NODE, PARENT, PINS, PLANNED, REQUEST, ROSTER,
    RUN, TERM_HELPER, terminal_pass,
)
from read_training_slurm_membership import allocated_dcu


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def hex64(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def read_file(base, name, limit):
    path = base/name
    need(base.is_absolute() and path.resolve(strict=True) == path and path.is_file(), 'unexpected original file: '+str(name))
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit, 'original file exceeds receipt bound: '+str(name))
    return raw


def phase_row(raw):
    need(0 < len(raw) <= 2*1024**2, 'missing or oversized original Slurm accounting')
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    need(all(len(row) == len(COLUMNS) for row in rows) and sum(row[0] == PARENT for row in rows) == 1,
         'independent accounting does not cover the original parent and columns')
    hits = [row for row in rows if row[1] == LABEL]
    need(len(hits) <= 1, 'original CPU label appears more than once')
    if not hits:
        return None
    [row] = hits
    need(re.fullmatch(PARENT+r'\.\d+', row[0]) and row[5] == NODE and row[2].split(),
         'original CPU step or location differs')
    return row


def assert_fields(doc, expected, message):
    need(type(doc) is dict and all(type(doc.get(key)) is type(value) and doc[key] == value
                                  for key, value in expected.items()), message)


def inventories(records_raw, training_raw, training_sha=TRAIN_SHA256):
    need(sha256(training_raw).hexdigest() == training_sha, 'original 72-row training document differs')
    original = training_raw.splitlines(keepends=True)
    if not all(row.endswith(b'\n') for row in original):
        raise ValueError('original row boundary differs')
    rows = records_raw.splitlines(keepends=True)
    need(len(rows) == len(original) == PLANNED['unique_rows'] and all(row.endswith(b'\n') for row in rows),
         'native audit did not retain all 72 ordered rows')
    longest = supervised = 0
    for ordinal, (source, record) in enumerate(zip(original, rows, strict=True), 1):
        item = json.loads(record)
        assert_fields(item, dict(ordinal=ordinal, source_line_SHA256=sha256(source).hexdigest()),
                      'native audit row identity or order differs')
        tokens, labels = item['tokens'], item['shifted_supervised_tokens']
        need(type(tokens) is int and type(labels) is int and 0 < labels < tokens <= 512
             and all(type(item[name]) is dict and hex64(item[name].get('SHA256')) for name in ('encoded', 'collated')),
             'native token, supervised-label or fingerprint record differs')
        longest = max(longest, tokens)
        supervised += labels
    return dict(actual_verified_rows=len(rows), maximum_row_tokens=longest, total_shifted_supervised_tokens=supervised)


def verify(root, raw_accounting, publication, *, data=DATA, training_sha=TRAIN_SHA256):
    need(type(publication) is str and re.fullmatch(r'[0-9a-f]{40}', publication), 'original publication was not bound')
    row = phase_row(raw_accounting)
    terminal = bool(row and row[2].split()[0] in TERMINAL)
    result = dict(state='ORIGINAL_ACCOUNTING_UNKNOWN', request=REQUEST, publication=publication, passed=False,
                  actual_verified_rows=0, planned_rows=72, training_READY=False, original_replay_allowed=False,
                  independent_accounting=dict(columns=COLUMNS, row=row, terminal=terminal,
                                              raw_SHA256=sha256(raw_accounting).hexdigest()))
    if not terminal:
        return result
    result['state'] = 'ORIGINAL_TERMINAL_UNQUALIFIED'
    if not terminal_pass(row, allocated_dcu):
        return result
    try:
        intent_raw = read_file(root, 'intent.json', 65536)
        intent = json.loads(intent_raw)
        epoch = intent['epoch']
        need(type(epoch) in (float, int) and math.isfinite(epoch) and epoch > 0, 'original CPU epoch differs')
        assert_fields(intent, dict(request=REQUEST, publication=publication, parent=PARENT, node=NODE,
                                   worker_deadline=epoch+390, whole_deadline=epoch+480, replay_allowed=False,
                                   source_SHA256=PINS, **PLANNED), 'original CPU intent differs')
        for name, expected in PINS.items():
            if name not in (MEMBERSHIP, TERM_HELPER, ROSTER):
                staged = read_file(root, 'src/'+Path(name).name, 150000)
                need(sha256(staged).hexdigest() == expected, 'original worker dependency differs: '+name)
        outcome_raw = read_file(root, 'outcome.json', 65536)
        outcome = json.loads(outcome_raw)
        assert_fields(outcome, dict(request=REQUEST, accounting_columns=COLUMNS, passed=True, planned=PLANNED,
                                    srun_return_code=0, failure_type=None, target_training_admitted=False,
                                    optimizer_updates=0, granted_dcu=0), 'original CPU controller outcome differs')
        assert_fields(outcome['terminal'], dict(terminal_visible=True, row=row), 'controller accounting differs from live Slurm')
        worker_raw = read_file(root, 'audit/result.json', 8192)
        worker_ref = dict(bytes=len(worker_raw), SHA256=sha256(worker_raw).hexdigest())
        need(outcome['worker_receipt'] == worker_ref, 'original worker result identity differs')
        worker = json.loads(worker_raw)
        assert_fields(worker, dict(passed=True, model_loaded=False, actual_rank_loader_observed=False,
                                   training_ready=False, epoch=epoch, deadline=epoch+390, **PLANNED),
                      'original worker result or scope differs')
        records_raw = read_file(root, 'audit/native-records-private.jsonl', 4*1024**2)
        capture = sha256(records_raw).hexdigest()
        need(worker['private_capture_SHA256'] == capture, 'original private capture digest differs')
        training_raw = read_file(data, TRAIN_FILE, 54246)
        facts = inventories(records_raw, training_raw, training_sha)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return dict(result, failure_type=type(error).__name__, failed_metadata_check=str(error)[:175])
    return dict(result, state='ORIGINAL_GEMMA_CPU72_INDEPENDENT_INPUT_VERIFICATION_PASS', passed=True, **facts,
                intent_SHA256=sha256(intent_raw).hexdigest(), outcome_SHA256=sha256(outcome_raw).hexdigest(),
                worker_receipt=worker_ref, private_capture_SHA256=capture, observed_encode_calls=144,
                observed_collator_calls=72, optimizer_updates=0, model_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', required=True)
    args = parser.parse_args()
    def expired(*_):
        raise TimeoutError('original Gemma CPU receipt read exceeded its 40-second budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(40)
    raw = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-20T00:00:00',
                                   '-o', ','.join(COLUMNS)], timeout=8)
    report = verify(RUN, raw, args.publication)
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
