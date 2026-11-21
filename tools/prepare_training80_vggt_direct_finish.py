"""Bind the original VGGT training audit to a claimed training80 finish packet."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from read_vggt_tum_bidirectional30_terminal import (
    LABEL, NODE, PARENT, REQUEST, accounting, allocated_dcu, completed, hex64, need, number, terminal,
)
from report_training80_backlog import QUEUE_DB, REPO, hex_value
from tuning_queue import DIRECT_STAGE, digest


CASE = 'T12.domain-tum-bidirectional30'
INTENT = f'local-only/{LABEL}-original-dispatch/intent.json'
DIRECT_RUN = digest(dict(stage=DIRECT_STAGE, request=REQUEST))


def bounded(path, limit=1024 * 1024):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, 'missing or oversized input: ' + path.name)
    return raw


def claimed_run(database):
    uri = database.resolve(strict=True).as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        row = connection.execute('SELECT stage,state,binding,handle FROM runs WHERE id=?', (DIRECT_RUN,)).fetchone()
    need(row is not None and row[0] == DIRECT_STAGE and row[1] in {'DISPATCHING', 'SUBMITTED', 'UNKNOWN'},
         'original VGGT direct claim is not active')
    return row[1], json.loads(row[2]), json.loads(row[3])


def prepare(claim, local_raw, target_raw, accounting_raw, audit_raw):
    state, packet, handle = claim
    need(state in {'DISPATCHING', 'SUBMITTED', 'UNKNOWN'}, 'no active original claim')
    placement = {PARENT: {NODE: 1}}
    need(packet['request'] == handle['request_id'] == REQUEST and packet['case'] == CASE and
         packet['kind'] == handle['direct_kind'] == 'training' and packet['requested_DCU'] == 1 and
         packet['placement_by_parent'] == handle['placement_by_parent'] == placement and
         handle['parent'] == PARENT and handle['nodes'] == [NODE] and handle['intent'] == INTENT and
         INTENT in packet['attempt_receipts'], 'claim does not bind the original VGGT training placement')
    need(hex64(packet['handoff_SHA256']) and packet['handoff_SHA256'] == handle['handoff_SHA256'] ==
         digest({key: value for key, value in packet.items() if key != 'handoff_SHA256'}),
         'claimed handoff bytes differ')
    local_intent, target_intent, audit = (json.loads(raw) for raw in (local_raw, target_raw, audit_raw))
    expected = dict(request=REQUEST, publication=packet['publication_commit'],
                    source_manifest_SHA256=packet['manifest']['SHA256'], claim_SHA256=packet['claim']['SHA256'])
    epoch = local_intent['epoch']
    need(hex_value(expected['publication'], 40) and hex64(expected['source_manifest_SHA256']) and
         hex64(expected['claim_SHA256']) and number(epoch) and epoch > 0 and
         hex64(local_intent['body_SHA256']) and local_intent['unknown_replay_allowed'] is False,
         'malformed original local intent')
    expected.update(epoch=epoch, deadline=epoch + 1800)
    need(all(document.get(key) == value and type(document[key]) is type(value)
             for document in (local_intent, target_intent) for key, value in expected.items()) and
         target_intent['parent'] == PARENT and target_intent['node'] == NODE,
         'local or target intent does not bind the original epoch and publication')
    rows = accounting(accounting_raw)
    need(terminal(rows['gpu']) and completed('pre', rows['pre']) and completed('post', rows['post']),
         'original numbered GPU terminal and completed pre/post are required')
    gpu = rows['gpu']
    need(allocated_dcu(gpu[6]) == 1 and 'node=1' in gpu[6].split(','),
         'original GPU row does not bind one allocated DCU on one node')
    need(handle.get('gpu_step', gpu[0]) == gpu[0], 'observed GPU step differs from the durable handle')
    original_hashes = audit['original_file_SHA256']
    need(audit['request'] == REQUEST and audit['phase_rows'] == rows and
         audit['accounting_SHA256'] == sha256(accounting_raw).hexdigest() and
         original_hashes['intent.json'] == sha256(target_raw).hexdigest() and
         all(hex64(original_hashes.get(name)) for name in ('before.json', 'after.json')) and
         audit['gpu_terminal'] is True and audit['independent_post_clean'] is True,
         'independent audit does not bind the actual accounting, target intent and post-clean evidence')
    count = audit['independently_verified_updates']
    passed = audit['passed']
    need(type(audit['planned_updates']) is int and audit['planned_updates'] == 30 and
         type(count) is int and 0 <= count <= 30 and type(passed) is bool and
         (not passed or (count == 30 and completed('gpu', gpu) and audit['original_collector_verified'] is True
                         and audit.get('recovery', {}).get('passed') is True)),
         'independent update counts or pass evidence differ')
    return dict(request_id=REQUEST, intent=INTENT, parent=PARENT, placement_by_parent=placement,
                qualification='training', terminal=True, post_clean=True, passed=passed, keeper_passed=None,
                planned=30, completed=count, missing=30-count, failed=0, rejected=0, canceled=0,
                promote=False, evidence_SHA256=sha256(audit_raw).hexdigest(),
                local_intent_SHA256=sha256(local_raw).hexdigest(), target_intent_SHA256=sha256(target_raw).hexdigest(),
                accounting_SHA256=audit['accounting_SHA256'], actual_gpu_step=gpu[0], actual_gpu_state=gpu[2],
                actual_gpu_exit_code=gpu[3], actual_pre_step=rows['pre'][0], actual_post_step=rows['post'][0],
                handoff_SHA256=packet['handoff_SHA256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_intent', type=Path)
    parser.add_argument('raw_accounting', type=Path)
    parser.add_argument('independent_audit', type=Path)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    args = parser.parse_args()
    original = (REPO / INTENT).resolve(strict=True)
    need(original.is_relative_to(REPO / 'local-only'), 'original local intent escaped local-only')
    result = prepare(claimed_run(args.queue_db), bounded(original), bounded(args.target_intent),
                     bounded(args.raw_accounting, 2 * 1024 * 1024), bounded(args.independent_audit))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
