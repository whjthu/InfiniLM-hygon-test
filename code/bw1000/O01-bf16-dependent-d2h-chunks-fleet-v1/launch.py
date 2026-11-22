"""Authorize one eligible-fleet transfer interaction comparison through independent shard reservations."""

from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from package import D2H_REFERENCE, ORIGINAL, QWEN_SHAPE, QWEN_VJP, TRANSFER_BASELINE, V5_TERMINAL, T01_PROVIDER, prepare, source
from scope import LABEL, REQUEST, RUN

if REQUEST != '93c5218eea4e4f1d8a246ba7583dac85':
    raise RuntimeError('Dependent same-stream D2H chunk fleet launcher loaded a different campaign scope')

sys.path.insert(0, str(REPO / 'tools'))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


BASE = HERE.relative_to(REPO)
CLAIM = BASE / 'ROOT_CLAIM.json'
DIRECTORY = Path('local-only') / (RUN+'-original')
CASE = 'O01.bf16-dependent-d2h-chunks-eligible-fleet'


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def priority_protection(schedule):
    protected = set(schedule['queue_held_nodes'])
    for item in schedule['original_launch_holds'].values():
        protected.update(item['nodes'])
    for row in schedule['rows']:
        if row['dispatch_state'] == 'proposed_for_root' or (
                row['kind'] == 'training' and row['dispatch_state'] == 'awaiting_original_queue_release'):
            protected.update(row['placement'])
    return protected


def reserve_wave(queue, campaign, publication, receipt, cutoff, claim_pin, directory):
    """Create all disjoint shard handles within the caller's transaction."""
    selected = set()
    handles = {}
    for label, unit in campaign['shards'].items():
        nodes = set(unit['nodes'])
        need(nodes and not (selected & nodes), 'overlapping shard reservation')
        selected.update(nodes)
        intent = (directory / 'shards' / (label+'.json')).as_posix()
        packet = dict(candidate=label, case=CASE, kind='operator', request=unit['request'],
                      placement_by_parent={unit['parent']:{node:8 for node in sorted(nodes)}},
                      publication_commit=publication, workload_SHA256=claim_pin, campaign_request=REQUEST,
                      snapshot_payload_SHA256=receipt['payload_SHA256'], snapshot_raw_SHA256=receipt['raw_response_SHA256'],
                      snapshot_request=receipt['request_id'], expires_local_monotonic=cutoff,
                      launcher=(BASE/'controller.py').as_posix(), invocation={'kind':'pinned_finite_code'},
                      replay_allowed=False)
        packet['handoff_SHA256'] = digest(packet)
        run = digest(dict(stage=DIRECT_STAGE, request=unit['request']))
        handle = reserve_handoff(queue, run, packet, intent)
        handles[label] = dict(run=run, handle=handle['handle'], packet=packet)
    need(selected == set(campaign['selected']), 'original graph wave denominator differs')
    return handles


def ship(publication, snapshot, transport, *, requested=None, repo=REPO, check_publication=published):
    """Reserve only fresh idle nodes. Leave unknown shards protected for reconciliation."""
    repo = repo.resolve(strict=True)
    check_publication_paths = {p.as_posix() for p in source(repo)[0].values()}
    check_publication_paths.update(p.as_posix() for p in (
        TRANSFER_BASELINE, ORIGINAL, D2H_REFERENCE, V5_TERMINAL, T01_PROVIDER, QWEN_VJP, QWEN_SHAPE, BASE/'package.py',BASE/'launch.py',BASE/'overlap.py',CLAIM,BACKLOG.relative_to(REPO),
        Path('tools/prepare_training80_dispatch_handoff.py'),Path('tools/report_training80_backlog.py'),
        Path('tools/tuning_queue.py')))
    raw_claim = (repo/CLAIM).read_bytes()
    claim = json.loads(raw_claim)
    need(claim == dict(label=LABEL,request=REQUEST,operator_GO=True,training_GO=False,Model_GO=False,
                       ownership='fresh idle whole nodes within their existing ledger zones',
                       requested_DCU_cap=592,priority='yield-to-actual-training-and-inference',replay_allowed=False),
         'published operator backfill claim differs')
    check_publication(repo,publication,sorted(check_publication_paths))
    receipt = snapshot['local_receipt']
    cutoff = receipt['received_monotonic'] + SNAPSHOT_MAX_AGE_S - receipt['roundtrip_s']
    directory = repo/DIRECTORY
    board = repo/BACKLOG.relative_to(REPO)
    queue = Queue(repo/QUEUE_DB.relative_to(REPO))
    with closing(queue.db):
        db = queue.db
        db.execute('BEGIN IMMEDIATE')
        try:
            need(not directory.exists() and not directory.is_symlink(), 'original campaign exists; reconcile')
            schedule = plan(snapshot,json.loads(board.read_bytes()),repo=repo,holds=queue_holds(db))
            protected = priority_protection(schedule)
            campaign, body, members = prepare(publication,time.time(),snapshot,protected,requested,repo=repo)
            need(set(members) <= check_publication_paths and time.monotonic()<cutoff
                 and len(body)<=131072 and 0<len(campaign['selected'])<=74,
                 'claim membership, payload or admission differs')
            handles = reserve_wave(queue,campaign,publication,receipt,cutoff,sha256(raw_claim).hexdigest(),DIRECTORY)
            directory.mkdir(mode=0o700)
            (directory/'shards').mkdir(mode=0o700)
            save(directory/'original-plan.json',campaign)
            save(directory/'original-shards.json', handles)
            for label, data in handles.items():
                save(directory/'shards'/(label+'.json'),data)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        result = dict(request=REQUEST,planned_nodes=len(campaign['selected']),
                      planned_cards=8*len(campaign['selected']),planned_shards=len(handles),
                      transport_attempted=False,remote_outcome='NOT_ATTEMPTED',terminal_verified=False,
                      replay_allowed=False)
        marked = set()
        try:
            with (directory/'stdout.private').open('xb') as stdout,(directory/'stderr.private').open('xb') as stderr:
                need(time.monotonic()<cutoff and 0<=time.time()-campaign['campaign_epoch']<20,
                     'original dispatch epoch or fresh admission expired')
                for label, row in handles.items():
                    queue.submission(row['run'],row['handle'],unknown=True)
                    marked.add(label)
                need(0<=time.time()-campaign['campaign_epoch']<20, 'original campaign expired before transport')
                result.update(transport_attempted=True,remote_outcome='UNKNOWN')
                result['transport_returncode'] = transport(body,stdout,stderr,575).returncode
        except Exception as error:
            result['transport_error'] = type(error).__name__
        path = directory/'transport.json'
        save(path,result)
        if not result['transport_attempted']:
            for label, row in handles.items():
                if label not in marked:
                    queue.retire_unattempted_direct(row['run'],row['handle'],sha256(path.read_bytes()).hexdigest())
    return result
