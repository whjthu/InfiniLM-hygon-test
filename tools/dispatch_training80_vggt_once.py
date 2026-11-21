"""Claim the frozen VGGT run, then guard its one injected transport call."""

from base64 import b64encode
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
import math
import time

from dispatch_glm474_fresh_kv_cpu_once import write_once
from preflight_vggt_tum_bidirectional30 import check_source, exact
from prepare_training80_dispatch_handoff import claim_handoff
from prepare_training80_vggt_admission import (
    ATTEMPT, CANDIDATE, CLAIM_SHA, INTENT, NODE, PACKAGE, PARENT, REQUEST, SOURCE_MANIFEST, exact_value,
)
from report_training80_backlog import QUEUE_DB, REPO, need
from tuning_queue import Queue


GUARD_RECEIPT = 'guard-pretransport.json'
CASE = 'T12.domain-tum-bidirectional30'


def load_scope(repo, packet):
    sources, _ = check_source(repo)
    package = repo / PACKAGE
    manifest = json.loads(exact(package, 'SOURCE_FILES.json', SOURCE_MANIFEST))['files']
    launcher_pin = manifest['launch.py']
    launcher_path = package / 'launch.py'
    exact_value(packet, dict(candidate=CANDIDATE, case=CASE, kind='training', request=REQUEST, requested_DCU=1,
                            placement_by_parent={PARENT: {NODE: 1}}, attempt_receipts=[INTENT, ATTEMPT],
                            claim=dict(path=(PACKAGE/'CLAIM.json').as_posix(), SHA256=CLAIM_SHA),
                            manifest=dict(path=(PACKAGE/'SOURCE_FILES.json').as_posix(), SHA256=SOURCE_MANIFEST),
                            launcher=dict(path=(PACKAGE/'launch.py').as_posix(), SHA256=launcher_pin['SHA256']),
                            invocation=dict(kind='coordinator_callback', entry='main',
                                            arguments=[packet['publication_commit']], injected_keyword='transport')),
                'expected the exact frozen VGGT handoff')
    claim = exact(package, 'CLAIM.json', CLAIM_SHA, manifest['CLAIM.json']['bytes'])
    sources['task-claim.json'] = claim
    files = {'src/'+name: dict(sha=sha256(raw).hexdigest(), bytes=len(raw), b64=b64encode(raw).decode('ascii'))
             for name, raw in sources.items()}
    worker_pin = manifest['run_tum_continuous_once.py']
    worker = exact(package, 'run_tum_continuous_once.py', worker_pin['SHA256'], worker_pin['bytes'])
    launcher = exact(package, 'launch.py', launcher_pin['SHA256'], launcher_pin['bytes'])
    program = compile(launcher, str(launcher_path), 'exec')
    namespace = {'__name__': 'guarded_frozen_vggt_launch', '__file__': str(launcher_path)}
    exec(program, namespace)
    return namespace['main'], files, worker


def read_callback_markers(intent, publication, files, worker, body, stdout, stderr, timeout):
    raw = {}
    for name in ('intent.json', 'attempt.json'):
        path = intent.parent / name
        need(path.resolve(strict=True) == path and path.is_file(), 'original VGGT marker path differs')
        with path.open('rb') as stream:
            raw[name] = stream.read(8193)
        need(0 < len(raw[name]) <= 8192, 'original VGGT marker is empty or too large')
    record, attempt = (json.loads(raw[name]) for name in ('intent.json', 'attempt.json'))
    epoch, started, limit = record['epoch'], attempt['started_epoch'], attempt['timeout_seconds']
    need(type(epoch) is float and all(type(value) in (int, float) and math.isfinite(value)
                                     for value in (epoch, started, limit, timeout)) and
         0 < epoch <= started < epoch + 1800 and 0 < timeout <= limit <= 1800,
         'original VGGT attempt clock or timeout differs')
    payload = dict(publication=publication, request=REQUEST, epoch=epoch, deadline=epoch+1800,
                   source_manifest_SHA256=SOURCE_MANIFEST, claim_SHA256=CLAIM_SHA)
    expected_body = ('PAYLOAD='+repr(dict(payload, files=files))+'\n').encode('ascii') + worker
    need(type(body) is bytes and body == expected_body, 'original VGGT transport body differs')
    exact_value(record, dict(payload, body_SHA256=sha256(body).hexdigest(), unknown_replay_allowed=False),
                'original VGGT intent does not bind this body')
    exact_value(attempt, dict(request=REQUEST, attempted=True), 'original VGGT attempt marker differs')
    need(stdout.name == str(intent.parent/'stdout.private') and stderr.name == str(intent.parent/'stderr.private'),
         'original VGGT capture paths differ')
    return record, started, {name: sha256(data).hexdigest() for name, data in raw.items()}


def callback_clock(packet, received, started, deadline):
    now = time.monotonic()
    wall = time.time()
    anchor = wall - now
    need(math.isfinite(now) and math.isfinite(wall) and received <= now and
         abs(anchor - packet['boot_epoch_anchor']) < 2 and started <= wall < deadline,
         'VGGT callback clock no longer binds the original attempt and device receipt')
    return now, anchor


def dispatch(snapshot, backlog, packet, transport, *, repo=REPO, database=QUEUE_DB):
    """Accept an unclaimed select packet and the existing finite stdin transport."""
    repo, database = repo.resolve(strict=True), database.resolve(strict=True)
    need(callable(transport) and database == repo / QUEUE_DB.relative_to(REPO),
         'the shared queue and existing bounded transport are required')
    snapshot, packet = deepcopy(snapshot), deepcopy(packet)
    intent = repo / INTENT
    need(intent.parent.resolve() == intent.parent and not intent.parent.exists(),
         'original VGGT local namespace exists or differs; reconcile without replay')
    launch, files, worker = load_scope(repo, packet)
    publication = packet['publication_commit']
    receipt_path = intent.parent / GUARD_RECEIPT
    result = dict(request=REQUEST, callback_reached=False, transport_attempted=False,
                  actual_slurm_submissions=0, replay_allowed=False)

    queue = Queue(database)
    with closing(queue.db):
        claim = claim_handoff(queue, snapshot, backlog, packet, INTENT, repo=repo)
        run, handle = claim['run'], claim['handle']
        admitted = snapshot['local_receipt']
        expires = claim['expires_local_monotonic']

        def guarded(body, stdout, stderr, timeout):
            need(not result['callback_reached'], 'original VGGT callback already consumed')
            result['callback_reached'] = True
            record, started, marker_hashes = read_callback_markers(
                intent, publication, files, worker, body, stdout, stderr, timeout)
            current = queue.dispatch_state(run)
            need(current['state'] == 'DISPATCHING' and current['handle'] == handle and
                 current['reservation']['snapshot_SHA256'] == admitted['payload_SHA256'] and
                 current['reservation']['dispatch_ticket'] == packet['handoff_SHA256'],
                 'original VGGT claim changed before transport')
            now, anchor = callback_clock(packet, admitted['received_monotonic'], started, record['deadline'])
            if now > expires:
                proof = dict(request=REQUEST, run=run, publication=publication, intent=INTENT,
                             marker_SHA256=marker_hashes, body_SHA256=record['body_SHA256'],
                             claim_SHA256=CLAIM_SHA, manifest_SHA256=SOURCE_MANIFEST,
                             handoff_SHA256=packet['handoff_SHA256'], device_receipt=admitted,
                             expires_local_monotonic=expires, callback_monotonic=now, boot_epoch_anchor=anchor,
                             launcher_epoch=record['epoch'], launcher_deadline=record['deadline'],
                             launcher_attempt_started_epoch=started,
                             reason='DEVICE_ADMISSION_EXPIRED_AT_ORIGINAL_CALLBACK',
                             remote_outcome='NOT_ATTEMPTED', transport_attempted=False, replay_allowed=False)
                write_once(receipt_path, proof)
                receipt_sha = sha256(receipt_path.read_bytes()).hexdigest()
                queue.retire_unattempted_direct(run, handle, receipt_sha)
                result['guard_receipt_SHA256'] = receipt_sha
                raise ValueError('VGGT device admission expired before original transport')

            queue.submission(run, handle, unknown=True)
            now, _ = callback_clock(packet, admitted['received_monotonic'], started, record['deadline'])
            need(now <= expires, 'VGGT device admission expired while arming transport')
            result.update(transport_attempted=True, actual_slurm_submissions=None)
            return transport(body, stdout, stderr, timeout)

        launcher_result, launcher_error = None, None
        try:
            launcher_result = launch(publication, guarded)
        except Exception as error:
            launcher_error = type(error).__name__
        state = queue.dispatch_state(run)['state']
        return dict(result, run=run, state=state, launcher_returncode=launcher_result,
                    launcher_error=launcher_error, terminal_verified=False)
