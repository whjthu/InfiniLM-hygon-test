"""Admit and send the bounded O01 single-node pilot via the existing finite transport."""

import base64
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from pilot import LABEL, NODE, PARENT, PCI, REQUEST, RUN_NAME, ZONE, need, source_identity, source_verify
sys.path.insert(0, str(REPO / 'tools'))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, fleet_view, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


BASE = HERE.relative_to(REPO)
CLAIM = BASE / 'ROOT_CPU_CLAIM.json'
ENTRY = BASE / 'pilot.py'
LEDGER = Path('coordination/training/training80-active-node-task-map-v1.json')
INTENT = Path('local-only') / (RUN_NAME + '-original') / 'intent.json'
CASE = 'T07.domain-style18-native12-input'


def source(repo):
    manifest_path = BASE / 'SOURCE_FILES.json'
    refs = json.loads((repo / manifest_path).read_bytes())['files']
    members = {name: BASE / name for name in refs}
    members.update({'CLAIM.json': CLAIM, 'SOURCE_FILES.json': manifest_path, 'pilot.py': ENTRY,
                    'sync_dispatch.py': Path('code/bw1000/fleet-qwen-projection-sync-depth-v1/dispatch.py'),
                    'admission.py': Path('code/bw1000/fleet-lora-isolated-backfill-v1/admission.py'),
                    'ledger.json': LEDGER})
    files = {}
    for name, relative in members.items():
        file = repo / relative
        need(file.resolve(strict=True) == file and file.is_relative_to(repo), 'source path is outside the published tree')
        files[name] = file.read_bytes()
    csv = repo / 'local-only/training80-user-all-nodes-live-first/ZImage-style18-train-f71b51.csv'
    files['metadata.csv'] = csv.read_bytes()
    source_verify(files)
    return members, files


def device(snapshot, repo):
    now = time.monotonic()
    _, nodes, fresh = fleet_view(snapshot, now, time.time() - now, repo / LEDGER)
    need(fresh, 'new CPU-only dispatch snapshot required')
    observed, facts = snapshot['nodes'][NODE], nodes[NODE]
    receipt = snapshot['local_receipt']
    expected = observed['PCIs']
    before = observed.get('device_probe_step')
    need(facts['admission'] == 'clear' and (facts['parent'], facts['zone']) == (PARENT, ZONE)
         and observed['active_steps'] == [] and observed.get('probe_error') is None
         and isinstance(before, str) and re.fullmatch(PARENT + r'\.[0-9]+', before)
         and len(set(expected)) == 8 and all(PCI.fullmatch(pci) for pci in expected)
         and all(type(observed.get(key)) is int and observed[key] == 0
                 for key in ('kfd_process_count', 'own_device_fd_process_count')),
         'candidate O01 node is not freshly clear')
    return dict(receipt, PCI=sorted(expected), before_CPU_step=before)


def protect(schedule):
    protected = set(schedule['queue_held_nodes'])
    for item in schedule['original_launch_holds'].values():
        protected.update(item['nodes'])
    for row in schedule['rows']:
        if (row['dispatch_state'] == 'proposed_for_root' or
                (row['kind'] == 'training' and row['dispatch_state'] == 'awaiting_original_queue_release')):
            protected.update(row['placement'])
    return protected


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def claim_local(queue, repo, board, snapshot, binding):
    db = queue.db
    slot = repo / INTENT
    need(slot.parent.resolve().is_relative_to(repo / 'local-only'), 'intent must stay in local-only')
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    db.execute('BEGIN IMMEDIATE')
    try:
        need(not slot.parent.exists() and not slot.parent.is_symlink(), 'original intent exists; reconcile')
        state = json.loads(board.read_bytes())
        schedule = plan(snapshot, state, repo=repo, ledger_path=repo / LEDGER, holds=queue_holds(db))
        need(NODE not in protect(schedule), 'a queued or held scope has this O01 node')
        need(device(snapshot, repo) == binding['device_admission'], 'CPU-only admission changed before reserve')
        receipt = reserve_handoff(queue, run, binding, INTENT.as_posix())
        slot.parent.mkdir(mode=0o700)
        save(slot, dict(binding, queue_run=run))
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    return receipt


def ship(publication, snapshot, transport, *, repo=REPO, board=BACKLOG, database=QUEUE_DB,
         check_publication=published):
    """Send once. A transport exit or unobserved result never releases the node."""
    need(callable(transport) and re.fullmatch(r'[a-f0-9]{40}', publication), 'published revision and existing transport needed')
    repo, board, database = (path.resolve(strict=True) for path in (repo, board, database))
    need(board == repo / BACKLOG.relative_to(REPO) and database == repo / QUEUE_DB.relative_to(REPO),
         'original board and shared database required')
    members, files = source(repo)
    check_publication(repo, publication, sorted({path.as_posix() for path in members.values()} | {
        (BASE / 'launch.py').as_posix(), board.relative_to(repo).as_posix(),
        'tools/prepare_training80_dispatch_handoff.py', 'tools/report_training80_backlog.py', 'tools/tuning_queue.py'}))
    snapshot = deepcopy(snapshot)
    admitted = device(snapshot, repo)
    claim = dict(path=CLAIM.as_posix(), SHA256=sha256(files['CLAIM.json']).hexdigest(),
                 bytes=len(files['CLAIM.json']))
    binding = dict(candidate=LABEL, case=CASE, kind='validation', request=REQUEST, publication_commit=publication,
                   workload_SHA256=digest(dict(case=CASE, claim=claim)),
                   placement_by_parent={PARENT: {NODE: 0}}, placement={NODE: 0},
                   snapshot_payload_SHA256=admitted['payload_SHA256'], device_admission=admitted,
                   expires_local_monotonic=admitted['received_monotonic']+SNAPSHOT_MAX_AGE_S-admitted['roundtrip_s'],
                   claim=claim, launcher=dict(path=ENTRY.as_posix(), SHA256=sha256(files['pilot.py']).hexdigest()),
                   invocation=dict(kind='pinned_finite_code'), source_contract_SHA256=source_identity(files),
                   transport_timeout_seconds=310, replay_allowed=False, counts_toward_training_reserve=False)
    binding['handoff_SHA256'] = digest(binding)
    queue = Queue(database)
    directory = (repo / INTENT).parent
    with closing(queue.db):
        local = claim_local(queue, repo, board, snapshot, binding)
        run, handle = local['run'], local['handle']
        result = dict(request=REQUEST, queue_run=run, transport_attempted=False, remote_outcome='NOT_ATTEMPTED',
                      terminal_verified=False)
        retire = False
        try:
            with (directory / 'stdout.private').open('xb') as out, (directory / 'stderr.private').open('xb') as err:
                if device(snapshot, repo) != admitted:
                    raise ValueError('original admission differs before transport')
                controller_plan = dict(request=REQUEST, epoch=time.time(), PCI={NODE: admitted['PCI']},
                                       before_CPU_step=admitted['before_CPU_step'],
                                       before_snapshot_SHA256=admitted['payload_SHA256'],
                                       source_contract_SHA256=binding['source_contract_SHA256'],
                                       publication=publication, handoff_SHA256=binding['handoff_SHA256'])
                save(directory / 'original-plan.json', controller_plan)
                embedded = {'plan': controller_plan, 'files': {
                    name: [sha256(raw).hexdigest(), base64.b64encode(raw).decode('ascii')]
                    for name, raw in sorted(files.items())}}
                code = ('PAYLOAD=' + repr(embedded) + '\n').encode('ascii') + files['pilot.py']
                need(device(snapshot, repo) == admitted, 'original admission expired while preparing transport')
                queue.submission(run, handle, unknown=True)
                result.update(transport_attempted=True, remote_outcome='UNKNOWN')
                reply = transport(code, out, err, binding['transport_timeout_seconds'])
                need(type(reply.returncode) is int, 'finite transport did not return an exit code')
                result['transport_returncode'] = reply.returncode
        except Exception as exc:
            retire = not result['transport_attempted']
            result['transport_error'] = type(exc).__name__
        receipt = directory / 'transport.json'
        save(receipt, result)
        if retire:
            queue.retire_unattempted_direct(run, handle, sha256(receipt.read_bytes()).hexdigest())
    return result
