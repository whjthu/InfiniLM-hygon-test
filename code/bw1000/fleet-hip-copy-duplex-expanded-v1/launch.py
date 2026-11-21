"""Reserve all eligible DMA shards before one finite campaign publication."""

import base64
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import time


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
REL = BASE.relative_to(REPO)
sys.path.insert(0, str(BASE))
from controller import EXCLUDED, FROZEN, MEMBERS, PCI, REQUEST, RUN, ZONES, need, qualify_pilot, read, save, sha, shards, source_identity, verify_source
sys.path.insert(0, str(REPO / 'tools'))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, fleet_view, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


WORKER = Path('code/bw1000/fleet-hip-copy-duplex-v2/SOURCE_FILES.json')
LEDGER = Path('coordination/training/training80-active-node-task-map-v1.json')
LOCAL = Path('local-only') / (RUN + '-original')
CASE = 'O01.hip-pinned-duplex-fleet-v1'


def source(repo):
    raw = read(repo / WORKER, 32 * 1024)
    need(sha(raw) == FROZEN['WORKER_MANIFEST.json'], 'qualified worker manifest differs')
    manifest = json.loads(raw)['files']
    refs = {name: Path(manifest[name]['path']) for name in ('worker.py','hip_runtime.py','node_sample.py','qualified_sampler.py')}
    refs.update({'CLAIM.json': REL/'CLAIM.json', 'controller.py': REL/'controller.py',
                 'WORKER_MANIFEST.json': WORKER, 'ledger.json': LEDGER,
                 'sync_dispatch.py': Path('code/bw1000/fleet-qwen-projection-sync-depth-v1/dispatch.py'),
                 'admission.py': Path('code/bw1000/fleet-lora-isolated-backfill-v1/admission.py')})
    need(set(refs) == MEMBERS and all((repo/file).is_relative_to(repo) for file in refs.values()), 'source scope differs')
    files = {name: read(repo/path, 32 * 1024) for name, path in refs.items()}
    verify_source(files)
    return refs, files


def available(snapshot, repo):
    now = time.monotonic()
    _, facts, fresh = fleet_view(snapshot, now, time.time() - now, repo / LEDGER)
    need(fresh, 'new CPU-only dispatch snapshot required')
    admitted = {}
    for node, row in sorted(facts.items()):
        if node == EXCLUDED or row['admission'] != 'clear' or row['zone'] not in ZONES:
            continue
        actual = snapshot['nodes'][node]
        pci, before = actual['PCIs'], actual.get('device_probe_step')
        if (len(set(pci)) == 8 and all(PCI.fullmatch(p) for p in pci) and isinstance(before, str)
                and re.fullmatch(row['parent'] + r'\.[0-9]+', before) and actual['active_steps'] == []
                and actual.get('probe_error') is None and all(type(actual.get(key)) is int and actual[key] == 0
                   for key in ('kfd_process_count','own_device_fd_process_count'))):
            admitted[node] = dict(PCI=sorted(pci), before_CPU_step=before, parent=row['parent'], zone=row['zone'])
    return admitted


def protect(schedule):
    occupied = set(schedule['queue_held_nodes'])
    for hold in schedule['original_launch_holds'].values():
        occupied.update(hold['nodes'])
    for transaction in schedule['rows']:
        if (transaction['dispatch_state'] == 'proposed_for_root' or
                (transaction['kind'] == 'training' and transaction['dispatch_state'] == 'awaiting_original_queue_release')):
            occupied.update(transaction['placement'])
    return occupied


def reserve(queue, repo, board, snapshot, publication, files, pilot_pin):
    folder = repo / LOCAL
    need(folder.resolve().is_relative_to(repo/'local-only'), 'campaign intent escaped its owned namespace')
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        need(not folder.exists() and not folder.is_symlink(), 'original campaign exists; reconcile once')
        schedule = plan(snapshot, json.loads(board.read_bytes()), repo=repo, ledger_path=repo/LEDGER, holds=queue_holds(db))
        occupied = protect(schedule)
        chosen = {node: row for node, row in available(snapshot,repo).items() if node not in occupied}
        groups = shards(chosen)
        need(bool(groups), 'no presently eligible whole nodes')
        receipt = snapshot['local_receipt']
        campaign = dict(request=REQUEST, epoch=time.time(), publication=publication, selected=chosen, shards=groups,
                        pilot_SHA256=pilot_pin, snapshot_SHA256=receipt['payload_SHA256'],
                        source_SHA256=source_identity(files))
        folder.mkdir(mode=0o700)
        save(folder/'plan.json', campaign)
        handles = {}
        claim = dict(path=(REL/'CLAIM.json').as_posix(), SHA256=sha(files['CLAIM.json']), bytes=len(files['CLAIM.json']))
        for label, unit in groups.items():
            parent, request = unit['parent'], unit['request']
            counts = {node:8 for node in unit['nodes']}
            binding = dict(candidate=label, case=CASE, kind='operator', campaign_request=REQUEST, request=request,
                           publication_commit=publication, workload_SHA256=digest(dict(case=CASE, claim=claim, label=label)),
                           placement_by_parent={parent:counts}, placement=counts, donor_zone=unit['zone'], borrower_zone='O01',
                           snapshot_payload_SHA256=campaign['snapshot_SHA256'], campaign_pilot_SHA256=pilot_pin,
                           expires_local_monotonic=receipt['received_monotonic']+SNAPSHOT_MAX_AGE_S-receipt['roundtrip_s'],
                           claim=claim, launcher=dict(path=(REL/'controller.py').as_posix(), SHA256=sha(files['controller.py'])),
                           invocation=dict(kind='pinned_finite_code'), replay_allowed=False, counts_toward_training_reserve=False)
            binding['handoff_SHA256'] = digest(binding)
            path = LOCAL / (label+'-intent.json')
            run = digest(dict(stage=DIRECT_STAGE, request=request))
            handoff = reserve_handoff(queue, run, binding, path.as_posix())
            save(repo/path, dict(binding, queue_run=run))
            handles[label]=handoff
        still_available = available(snapshot,repo)
        need(all(still_available.get(node) == row for node,row in chosen.items()), 'device receipt changed before durable reserve')
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return campaign, handles


def ship(publication, snapshot, pilot_original_raw, transport, *, repo=REPO, board=BACKLOG, database=QUEUE_DB,
         check_publication=published):
    """Publish once after acquiring distinct local reservations for the exact planned shards."""
    need(callable(transport) and re.fullmatch('[a-f0-9]{40}', publication), 'published revision and existing transport required')
    repo, board, database = (path.resolve(strict=True) for path in (repo,board,database))
    need(board==repo/BACKLOG.relative_to(REPO) and database==repo/QUEUE_DB.relative_to(REPO), 'exact shared queue required')
    pilot_pin = qualify_pilot(pilot_original_raw)
    refs, files = source(repo)
    check_publication(repo, publication, sorted({p.as_posix() for p in refs.values()} | {
        (REL/'launch.py').as_posix(), board.relative_to(repo).as_posix(),
        'tools/prepare_training80_dispatch_handoff.py','tools/report_training80_backlog.py','tools/tuning_queue.py'}))
    snapshot = deepcopy(snapshot)
    queue = Queue(database)
    folder = repo / LOCAL
    with closing(queue.db):
        campaign, handles = reserve(queue,repo,board,snapshot,publication,files,pilot_pin)
        attempts = dict(request=REQUEST, planned_nodes=len(campaign['selected']), planned_cards=8*len(campaign['selected']),
                        planned_shards=len(handles), local_reservations=len(handles), transport_attempted=False,
                        remote_outcome='NOT_ATTEMPTED', actual_dispatched_cards=None)
        armed = set()
        try:
            envelope = dict(plan=campaign, files={name:[sha(raw),base64.b64encode(raw).decode('ascii')]
                                                  for name,raw in sorted(files.items())})
            code = ('PAYLOAD='+repr(envelope)+'\n').encode('ascii') + files['controller.py']
            with (folder/'stdout.private').open('xb') as out, (folder/'stderr.private').open('xb') as err:
                fresh = available(snapshot,repo)
                need(time.time() - campaign['epoch'] < 10 and
                     all(fresh.get(node)==row for node,row in campaign['selected'].items()),
                     'original admission expired before campaign transport')
                for label, handle in handles.items():
                    queue.submission(handle['run'],handle['handle'],unknown=True)
                    armed.add(label)
                attempts.update(transport_attempted=True,remote_outcome='UNKNOWN')
                code_rc = transport(code,out,err,230).returncode
                need(type(code_rc) is int, 'finite transport did not return an exit code')
                attempts['transport_returncode'] = code_rc
        except Exception as exc:
            attempts['transport_error'] = type(exc).__name__
        attempts['armed_shards'] = sorted(armed)
        receipt = folder/'transport.json'
        save(receipt, attempts)
        for label in handles.keys() - armed:
            handle = handles[label]
            queue.retire_unattempted_direct(handle['run'],handle['handle'],sha(receipt.read_bytes()))
    return attempts
