"""Reserve the shared O01 node before the sole Deep CSA build transport call."""

from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
import time

from control_deep_csa_identity_compile import (
    CLAIM, HOST, LABEL, LEDGER, MEMBERSHIP, NODE, PARENT, PLACEMENT, REQUEST, ROOT, STAGE_SHA, ZONE, scope,
)
from dispatch_glm474_fresh_kv_cpu_once import write_once
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import (BACKLOG, QUEUE_DB, REPO, fleet_view, hex_value, need,
                                      queue_holds, read_original_holds)
from sample_training80_busy import PCI
from tuning_queue import DIRECT_STAGE, Queue, SNAPSHOT_MAX_AGE_S, digest


IDENT = 'deep-csa-identity-swa-host-compile-v1'
CASE = 'O01.deep-csa-identity-swa-compile'
INTENT = f'local-only/{LABEL}-original-dispatch/intent.json'
WORKLOAD = digest(dict(case=CASE, stage=STAGE_SHA))
ENTRY = 'tools/dispatch_deep_csa_identity_compile_once.py'
MODES = ('--prepare-stage', '--dispatch-original', '--reconcile-existing')
SUPPORT = {'tools/prepare_training80_dispatch_handoff.py', 'tools/report_training80_backlog.py',
           'tools/sample_training80_busy.py', 'tools/tuning_queue.py'}


def backlog_identity(claim_ref):
    return dict(id=IDENT, case=CASE, kind='validation', state='blocked', cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, evidence=claim_ref, attempt_receipts=[INTENT])


def host_argv(remote_repository, claim_pin, claim_sha, mode):
    root = PurePosixPath(remote_repository)
    need(root.as_posix() == remote_repository and root.is_relative_to(ROOT.as_posix())
         and all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) and name != '..' for name in root.parts[1:])
         and mode in MODES, 'invalid target project checkout or entry mode')
    bootstrap = (
        'import hashlib,runpy,sys\n'
        'from pathlib import Path\n'
        'entry,sibling=map(Path,sys.argv[1:3])\n'
        'for path,pin in zip((entry,sibling),sys.argv[3:5],strict=True):\n'
        '    if path.resolve(strict=True)!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:\n'
        "        raise ValueError('frozen controller or membership bytes differ')\n"
        'sys.path.append(str(entry.parent))\n'
        "sys.argv=[str(entry),'--claim-sha256',sys.argv[5],sys.argv[6]]\n"
        "runpy.run_path(str(entry),run_name='__main__')\n"
    )
    return ('/usr/local/python3.12/bin/python3', '-I', '-S', '-B', '-u', '-c', bootstrap,
            str(root / HOST), str(root / MEMBERSHIP), claim_pin[HOST], claim_pin[MEMBERSHIP], claim_sha, mode)


def device_admission(snapshot, ledger):
    need(snapshot.get('snapshot_kind') == 'training80-dispatch-v1' and isinstance(snapshot.get('local_receipt'), dict),
         'the original received device snapshot is required')
    now = time.monotonic()
    _, nodes, fresh = fleet_view(snapshot, now, time.time() - now, ledger)
    need(fresh, 'device receipt expired or belongs to a different local clock')
    row, probe, receipt = nodes[NODE], snapshot['nodes'][NODE], snapshot['local_receipt']
    pcis = probe['PCIs']
    step = probe.get('device_probe_step')
    need(row['admission'] == 'clear' and row['parent'] == probe['parent'] == PARENT
         and row['zone'] == probe['zone'] == ZONE and probe.get('probe_error') is None
         and isinstance(step, str) and re.fullmatch(PARENT + r'\.[0-9]+', step)
         and len(set(pcis)) == len(pcis) == 8 and all(PCI.fullmatch(pci) for pci in pcis)
         and all(type(probe.get(key)) is int and probe[key] == 0
                 for key in ('kfd_process_count', 'own_device_fd_process_count'))
         and hex_value(receipt['request_id'], 32), 'original node lacks PCI, KFD and PID clearance')
    return dict(receipt, PCIs=sorted(pcis), device_probe_step=step)


def claim_local(queue, repo, board, approved, snapshot, binding):
    path = repo / INTENT
    local = repo / 'local-only'
    need(local.resolve(strict=True) == local and path.resolve().is_relative_to(local), 'original intent escaped local-only')
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        need(board.read_bytes() == approved, 'backlog changed after the publication check')
        rows = json.loads(approved)['items']
        retained = queue_holds(db)
        need(not path.parent.exists() and not path.parent.is_symlink() and
             all(row.get('direct_request') != REQUEST and
                 (row.get('direct_workload_SHA256') != WORKLOAD or row['state'] == 'RETIRED_PRELAUNCH')
                 for row in retained), 'original request or workload is already recorded')
        other, _, _ = read_original_holds(repo, rows, snapshot['nodes'], retained)
        need(not any(NODE in row['nodes'] for row in (*retained, *other.values())), 'the node has another original hold')
        need(device_admission(snapshot, repo / LEDGER) == binding['device_admission'],
             'device receipt changed before reservation')
        claim = reserve_handoff(queue, run, binding, INTENT)
        path.parent.mkdir(mode=0o700)
        write_once(path, dict(binding, queue_run=run))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return claim, path


def dispatch(publication, remote_repository, transport, *, stage_receipt, snapshot, repo=REPO,
             backlog=BACKLOG, database=QUEUE_DB, check_publication=published):
    """Invoke the existing finite argv transport after durable shared admission."""
    need(hex_value(publication, 40) and callable(transport), 'published revision and existing transport required')
    repo, board, database = (path.resolve(strict=True) for path in (repo, backlog, database))
    need(board == repo / BACKLOG.relative_to(REPO) and database == repo / QUEUE_DB.relative_to(REPO),
         'expected the shared planning database and backlog')
    expected_stage = dict(request=REQUEST, stage_manifest_SHA256=STAGE_SHA, members=9, actual_new_Slurm_steps=0)
    need(type(stage_receipt) is bytes and 0 < len(stage_receipt) <= 4096, 'original authenticated stage receipt required')
    staged = json.loads(stage_receipt)
    need(staged == expected_stage and type(staged['members']) is int and type(staged['actual_new_Slurm_steps']) is int,
         'target stage receipt differs')
    claim_raw = (repo / CLAIM).read_bytes()
    claim_ref = dict(path=CLAIM, bytes=len(claim_raw), SHA256=sha256(claim_raw).hexdigest())
    approved = board.read_bytes()
    rows = [row for row in json.loads(approved)['items'] if row['id'] == IDENT]
    need(len(rows) == 1 and all(rows[0].get(key) == value for key, value in backlog_identity(claim_ref).items()),
         'expected the exact shared backlog row')
    pins = scope(repo, claim_ref['SHA256'])['host_files']
    check_publication(repo, publication, sorted({ENTRY, CLAIM, board.relative_to(repo).as_posix(), *pins, *SUPPORT}))
    argv = host_argv(remote_repository, pins, claim_ref['SHA256'], '--dispatch-original')
    snapshot = deepcopy(snapshot)
    device = device_admission(snapshot, repo / LEDGER)
    binding = dict(candidate=IDENT, case=CASE, kind='validation', request=REQUEST, publication_commit=publication,
                   workload_SHA256=WORKLOAD, placement_by_parent={PARENT:{NODE:0}}, placement=PLACEMENT,
                   snapshot_payload_SHA256=device['payload_SHA256'], device_admission=device,
                   expires_local_monotonic=device['received_monotonic']+SNAPSHOT_MAX_AGE_S-device['roundtrip_s'],
                   claim=claim_ref, launcher=dict(path=HOST, SHA256=pins[HOST]), invocation={'kind':'pinned_argv'},
                   argv_SHA256=digest(argv), stage_receipt_SHA256=sha256(stage_receipt).hexdigest(),
                   stage_manifest_SHA256=STAGE_SHA, transport_timeout_seconds=320, replay_allowed=False,
                   counts_toward_training_reserve=False)
    binding['handoff_SHA256'] = digest(binding)
    queue = Queue(database)
    with closing(queue.db):
        claim, intent = claim_local(queue, repo, board, approved, snapshot, binding)
        run, handle = claim['run'], claim['handle']
        result = dict(request=REQUEST, queue_run=run, remote_outcome='NOT_ATTEMPTED',
                      transport_attempted=False, terminal_verified=False, replay_allowed=False)
        retire = False
        try:
            with intent.with_name('stdout.private').open('xb') as out, intent.with_name('stderr.private').open('xb') as err:
                try:
                    need(device_admission(snapshot, repo / LEDGER) == device, 'device receipt changed before transport')
                except ValueError:
                    retire = True
                    raise
                queue.submission(run, handle, unknown=True)
                result['remote_outcome'] = 'UNKNOWN'
                need(device_admission(snapshot, repo / LEDGER) == device, 'device receipt expired while arming')
                result['transport_attempted'] = True
                code = transport(argv, out, err, binding['transport_timeout_seconds']).returncode
                need(type(code) is int, 'finite transport did not return an exit code')
                result['transport_returncode'] = code
        except Exception as error:
            result['transport_error'] = type(error).__name__
        receipt = intent.with_name('transport.json')
        write_once(receipt, result)
        if retire:
            queue.retire_unattempted_direct(run, handle, sha256(receipt.read_bytes()).hexdigest())
    return result
