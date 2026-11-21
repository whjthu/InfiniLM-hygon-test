"""Reserve and dispatch one published four-node Graph64 replication."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from package import prepare, source
from scope import LABEL, PREFERRED, REQUEST, RUN, placement

sys.path.insert(0, str(REPO / 'tools'))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, TRAINING80_LEDGER, fleet_view, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


BASE = HERE.relative_to(REPO)
CLAIM = BASE / 'ROOT_CLAIM.json'
INTENT = Path('local-only') / (RUN + '-original') / 'intent.json'
CASE = 'O01.resident-bf16-graph64-32card-v1'


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def node_devices(snapshot, repo):
    """Accept the four published placements from one fresh ownership observation."""
    now = time.monotonic()
    _, nodes, fresh = fleet_view(snapshot, now, time.time()-now, repo / TRAINING80_LEDGER.relative_to(REPO))
    need(fresh, 'fresh CPU-only device inventory required')
    selected = {}
    for parent, names in PREFERRED.items():
        for name in names:
            sample = snapshot['nodes'][name]
            ownership = nodes[name]
            need(ownership['admission'] == 'clear' and (ownership['parent'], ownership['zone']) == (parent,'O01')
                 and sample['membership_valid'] is sample['PID_clean'] is sample['KFD_clean'] is True
                 and not sample['active_steps'] and sample['probe_error'] is None,
                 'published O01 node is not clear: '+name)
            selected[name] = dict(parent=parent, zone='O01', PCI=sorted(sample['PCIs']),
                                  before_CPU_step=sample['device_probe_step'])
    placement(selected)
    return selected


def ship(publication, snapshot, transport, *, repo=REPO, check_publication=published):
    """Preserve the original claim on uncertain transport or a partial shard result."""
    repo = repo.resolve(strict=True)
    board = repo / BACKLOG.relative_to(REPO)
    path = repo / QUEUE_DB.relative_to(REPO)
    raw_claim = (repo / CLAIM).read_bytes()
    claim = json.loads(raw_claim)
    approved = {parent:{node:8 for node in names} for parent,names in PREFERRED.items()}
    need(claim == dict(label=LABEL, request=REQUEST, operator_GO=True, Model_GO=False,
                       placement=approved, planned_cards=32, new_GPU_runs=0, replay_allowed=False),
         'published root claim differs')
    verified_refs, _ = source(repo)
    paths = {p.as_posix() for p in verified_refs.values()}
    paths.update(p.as_posix() for p in (BASE/'package.py',BASE/'launch.py',CLAIM,
        Path('results/bw1000/2026-11-22/O01-BF16-graph64-original674-eight-card-paired-measurement.json'),
        BACKLOG.relative_to(REPO),Path('tools/prepare_training80_dispatch_handoff.py'),
        Path('tools/report_training80_backlog.py'),Path('tools/tuning_queue.py')))
    check_publication(repo, publication, sorted(paths))
    admission = node_devices(snapshot,repo)
    receipt = snapshot['local_receipt']
    cutoff = receipt['received_monotonic'] + SNAPSHOT_MAX_AGE_S - receipt['roundtrip_s']
    packet = dict(candidate=LABEL, case=CASE, kind='operator', request=REQUEST,
                  publication_commit=publication, workload_SHA256=sha256(raw_claim).hexdigest(),
                  snapshot_payload_SHA256=receipt['payload_SHA256'], snapshot_raw_SHA256=receipt['raw_response_SHA256'],
                  placement_by_parent=approved, snapshot_request=receipt['request_id'],
                  before=admission, replay_allowed=False, expires_local_monotonic=cutoff)
    packet['handoff_SHA256'] = digest(packet)
    directory = (repo / INTENT).parent
    run = digest(dict(stage=DIRECT_STAGE,request=REQUEST))
    queue = Queue(path)
    with closing(queue.db):
        db=queue.db
        db.execute('BEGIN IMMEDIATE')
        try:
            need(not directory.exists() and not directory.is_symlink(), 'original launch exists; reconcile')
            schedule=plan(snapshot,json.loads(board.read_bytes()),repo=repo,holds=queue_holds(db))
            protected=set(schedule['queue_held_nodes'])
            for item in schedule['original_launch_holds'].values():
                protected.update(item['nodes'])
            for row in schedule['rows']:
                if row['dispatch_state'] == 'proposed_for_root' or (row['kind']=='training' and
                        row['dispatch_state']=='awaiting_original_queue_release'):
                    protected.update(row['placement'])
            need(not protected.intersection(admission) and node_devices(snapshot,repo)==admission
                 and time.monotonic()<cutoff, 'fresh selected nodes conflict with earlier claims')
            local=reserve_handoff(queue,run,packet,INTENT.as_posix())
            directory.mkdir(mode=0o700)
            save(repo / INTENT,dict(packet,queue_run=run))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        handle=local['handle']
        result=dict(request=REQUEST,queue_run=run,transport_attempted=False,remote_outcome='NOT_ATTEMPTED',
                    terminal_verified=False,planned_cards=32,replay_allowed=False)
        try:
            with (directory/'stdout.private').open('xb') as out,(directory/'stderr.private').open('xb') as err:
                need(node_devices(snapshot,repo)==admission and time.monotonic()<cutoff,
                     'admission expired before the original epoch')
                original, input_bytes, refs=prepare(publication,time.time(),admission,receipt['payload_SHA256'],repo=repo)
                need(set(refs).issubset(paths) and time.monotonic()<cutoff, 'publication or admission changed')
                save(directory/'original-plan.json',original)
                queue.submission(run,handle,unknown=True)
                result.update(transport_attempted=True,remote_outcome='UNKNOWN')
                result['transport_returncode']=transport(input_bytes,out,err,215).returncode
        except Exception as error:
            result['transport_error']=type(error).__name__
        terminal=directory/'transport.json'
        save(terminal,result)
        if not result['transport_attempted']:
            queue.retire_unattempted_direct(run,handle,sha256(terminal.read_bytes()).hexdigest())
    return result
