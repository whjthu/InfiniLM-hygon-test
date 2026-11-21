"""Bind the InternVL CPU input-audit callback to a durable shared-node reservation."""

from contextlib import closing, contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import ModuleType

from read_training_slurm_membership import allocated_dcu
from report_training80_backlog import (
    BACKLOG, QUEUE_DB, REPO, TRAINING80_LEDGER, hex_value, need, pin_file, queue_holds, read_original_holds,
)
from tuning_queue import DIRECT_STAGE, Queue, canonical, digest
from verify_training80_manifest import verify_package_members


PACKAGE = 'code/bw1000/training-internvl-USGS72-native-CPU-v1'
PACKAGE_MANIFEST = '9f1119107708470075fd61bc39a82c1d0445c7d93d5ca0c0a02acc05fee56fbb'
REQUEST = '8110b01cd24846f8b6fcd08f0abda5a0'
IDENT = 'internvl-usgs72-native-input-cpu'
PARENT, NODE, GROUP = '1087656', 'a14r3n04', 'T03'
TARGET_COMMAND = ('/usr/local/python3.12/bin/python3', '-I', '-S', '-B', '-')
SCOPE = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-USGS72-native-CPU-v1'
INTENT = 'local-only/training-internvl-USGS72-native-CPU-v1-original-dispatch/intent.json'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def scope(repo, manifest_sha):
    need(manifest_sha == PACKAGE_MANIFEST, 'exact adopted Native72 source manifest required')
    mf = dict(path=f'{PACKAGE}/SOURCE_FILES.json', SHA256=manifest_sha)
    verify_package_members(repo, mf)
    path = repo/PACKAGE
    claim_raw = (path/'CLAIM.json').read_bytes()
    claim, plan = [json.loads((path/name).read_bytes()) for name in ('CLAIM.json', 'PLAN.json')]
    read_claim = json.loads((path/'TERMINAL_READ_CLAIM.json').read_bytes())
    need(claim['request_id'] == plan['request_id'] == read_claim['request_id'] == REQUEST
         and claim['root_GO'] is claim['CPU_GO'] is read_claim['root_GO'] is read_claim['READ_GO'] is True
         and claim['GPU_GO'] is claim['Model_GO'] is claim['training_GO'] is claim['READY'] is False
         and (plan['parent'], plan['node'], plan['CPU'], plan['GPU'], plan['whole_seconds'], plan['RUN'])
         == (PARENT, NODE, 8, 0, 300, SCOPE), 'CPU-only adopted InternVL scope differs')
    return dict(bytes=len(claim_raw), SHA256=sha256(claim_raw).hexdigest())


@contextmanager
def local_common(folder):
    """Keep the candidate's unqualified common.py import local to this entrypoint."""
    path = folder/'common.py'
    module = ModuleType('common')
    module.__file__ = str(path)
    prior = sys.modules.get('common')
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    sys.modules['common'] = module
    try:
        yield
    finally:
        if prior is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = prior


def intent_record(repo):
    path = repo/INTENT
    need(path.resolve(strict=True) == path and path.is_file(), 'canonical original launch intent required')
    with path.open('r+b') as stream:
        raw = stream.read(8193)
        need(0 < len(raw) <= 8192, 'bounded launch intent required')
        os.fsync(stream.fileno())
    return json.loads(raw), sha256(raw).hexdigest()


def register(queue, repo, backlog_path, expected, context, *, wall=time.time):
    original, original_sha = intent_record(repo)
    keys = ('epoch', 'deadline', 'request_id', 'publication', 'payload_SHA256')
    need(all(original[key] == context[key] for key in keys) and context['request_id'] == REQUEST
         and hex_value(context['publication'], 40) and hex_value(context['payload_SHA256'], 64)
         and original['MF_SHA256'] == context['source_manifest_SHA256'] == PACKAGE_MANIFEST
         and original['GPU'] == 0 and original['target_command'] == list(TARGET_COMMAND)
         and context['target_command'] == TARGET_COMMAND and context['scope'] == SCOPE
         and context['deadline'] == context['epoch']+300, 'transport does not bind the original CPU request')
    need(0 <= wall()-context['epoch'] < 290, 'original CPU transport epoch expired')
    roster = json.loads(TRAINING80_LEDGER.read_bytes())
    nodes = {node for group in roster['groups'].values() for group_nodes in group['allocations'].values() for node in group_nodes}
    need(NODE in roster['groups'][GROUP]['allocations'][PARENT], 'frozen InternVL node is outside the current ledger')
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        items = json.loads(backlog_path.read_bytes())['items']
        rows = [row for row in items if row['id'] == IDENT]
        need(len(rows) == 1 and all(rows[0].get(key) == value for key,value in {
            'kind': 'validation', 'state': 'blocked', 'cpu_prerequisite': True,
            'placement': {NODE: 0}, 'cpu_request': REQUEST, 'attempt_receipts': [INTENT],
        }.items()), 'missing exact CPU prerequisites row')
        pin_file(repo, rows[0]['evidence'], ('results/bw1000/', 'coordination/tasks/'))
        run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
        need(db.execute('SELECT id FROM runs WHERE id=?', (run,)).fetchone() is None,
             'original InternVL request already claimed; reconcile')
        retained = queue_holds(db)
        direct, _, _ = read_original_holds(repo, items, nodes, retained)
        need(direct.get(IDENT, {}).get('nodes') == [NODE], 'original intent is not visible to normal dispatch planning')
        competing = [*retained, *(value for key,value in direct.items() if key != IDENT)]
        need(not any(NODE in item['nodes'] for item in competing), 'another request already holds the InternVL node')
        handle = dict(request_id=REQUEST, intent=INTENT, nodes=[NODE], parent=PARENT, direct_kind='validation',
                      placement_by_parent={PARENT:{NODE:0}})
        spec = dict(case=rows[0]['case'], kind='validation', CPU=8, DCU=0, scope=expected['SHA256'])
        candidate = digest(dict(spec, request=REQUEST))
        evidence = dict(context, original_intent_SHA256=original_sha, source_claim=expected, replay_allowed=False)
        need(wall() < context['epoch']+290, 'original CPU transport epoch expired during claim')
        db.execute('INSERT INTO candidates(id,spec) VALUES(?,?)', (candidate, canonical(spec).decode()))
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)', (run, candidate, DIRECT_STAGE, 'DISPATCHING',
            canonical(evidence).decode(), canonical([NODE]).decode(), canonical(handle).decode(), None, wall()))
        queue.event(run, 'reserved', dict(parent=PARENT, nodes=[NODE], intent_SHA256=original_sha))
        queue.event(run, 'dispatch_claimed', dict(handle=handle))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return run


def dispatch(publication, manifest_sha, transport_start, *, repo=REPO, backlog=BACKLOG, database=QUEUE_DB):
    """The supplied existing authenticated transport receives the original launcher's bounded context."""
    need(hex_value(publication, 40) and callable(transport_start), 'published scope and existing authenticated callback required')
    repo = repo.resolve(strict=True)
    expected = scope(repo,manifest_sha)
    invoked = None

    def guarded(context):
        nonlocal invoked
        need(invoked is None and context['publication'] == publication, 'callback may only carry the original request once')
        queue = Queue(database.resolve(strict=True))
        with closing(queue.db):
            invoked = register(queue, repo, backlog, expected, context)
        return transport_start(context)

    folder = repo/PACKAGE
    with local_common(folder):
        launcher = runpy.run_path(str(folder/'launch_once.py'))['main']
        rc = launcher(publication, manifest_sha, transport_start=guarded)
    return dict(request=REQUEST, queue_run=invoked, launcher_rc=rc, terminal_verified=False)


def finish_payload(original, readout, reader_sha, claim):
    row = readout['actual_accounting']
    need(readout['independent_metadata_consumed'] is True and readout['request_id'] == original['request_id'] == REQUEST
         and readout['publication'] == original['publication'] and readout['epoch'] == original['epoch']
         and readout['deadline'] == original['deadline'] and readout['claim'] == claim
         and readout['source_manifest']['SHA256'] == original['MF_SHA256'] == PACKAGE_MANIFEST
         and readout['actual_CPU_steps'] == 1 and readout['replay_allowed'] is False
         and hex_value(reader_sha, 64), 'independent result does not bind the original CPU audit')
    need(isinstance(row, list) and len(row) == 7 and re.fullmatch(PARENT+r'\.\d+', row[0])
         and (row[1],row[5]) == (Path(SCOPE).name,NODE) and row[2].split()[0] in TERMINAL
         and allocated_dcu(row[6]) == 0, 'original numbered CPU step is not terminal')
    for name in ('srun-control','container-control'):
        item = readout['controllers'][name]
        post = item['post']
        need(item['held_child_reaped'] is True and item['owned_processes_remaining'] == 0 and not item['cleanup_errors']
             and post['same_host'] is post['same_boot'] is post['held_child_reaped'] is True
             and post['controller_device_FDs'] == [] and post['owned_descendants_remaining'] == 0
             and post['owned_device_FD_holders'] == {}, 'original independent post-clean evidence is incomplete')
    planned, completed = readout['planned_rows'], readout['completed_rows']
    failed = (readout['native_call_counts'] or {}).get('failed_rows',0)
    need(type(planned) is type(completed) is type(failed) is int and planned == 72 and failed >= 0
         and 0 <= completed <= planned-failed, 'original item denominator differs')
    passed = readout['qualification_passed'] is True
    need(not passed or (row[2:4] == ['COMPLETED','0:0'] and completed == planned),
         'successful native audit lacks all original observations')
    return dict(request_id=REQUEST,intent=INTENT,parent=PARENT,placement_by_parent={PARENT:{NODE:0}},
                terminal=True,post_clean=True,evidence_SHA256=reader_sha,planned=planned,completed=completed,
                failed=failed,missing=planned-completed-failed,rejected=0,canceled=0,passed=passed,
                qualification='validation',keeper_passed=None)


def finish(reader_path: Path, reader_sha, *, repo=REPO, database=QUEUE_DB):
    """Consume the private independent receipt and release only the original CPU scope."""
    repo = repo.resolve(strict=True)
    claim = scope(repo,PACKAGE_MANIFEST)
    original, _ = intent_record(repo)
    with reader_path.open('rb') as source:
        data = source.read(16385)
    need(0 < len(data) <= 16384 and sha256(data).hexdigest() == reader_sha, 'bounded private reader receipt differs')
    result = finish_payload(original,json.loads(data),reader_sha,claim)
    queue = Queue(database.resolve(strict=True))
    run = digest(dict(stage=DIRECT_STAGE,request=REQUEST))
    with closing(queue.db):
        queue.finish(run,result)
    return dict(request=REQUEST,queue_run=run,state='TERMINAL',passed=result['passed'],evidence_SHA256=reader_sha)
