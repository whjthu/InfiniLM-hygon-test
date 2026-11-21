"""Hold the frozen GLM474 O01 CPU request before its sole transport invocation."""

from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import PurePosixPath
import time

from report_training80_backlog import (
    BACKLOG, QUEUE_DB, REPO, TRAINING80_LEDGER, has_original_intent, need, pin_file, queue_holds,
    read_original_holds,
)
from verify_training80_manifest import verify_package_members
from tuning_queue import DIRECT_STAGE, Queue, canonical, digest


PACKAGE = 'code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-host-compile-v1'
HOST_PUBLICATION = 'b06cec29ee58f2837c59a95554cec65931c5b85b'
CLAIM = dict(path=PACKAGE + '/CLAIM.json', SHA256='005a011e94cbe1fa5f6f11a735d1deb1afd9ee88656632656b48e091c576a5f9')
HOST = dict(path=PACKAGE + '/host_once.py', SHA256='490c8d242b28e9a1cbe7eb8e4c15c4bdcf0b9534c953c77f43cfb877d6341db4')
MANIFEST = dict(path=PACKAGE + '/SOURCE_FILES.json',
                SHA256='a5ee7e51b17193d39dfd5615ae17656c94bdde74f942435cf54b79bfd20df30b')
TASK = dict(path='coordination/tasks/glm-r474-fresh-kv-pack-gfx936-cpu-v1.json',
            SHA256='6c20deb38e5a726501f044588afc24047a07fd76e3701fe9f2104694f9da2723')
PROJECT = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'GLM474-fresh-kv-pack-gfx936-CPU-v1'
REQUEST = 'b9a7f0b78c566911ed40004fc906ce44'
PARENT, NODE, ZONE = '1103606', 'f12r2n13', 'O01'
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
IDENT = 'glm-r474-fresh-kv-pack-gfx936-host-compile-v1'
INTENT = f'local-only/{LABEL}-original-dispatch/intent.json'


def backlog_row():
    return dict(id=IDENT, case='O01.glm474-fresh-kv-pack-compile', kind='validation', state='blocked',
                priority=910, cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, attempt_receipts=[INTENT], evidence=TASK,
                next_action='Protect the original zero-GRES compile intent and reconcile it with the frozen HOST entry.')


def frozen_scope(repo):
    verify_package_members(repo, MANIFEST, required_files=(CLAIM, HOST))
    claim = json.loads(pin_file(repo, CLAIM, ('code/bw1000/',)))
    task = json.loads(pin_file(repo, TASK, ('coordination/tasks/',)))
    need(claim['request'] == REQUEST and claim['label'] == LABEL and claim['placement'] == PLACEMENT
         and claim['wall_seconds'] == 300 and claim['CPU_GO'] is True and claim['GPU_GO'] is False
         and claim['Model_GO'] is False and claim['replay_allowed'] is False
         and claim['host_entry_SHA256'] == HOST['SHA256'] and task['label'] == LABEL
         and claim['target_manifest_SHA256'] == task['fixed_package']['generated_target_manifest_SHA256'],
         'original frozen CPU compile scope differs')
    for path, sha in claim['shared_files'].items():
        pin_file(repo, dict(path=path, SHA256=sha), ('code/bw1000/', 'tools/', 'coordination/training/'))
    return claim


def original_argv(remote_repository):
    root = PurePosixPath(remote_repository)
    need(root.is_relative_to(PROJECT) and root.as_posix() == remote_repository and '..' not in root.parts
         and all(part.isascii() and all(ch.isalnum() or ch in '-_.' for ch in part) for part in root.parts[1:]),
         'expected a canonical target checkout inside the existing project')
    return ('/usr/local/python3.12/bin/python3', '-I', '-S', '-B', str(root / HOST['path']),
            '--claim-sha256', CLAIM['SHA256'])


def write_once(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def claim_local_intent(repo, backlog, database, record):
    intent = repo / INTENT
    local_root = repo / 'local-only'
    need(local_root.resolve(strict=True) == local_root and intent.resolve().is_relative_to(local_root),
         'original intent must use the canonical local-only directory')
    queue = Queue(database.resolve(strict=True))
    with closing(queue.db) as db:
        db.execute('BEGIN IMMEDIATE')
        try:
            run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
            need(not has_original_intent(repo, [INTENT]) and db.execute(
                'SELECT 1 FROM runs WHERE id=?', (run,)).fetchone() is None,
                'original request already claimed; reconcile it')
            jobs = json.loads(backlog.read_bytes())['items']
            rows = [job for job in jobs if job['id'] == IDENT]
            need(rows == [backlog_row()], 'register the exact original-intent row in the shared backlog first')
            ledger = json.loads(TRAINING80_LEDGER.read_bytes())
            need(NODE in ledger['groups'][ZONE]['allocations'][PARENT], 'frozen O01 placement differs')
            nodes = {node for group in ledger['groups'].values() for listed in group['allocations'].values() for node in listed}
            retained = queue_holds(db)
            direct, _, _ = read_original_holds(repo, jobs, nodes, retained)
            need(not any(NODE in item['nodes'] for item in (*retained, *direct.values())),
                 'another original request already holds the O01 node')
            intent.parent.mkdir()
            record = dict(record, queue_run=run)
            write_once(intent, record)
            spec = dict(case=rows[0]['case'], kind='validation', CPU=4, DCU=0, scope_SHA256=CLAIM['SHA256'])
            candidate = digest(dict(spec, request=REQUEST))
            handle = dict(request_id=REQUEST, intent=INTENT, nodes=[NODE], direct_kind='validation',
                          parent=PARENT, placement_by_parent={PARENT: {NODE: 0}})
            db.execute('INSERT INTO candidates(id,spec) VALUES(?,?)', (candidate, canonical(spec).decode()))
            db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)', (run, candidate, DIRECT_STAGE,
                'DISPATCHING', canonical(record).decode(), canonical([NODE]).decode(),
                canonical(handle).decode(), None, time.time()))
            queue.event(run, 'reserved', dict(parent=PARENT, nodes=[NODE], claim_SHA256=CLAIM['SHA256']))
            queue.event(run, 'dispatch_claimed', dict(handle=handle))
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return intent, run


def dispatch(publication, remote_repository, transport, *, repo=REPO, backlog=BACKLOG, database=QUEUE_DB):
    """Call the coordinator's existing authenticated argv transport once."""
    need(publication == HOST_PUBLICATION and callable(transport), 'verified HOST publication and existing transport required')
    repo = repo.resolve(strict=True)
    claim = frozen_scope(repo)
    argv = original_argv(remote_repository)
    record = dict(request=REQUEST, label=LABEL, publication=publication, source_publication=claim['published_source_commit'],
                  placement=PLACEMENT, kind='validation', claim_SHA256=CLAIM['SHA256'], host_entry_SHA256=HOST['SHA256'],
                  target_manifest_SHA256=claim['target_manifest_SHA256'], target_host_stage=claim['host_stage'],
                  target_intent=str(PurePosixPath(claim['host_result']).with_name('intent.json')),
                  target_host_result=claim['host_result'], argv=argv, local_intent=INTENT,
                  claimed_UTC=datetime.now(timezone.utc).isoformat(), transport_timeout_seconds=320,
                  remote_outcome='UNKNOWN', replay_allowed=False, counts_toward_training_reserve=False)
    intent, run = claim_local_intent(repo, backlog, database, record)
    outcome = dict(request=REQUEST, local_intent=INTENT, transport_returncode=None, transport_error=None,
                   remote_outcome='UNKNOWN', replay_allowed=False, terminal_verified=False, queue_run=run)
    with intent.with_name('stdout.private').open('xb') as stdout, intent.with_name('stderr.private').open('xb') as stderr:
        try:
            result = transport(argv, stdout, stderr, record['transport_timeout_seconds'])
            need(type(result.returncode) is int, 'transport did not return an exit code')
            outcome['transport_returncode'] = result.returncode
        except Exception as error:
            outcome['transport_error'] = type(error).__name__
    write_once(intent.with_name('transport.json'), outcome)
    return outcome


if __name__ == '__main__':
    print(json.dumps(backlog_row(), ensure_ascii=False, indent=2, allow_nan=False))
