"""Reserve the original SDXL FP32 attention diagnostic before contacting the cluster."""

from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from dispatch_glm474_fresh_kv_cpu_once import write_once
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import (BACKLOG, QUEUE_DB, REPO, TRAINING80_LEDGER, fleet_view,
                                      hex_value, need, pin_file, queue_holds, read_original_holds)
from sample_training80_busy import PCI
from tuning_queue import DIRECT_STAGE, Queue, SNAPSHOT_MAX_AGE_S, digest


PACKAGE = 'code/bw1000/training-sdxl-sdpa-adjoint-fp32-v1'
CASE = 'T06.sft.operator-sdpa-adjoint'
LABEL = 'training-sdxl-sdpa-adjoint-fp32-v1'
IDENT = 'sdxl-fp32-sdpa-adjoint-O01-first-update'
REQUEST = 'a93d41a0d31140f1ad99bb105a024e99'
INTENT = f'local-only/{LABEL}-original-dispatch/intent.json'
PARENT, NODE, ZONE = '1087656', 'f12r3n20', 'O01'
TARGET = f'/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/{LABEL}-{REQUEST}'
LEDGER = TRAINING80_LEDGER.relative_to(REPO)
BUILD = dict(path=PACKAGE+'/TARGET_BUILD_FILES.json', SHA256='d1f65e34455d62afabc488188a66ce827732545a0f6b7b1cbc1cc092e1b69b7f')
TASK = 'coordination/tasks/training-sdxl-sdpa-adjoint-fp32-target-v1.json'
MANIFEST = 'abd85b0954a71987f80da4273cc29cce6727832fff08a2d16deb65a5a3783833'
BOOTSTRAP = '''import sys
sys.path.insert(0,sys.argv[1])
from launch_once import bootstrap
from render_target import render
script,manifest=bootstrap(sys.argv[2],render())
if manifest!=sys.argv[3]:raise ValueError('target manifest differs')
sys.stdout.buffer.write(script)
'''


def scope(repo, job):
    need(job['id'] == IDENT and job['case'] == CASE and job['state'] == 'blocked'
         and job['kind'] == 'validation' and job['placement'] == {NODE: 1}
         and job['direct_request'] == REQUEST and job['attempt_receipts'] == [INTENT]
         and job['evidence']['path'] == TASK, 'register the exact SDXL direct scope in the shared backlog')
    task = json.loads(pin_file(repo, job['evidence'], ('coordination/tasks/',)))
    build = json.loads(pin_file(repo, BUILD, ('code/bw1000/',)))
    need(task['claim']['path'] == PACKAGE+'/CLAIM.json' and task['launch']['path'] == PACKAGE+'/launch_once.py',
         'original SDXL entry paths differ')
    claim = json.loads(pin_file(repo, task['claim'], ('code/bw1000/',)))
    need(task['request'] == claim['request'] == REQUEST and task['label'] == claim['label'] == LABEL
         and task['target_build']['SHA256'] == BUILD['SHA256']
         and task['generated_target_manifest_SHA256'] == build['generated_target_manifest_SHA256'] == MANIFEST
         and task['placement'] == {key: claim['hygon'][key] for key in ('parent','node','zone','DCU')}
         == dict(parent=PARENT,node=NODE,zone=ZONE,DCU=1) and claim['max_invocations'] == 1
         and claim['GPU_GO'] is True and claim['formal_training_GO'] is False and claim['replay_allowed'] is False,
         'original SDXL target claim differs')
    published_paths = {TASK, BUILD['path'], LEDGER.as_posix(), 'tools/dispatch_sdxl_sdpa_adjoint_once.py',
                       'tools/prepare_training80_dispatch_handoff.py', 'tools/report_training80_backlog.py'}
    for name, wanted in build['files'].items():
        need(Path(name).name == name, 'invalid original target member')
        ref = dict(path=f'{PACKAGE}/{name}', SHA256=wanted['SHA256'])
        need(len(pin_file(repo, ref, ('code/bw1000/',))) == wanted['bytes'], 'source byte count differs')
        published_paths.add(ref['path'])
    return task, published_paths


def device_admission(snapshot, ledger):
    need(isinstance(snapshot, dict) and isinstance(snapshot.get('local_receipt'), dict),
         'the original received device snapshot is required')
    now = time.monotonic()
    _, nodes, fresh = fleet_view(snapshot, now, time.time() - now, ledger)
    need(fresh, 'SDXL device receipt expired or belongs to a different local clock')
    slot, probe, receipt = nodes[NODE], snapshot['nodes'][NODE], snapshot['local_receipt']
    pcis, step = probe['PCIs'], probe.get('device_probe_step')
    need(slot['admission'] == 'clear' and slot['zone'] == ZONE and slot['parent'] == PARENT
         and probe.get('probe_error') is None and isinstance(step, str) and re.fullmatch(PARENT+r'\.[0-9]+', step)
         and all(PCI.fullmatch(pci) for pci in pcis) and hex_value(receipt['request_id'], 32)
         and all(type(probe.get(key)) is int and probe[key] == 0
                 for key in ('kfd_process_count','own_device_fd_process_count')),
         'SDXL node lacks same-parent PCI, KFD and PID clearance')
    return {**receipt, 'device_probe_step':step, 'PCIs':sorted(pcis)}


def prepare_claim(queue, repo, board, approved_board, snapshot, binding):
    intent = repo/INTENT
    local = repo/'local-only'
    need(local.resolve(strict=True)==local and intent.resolve().is_relative_to(local), 'original intent escaped local-only')
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        need(board.read_bytes() == approved_board, 'shared backlog changed before reservation')
        jobs = json.loads(approved_board)['items']
        retained = queue_holds(db)
        need(not intent.parent.exists() and not intent.parent.is_symlink() and all(
            row.get('direct_request') != REQUEST for row in retained), 'original SDXL request already has a local scope')
        need(all(row.get('direct_workload_SHA256') != binding['workload_SHA256'] or
                 row['state'] == 'RETIRED_PRELAUNCH' for row in retained), 'the SDXL workload already has a recorded scope')
        holds, _, _ = read_original_holds(repo, jobs, snapshot['nodes'], retained)
        need(not any(NODE in row['nodes'] for row in (*retained, *holds.values())),
             'another original request holds the SDXL O01 node')
        need(device_admission(snapshot, repo/LEDGER) == binding['device_admission'],
             'SDXL device receipt changed before reservation')
        claim = reserve_handoff(queue, run, binding, INTENT)
        intent.parent.mkdir(mode=0o700)
        write_once(intent, dict(binding, queue_run=run))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return claim, intent


def dispatch(publication, transport, *, snapshot, repo=REPO, backlog=BACKLOG, database=QUEUE_DB,
             check_publication=published):
    """Call the existing bounded stdin transport after a verified and durable O01 reservation."""
    need(hex_value(publication, 40) and callable(transport), 'published revision and existing transport required')
    repo, board, database = (path.resolve(strict=True) for path in (repo, backlog, database))
    need(board == repo/BACKLOG.relative_to(REPO) and database == repo/QUEUE_DB.relative_to(REPO),
         'expected the shared planning database and backlog')
    approved = board.read_bytes()
    [job] = [row for row in json.loads(approved)['items'] if row['id'] == IDENT]
    task, members = scope(repo, job)
    check_publication(repo, publication, sorted(members | {board.relative_to(repo).as_posix()}))
    script = subprocess.check_output([sys.executable, '-I','-S','-B','-c',BOOTSTRAP,str(repo/PACKAGE),
                                      publication,MANIFEST], cwd=repo, timeout=12, stdin=subprocess.DEVNULL)
    need(0 < len(script) <= 128*1024, 'invalid original staged script size')
    snapshot = deepcopy(snapshot)
    device = device_admission(snapshot, repo/LEDGER)
    binding = dict(candidate=IDENT, kind='validation', case=CASE, request=REQUEST, publication_commit=publication,
                   workload_SHA256=digest(dict(case=CASE, manifest=MANIFEST)), placement_by_parent={PARENT:{NODE:1}},
                   snapshot_payload_SHA256=device['payload_SHA256'], device_admission=device,
                   expires_local_monotonic=device['received_monotonic']+SNAPSHOT_MAX_AGE_S-device['roundtrip_s'],
                   claim=task['claim'], launcher=task['launch'], invocation={'kind':'pinned_stdin'},
                   target=TARGET, source_manifest_SHA256=MANIFEST, body_SHA256=sha256(script).hexdigest(),
                   transport_timeout_seconds=120, replay_allowed=False, counts_toward_training_reserve=False)
    binding['handoff_SHA256'] = digest(binding)
    queue = Queue(database)
    with closing(queue.db):
        claim, intent = prepare_claim(queue, repo, board, approved, snapshot, binding)
        run, handle = claim['run'], claim['handle']
        result = dict(request=REQUEST, queue_run=run, remote_outcome='NOT_ATTEMPTED',
                      transport_attempted=False, terminal_verified=False, replay_allowed=False)
        retire = False
        try:
            with intent.with_name('stdout.private').open('xb') as out, intent.with_name('stderr.private').open('xb') as err:
                try:
                    need(device_admission(snapshot, repo/LEDGER) == device, 'SDXL device receipt changed before transport')
                except ValueError:
                    retire = True
                    raise
                queue.submission(run, handle, unknown=True)
                result['remote_outcome'] = 'UNKNOWN'
                need(device_admission(snapshot, repo/LEDGER) == device, 'SDXL device receipt expired while arming transport')
                result['transport_attempted'] = True
                code = transport(script, out, err, binding['transport_timeout_seconds']).returncode
                need(type(code) is int, 'finite transport did not return an exit code')
                result['transport_returncode'] = code
        except Exception as error:
            result['transport_error'] = type(error).__name__
        receipt = intent.with_name('transport.json')
        write_once(receipt, result)
        if retire:
            queue.retire_unattempted_direct(run, handle, sha256(receipt.read_bytes()).hexdigest())
    return result
