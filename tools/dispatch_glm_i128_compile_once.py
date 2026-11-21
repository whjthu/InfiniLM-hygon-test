"""Hold the original GLM routed-I128 compile scopes before authenticated dispatch."""

import base64
from contextlib import closing
from hashlib import sha256
import json
import math
import runpy
import time

from dispatch_glm474_fresh_kv_cpu_once import PROJECT, write_once
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, REPO, TRAINING80_LEDGER, hex_value, need, pin_file, queue_holds, read_original_holds
from tuning_queue import DIRECT_STAGE, Queue, canonical, digest


PACKAGE = 'code/bw1000/GLM-C1-layer-observer-r474'
AUDIT = dict(path=PACKAGE+'/routed-i128-cpu-artifact-audit-v1/audit_once.py',
             SHA256='5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46')
ARMS = ('gate', 'down')


def scope(repo, arm):
    need(arm in ARMS, 'unknown original GLM routed compile arm')
    pin_file(repo, AUDIT, ('code/bw1000/',))
    audit = runpy.run_path(str(repo/AUDIT['path']))
    plan, variants, manifest_sha, source = audit['load'](arm, repo)
    label = plan['label']
    request = digest(dict(label=label, parent=plan['parent'], node=plan['node'], manifest_SHA256=manifest_sha))[:32]
    ident = f'glm-i128-{arm}-rows-gfx936-host-compile-v1'
    intent = f'local-only/{label}/intent.json'
    row = dict(id=ident, case=f'I02.glm-i128-{arm}-compile', kind='validation', state='blocked',
               cpu_prerequisite=True, cpu_request=request, placement={plan['node']: 0},
               priority=920+ARMS.index(arm), attempt_receipts=[intent],
               evidence=dict(path=f'coordination/tasks/glm-i128-{arm}-rows-gfx936-cpu-v1.json',
                             SHA256=audit['BRANCHES'][arm][3]),
               next_action='Reconcile the original label and use the coordinated zero-GRES compile entry once.')
    return dict(arm=arm, label=label, request=request, parent=plan['parent'], node=plan['node'],
                manifest_SHA256=manifest_sha, variants=list(variants), source=source, intent=intent, row=row)


def published_inputs(repo, expected, publication):
    source = expected['source']
    raw = (source/'SOURCE_FILES.json').read_bytes()
    manifest = json.loads(raw)['files_SHA256']
    need(sha256(raw).hexdigest() == expected['manifest_SHA256'], 'original compile manifest drift')
    task = expected['row']['evidence']
    pin_file(repo, task, ('coordination/tasks/',))
    names = [*manifest,'SOURCE_FILES.json']
    published(repo, publication, [*((source/name).relative_to(repo).as_posix() for name in names),task['path']])
    return {name:(source/name).read_bytes() for name in names}, expected['manifest_SHA256']


def arm_packet(expected, publication, files, mf):
    raw = files['SOURCE_FILES.json']
    manifest = json.loads(raw)['files_SHA256']
    need(mf == expected['manifest_SHA256'] == sha256(raw).hexdigest() and
         set(files) == set(manifest)|{'SOURCE_FILES.json'} and
         all(sha256(files[name]).hexdigest() == pin for name,pin in manifest.items()),
         'source bytes do not match the original compile')
    payload = dict(publication=publication, request=expected['request'], manifest_SHA256=mf,
                   files={name:base64.b64encode(contents).decode('ascii') for name,contents in files.items()})
    body = ('PAYLOAD = '+repr(payload)+'\n').encode('ascii') + files['receive.py']
    need(len(body) <= 160000, 'original compile packet exceeds transport bound')
    return body


def claim(queue, repo, backlog_path, expected, intent_record):
    intent = repo/expected['intent']
    local = repo/'local-only'
    need(local.resolve(strict=True) == local and intent.resolve().is_relative_to(local),
         'compile intent must remain in the canonical local-only tree')
    run = digest(dict(stage=DIRECT_STAGE, request=expected['request']))
    node, parent = expected['node'], expected['parent']
    groups = json.loads(TRAINING80_LEDGER.read_bytes())['groups']
    need(node in groups['I02']['allocations'][parent], 'frozen compile is outside the owning I02 placement')
    nodes = {name for group in groups.values() for member in group['allocations'].values() for name in member}
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        jobs = json.loads(backlog_path.read_bytes())['items']
        matches = [item for item in jobs if item['id'] == expected['row']['id']]
        need(matches == [expected['row']], 'shared backlog must register this exact original compile')
        need(db.execute('SELECT 1 FROM runs WHERE id=?', (run,)).fetchone() is None
             and not intent.parent.exists() and not intent.parent.is_symlink(),
             'original local compile namespace exists; reconcile without replay')
        holds, _, _ = read_original_holds(repo, jobs, nodes)
        need(not any(node in row['nodes'] for row in (*queue_holds(db), *holds.values())),
             'another original request already holds the compile node')
        intent.parent.mkdir()
        write_once(intent, dict(intent_record, queue_run=run))
        spec = dict(case=expected['row']['case'], kind='validation', CPU=4, DCU=0,
                    scope_SHA256=expected['manifest_SHA256'])
        candidate = digest(dict(spec, request=expected['request']))
        ticket = digest(dict(request=expected['request'], node=node, parent=parent,
                             manifest_SHA256=expected['manifest_SHA256']))
        handle = dict(request_id=expected['request'], intent=expected['intent'], nodes=[node], parent=parent,
                      direct_kind='validation', placement_by_parent={parent:{node:0}}, handoff_SHA256=ticket)
        db.execute('INSERT INTO candidates(id,spec) VALUES(?,?)', (candidate,canonical(spec).decode()))
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)', (run,candidate,DIRECT_STAGE,'DISPATCHING',
                   canonical(intent_record).decode(),canonical([node]).decode(),canonical(handle).decode(),None,time.time()))
        queue.event(run,'reserved',dict(parent=parent,nodes=[node],dispatch_ticket=ticket))
        queue.event(run,'dispatch_claimed',dict(handle=handle))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return run,handle,intent


def read_owned_process(path, expected, publication):
    with path.open('rb') as stream:
        raw = stream.read(65537)
    need(0 < len(raw) <= 65536, 'missing or oversized original launch receipt')
    value = json.loads(raw)
    epoch = value['epoch']
    intended = PROJECT/'results'/(expected['label']+'-intent.json')
    need(value['request'] == expected['request'] and value['publication'] == publication and
         value['manifest_SHA256'] == expected['manifest_SHA256'] and value['GPU'] is False and
         type(epoch) in (int,float) and math.isfinite(epoch) and epoch > 0 and value['deadline'] == epoch+300 and
         all(type(value[key]) is int and value[key]>0 for key in ('pid','pgid','startticks')) and
         value['pid'] == value['pgid'] and value['intent'] == str(intended) and
         isinstance(value['host'],str) and bool(value['host']),
         'original srun owner receipt differs')
    return value


def dispatch(arm, publication, transport, *, repo=REPO, backlog=BACKLOG, database=QUEUE_DB,
             publication_inputs=published_inputs):
    """The caller supplies its already authorized finite body transport."""
    need(hex_value(publication,40) and callable(transport), 'published source and existing transport required')
    repo = repo.resolve(strict=True)
    expected = scope(repo,arm)
    files, manifest = publication_inputs(repo,expected,publication)
    body = arm_packet(expected,publication,files,manifest)
    cap = 60
    record = dict(request=expected['request'], publication=publication, manifest_SHA256=manifest,
                  body_SHA256=sha256(body).hexdigest(), parent=expected['parent'], node=expected['node'],
                  GPU=0, transport_timeout_seconds=cap, replay_allowed=False)
    queue = Queue(database.resolve(strict=True))
    with closing(queue.db):
        run,handle,intent = claim(queue,repo,backlog,expected,record)
        out = intent.with_name('stdout.private')
        result = dict(request=expected['request'], queue_run=run, remote_outcome='UNKNOWN',
                      root_numbered_step=None, terminal_verified=False, replay_allowed=False)
        try:
            with out.open('xb') as stdout, intent.with_name('stderr.private').open('xb') as stderr:
                rc = transport(body,stdout,stderr,cap).returncode
            need(type(rc) is int, 'expected transport exit code')
            result['rc'] = rc
            if rc == 0:
                receipt = read_owned_process(out,expected,publication)
                result.update(remote_outcome='OWNED_SRUN_STARTED',receipt=receipt)
        except Exception as error:
            result['transport_error'] = type(error).__name__
        write_once(intent.with_name('transport.json'),result)
        # srun PID or a transport failure alone cannot establish the numbered step.
        queue.submission(run,handle,unknown=True)
    return result
