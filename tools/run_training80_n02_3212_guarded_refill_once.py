"""Publish the n02 successor, read the exact incumbent, and gate one standby send."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import time
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-n02-3212-guarded-standby-renewal-v1'
WRAPPER = ROOT / 'tools/run_training80_n02_3212_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
PREP = ROOT / 'tools/prepare_training80_n02_3212_guarded_successor.py'
PREP_SHA = '36037543745d1786a8f4906078beb6ed6343b339553892f2c1527352af986e22'
ROLE = ROOT / 'coordination/training/training80-replacement1194900-node-task-map-v1.json'
ROLE_SHA = '11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d'
REPORT = ROOT / 'tools/report_training80_backlog.py'
REPORT_SHA = '476c32fbe306119ba25632dac39d1a0f6fdce92bc24c4d30f9ce88851edc31c0'
BACKLOG = ROOT / 'coordination/training/training80-backlog-v1.json'
QUEUE = LOCAL / 'tuning/queue.sqlite3'
NODE, PARENT = 'f13r1n02', '1103606'
REQUEST = '1d5660a0998a439cb7526ae749b2575b'
OLD, OLD_REQUEST = '1103606.3212', 'fb95a1f66e744a34924e3d56a212f1b8'
INTENT, SEND_TOKEN, REPLY = (LOCAL / f'{STEM}-{kind}.json' for kind in ('intent', 'send', 'reply'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read_file(path, expected=None, limit=262144):
    require(path.resolve(strict=True) == path and path.is_relative_to(ROOT), 'exact configured source path differs')
    with path.open('rb') as file:
        raw = file.read(limit + 1)
    require(0 < len(raw) <= limit and (expected is None or digest(raw) == expected), 'exact receipt or source bytes differ')
    return raw


def role_owner():
    data = json.loads(read_file(ROLE, ROLE_SHA))
    priority = data['groups']['I01']
    assigned = [i for i, group in data['groups'].items() if NODE in group['allocations'].get(PARENT, [])]
    require(assigned == ['I01'] and priority['model'] == 'Kimi-K3' and priority['owner'] == '/root', 'current inference owner differs')
    return priority['owner']


def check_inference_holds(holds, jobs, direct, errors):
    pending_states = {'ready', 'queued', 'submitting', 'submitted', 'running', 'active', 'reserved', 'reconcile', 'unknown'}
    require(not any(NODE in row['nodes'] for row in holds)
            and not any(NODE in row['nodes'] for row in direct.values()), 'existing live or unresolved work hold on n02')
    for job in jobs:
        if NODE not in job['placement']:
            continue
        require(job['id'] not in errors and job['state'] not in pending_states, 'n02 ready or unresolved original work has priority')
        if job['kind'] == 'inference':
            require(job['state'] in {'blocked', 'completed', 'terminal', 'retired'}, 'unsupported inference state on n02')


def live_inference_priority():
    owner = role_owner()
    read_file(REPORT, REPORT_SHA)
    # The existing report imports other tools from its own directory.
    sys.path.insert(0, str(REPORT.parent))
    try:
        report = runpy.run_path(str(REPORT))
        holds = report['read_queue_holds'](QUEUE)
        jobs = json.loads(read_file(BACKLOG))['items']
        known_nodes = set(report['allocation'](report['TRAINING80_LEDGER'].read_bytes())[2])
        known_nodes.update(node for row in jobs for node in row['placement'])
        direct, errors, _ = report['read_original_holds'](ROOT, jobs, known_nodes, holds)
    finally:
        sys.path.pop(0)
    require(NODE in known_nodes, 'current n02 inference placement could not be verified')
    check_inference_holds(holds, jobs, direct, errors)
    return dict(role='I01', model='Kimi-K3', owner=owner, node=NODE, parent=PARENT,
                state='RELEASED_OR_UNRESERVED', authority_verified=True)


def load(*, draft=False):
    read_file(PREP, PREP_SHA)
    prep = runpy.run_path(str(PREP))
    material = prep['load']()
    raw = read_file(CLAIM if not draft or CLAIM.exists() else DRAFT)
    claim = json.loads(raw)
    require((claim['scope'], claim['state'], claim['node'], claim['parent'], claim['request'], claim['previous'],
             claim['previous_request'], claim['previous_zero_GRES'], claim['max_invocations'], claim['replay_allowed'])
            == (STEM, 'PREPARED_ONLY_ACTUAL_TERMINAL_REQUIRED', NODE, PARENT, REQUEST, OLD, OLD_REQUEST,
                '1103606.3211', 1, False)
            and (claim['worker_budget_epoch'], claim['prepare_epoch']) == (prep['BUDGET'], prep['PREP'])
            and claim['previous_worker_pins'] == material[-1] and claim['launcher'] == WRAPPER.relative_to(ROOT).as_posix()
            and claim['runtime_PREP_SHA256'] == PREP_SHA and claim['inference_role_SHA256'] == ROLE_SHA
            and claim['priority_reader_SHA256'] == REPORT_SHA and claim['inference_role'] == 'I01.Kimi-K3'
            and claim['resources'] == dict(CPU=8, GPU=8, memory_MiB=16384, Slurm_cap_seconds=11100, preemptible=True),
            'exact generated n02 scope differs')
    role_owner()
    return prep, material, claim, digest(raw)


def verify_publication(revision, prep, material):
    require(Path(__file__).resolve() == WRAPPER and re.fullmatch(r'[a-f0-9]{40}', revision), 'live send requires exact installed wrapper and publication')
    previous, base = material[:2]
    previous['pinned_publication'](base, revision)
    git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + ROOT.as_posix()]
    for path in (WRAPPER, CLAIM, ROLE, REPORT, PREP):
        ref = revision + ':' + path.relative_to(ROOT).as_posix()
        got = subprocess.check_output([*git,'cat-file','blob',ref], cwd=ROOT, stderr=subprocess.PIPE, timeout=8)
        require(got == read_file(path), 'published n02 scope, role or priority source differs')
    # Check the published PREP source and its pinned private observation inputs before launch.
    read_file(PREP, PREP_SHA)


def read_path(read_id):
    require(isinstance(read_id, str) and re.fullmatch(r'[a-f0-9]{32}', read_id) and int(read_id,16) > 0 and read_id != REQUEST,
            'a new exact read identifier is required')
    return LOCAL / f'{STEM}-predecessor-{read_id}.json'


def observe_predecessor(prep, material, read_id):
    require(time.time() >= prep['PREP'], 'preparation window has not begun')
    config = prep['read_programs'](material, REQUEST, read_id)
    base = material[1]
    helper = runpy.run_path(str(base['HELPER']))
    command = shlex.join(['/usr/local/python3.12/bin/python3','-I','-S','-B','-'])
    path = read_path(read_id)
    source = config['read_predecessor'].encode()
    with path.open('x', encoding='utf8') as receipt:
        start = time.monotonic()
        anchor = time.time() - start
        try:
            answer = subprocess.run([*helper['ssh'](ROOT.parent), command], input=source, capture_output=True, timeout=25)
            rc, stdout, stderr = answer.returncode, answer.stdout, answer.stderr
        except subprocess.TimeoutExpired as error:
            rc, stdout, stderr = None, error.stdout or b'', error.stderr or b''
        except (OSError, ValueError, AssertionError, StopIteration, subprocess.SubprocessError) as error:
            rc, stdout, stderr = None,b'',str(error).encode()
        now = time.monotonic()
        body = None
        if rc == 0 and 0 < len(stdout) <= 8192 and 0 <= now-start <= 25:
            try:
                selected = json.loads(stdout)
                if (selected['read_id'], selected['predecessor'], selected['next_request'], selected['new_Slurm_operations']) == (read_id, OLD, REQUEST, 0):
                    body = selected
            except (KeyError, TypeError, ValueError):
                pass
        json.dump(dict(rc=rc, result=body, received_monotonic=now, roundtrip_s=now-start, boot_epoch_anchor=anchor,
                       stdout_SHA256=digest(stdout), stderr_SHA256=digest(stderr), source_SHA256=digest(source), replay_allowed=False),
                  receipt, sort_keys=True, allow_nan=False)
    return dict(receipt=path.name, SHA256=digest(read_file(path)), state=(body or {}).get('state','UNCONFIRMED'), replay_allowed=False)


def dispatch(prep, material, revision, claim_sha, terminal_arg):
    require(not any(os.path.lexists(path) for path in (INTENT,SEND_TOKEN,REPLY)), 'original request already has a marker, reconcile it')
    verify_publication(revision, prep, material)
    read_id, receipt_sha = terminal_arg
    observed = json.loads(read_file(read_path(read_id), receipt_sha))
    priority = live_inference_priority()
    config = prep['guarded_successor'](material, REQUEST, read_id, observed, revision, priority,
                                        monotonic=time.monotonic(), wall=time.time())
    require(0 <= time.monotonic() - observed['received_monotonic'] <= prep['CAP'], 'terminal proof expired during local authority reads')
    base = material[1]
    helper = runpy.run_path(str(base['HELPER']))
    sender = runpy.run_path(str(base['TRANSPORT']))
    argv = helper['ssh'](ROOT.parent)
    require(0 <= time.monotonic() - observed['received_monotonic'] <= prep['CAP'], 'terminal proof expired before local send reservation')
    data = dict(request=REQUEST, publication=revision, terminal_read_id=read_id, terminal_read_SHA256=receipt_sha,
                exact_program_SHA256=digest(config['launch'].encode()), claim_SHA256=claim_sha, GPU_requested=8,
                inference_role=priority['role'], replay_allowed=False)
    config['scope']['save'](INTENT,data)
    answer = sender['send'](argv, request=REQUEST, program=config['launch'], data=b'', send_token=SEND_TOKEN,
                            authority_seconds=18, outcome_seconds=82)
    config['scope']['save'](REPLY,dict(data,transport=answer))
    return dict(receipt=REPLY.name,SHA256=digest(read_file(REPLY)),guard_state=(answer.get('response') or {}).get('state'),
                request=REQUEST,replay_allowed=False)


def observe_successor(prep, material, claim):
    base = material[1]
    old = base['observe']
    observer = FunctionType(old.__code__, dict(old.__globals__, STEM=STEM, INTENT=INTENT, TOKEN=SEND_TOKEN))
    # Observation is attached to the prior send namespace. This never renders a new launch.
    config = prep['read_programs'](material, REQUEST, 'f' * 32)
    result = observer(claim,config)
    body = result.get('observed') or {}
    return dict(file=result['file'],SHA256=result['SHA256'],state=body.get('state'),step=body.get('step'),
                verified_progress=body.get('actual_DCU_progress'),replay_allowed=False)


def fixture_holds():
    check_inference_holds([],[],{},{})
    for holds, jobs, direct, errors in (
            ([{'nodes':[NODE]}], [], {}, {}), ([], [], {'known':{'nodes':[NODE]}}, {}),
            ([], [{'id':'I01','kind':'inference','placement':[NODE],'state':'ready'}], {}, {}),
            ([], [{'id':'I01','kind':'inference','placement':[NODE],'state':'recovering'}], {}, {}),
            ([], [{'id':'I01','kind':'inference','placement':[NODE],'state':'blocked'}], {}, {'I01':'incomplete direct marker'})):
        try:
            check_inference_holds(holds,jobs,direct,errors)
        except ValueError:
            continue
        raise ValueError('inference priority fixture admitted a live, reserved or unknown scope')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--check-publication')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args = parser.parse_args()
    require(bool(args.publication) == bool(args.terminal_receipt), 'publication requires exact fresh predecessor receipt')
    prep, material, claim, claim_sha = load(draft=args.check)
    if args.check:
        fixture_holds()
        result = dict(prep['offline_check'](), claim_SHA256=claim_sha, request=REQUEST, authority_fixture_pass=1,
                      authority_fixture_vetoes=5, active_terminal_known=False)
    elif args.check_publication:
        verify_publication(args.check_publication,prep,material)
        result = dict(publication=args.check_publication, request=REQUEST, claim_SHA256=claim_sha, Slurm=0, SCNet=0)
    elif args.observe_predecessor:
        result = observe_predecessor(prep,material,args.observe_predecessor)
    elif args.observe_original:
        result = observe_successor(prep,material,claim)
    else:
        result = dispatch(prep,material,args.publication,claim_sha,args.terminal_receipt)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(error_class=type(error).__name__, detail_SHA256=digest(str(error).encode()))))
        raise SystemExit(1)
