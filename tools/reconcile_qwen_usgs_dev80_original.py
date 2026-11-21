"""Read the fixed Dev80 request without starting it or checking generation outputs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess

from read_training_slurm_membership import snapshot


REPO = Path(__file__).resolve().parents[1]
RESULTS = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LEDGER = Path('coordination/training/training80-active-node-task-map-v1.json')
CLAIM = Path('code/bw1000/training-qwen-usgs-dev80-base-adapter-v1/CLAIM.json')
CLAIM_SHA = '9d8fefbf5bbca0791432635ab1bbd5f2158740e8bdf34b4484b1159c3decf1eb'
AUTH_RECEIPT = Path('results/bw1000/2026-11-20/training-qwen-usgs-dev80-pre-exec-auth-rejected.json')
AUTH_SHA = 'db60ef522d6f1be69bdf0db9df9464c128a27332b20f61a604fa16b029b9e806'
PUBLICATION = '94812e070df7ed45d7141eed148d433b428a9782'
LABEL = 'training-qwen-usgs-dev80-base-adapter-v1'
REQUEST = 'bb61591025d94fdaba421ac15d5d4fa7'
PARENT = '1103606'
NODES = {'base':'f13r1n10','domain':'f13r1n11'}
PHASES = ('pre','gpu','post')
LABELS = {f'{LABEL}-{arm}-{phase}':node for arm,node in NODES.items() for phase in PHASES}
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED',
            'BOOT_FAIL','DEADLINE','REVOKED'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def raw_file(path, limit, *, wanted=None):
    need(path.is_file() and not path.is_symlink(), 'expected an original plain evidence file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and (wanted is None or sha256(raw).hexdigest() == wanted),
         'original evidence size or SHA-256 differs')
    return raw


def pre_exec_receipt(repo):
    raw = raw_file(repo / AUTH_RECEIPT, 32768, wanted=AUTH_SHA)
    receipt = json.loads(raw)
    need(receipt['request'] == REQUEST and receipt['publication'] == PUBLICATION and receipt['transport_rc'] == 255
         and receipt['phase'] == 'SSH_USER_AUTHENTICATION_REJECTED_BEFORE_REMOTE_EXEC'
         and receipt['actual_GPU_submitted_by_client'] is False and receipt['replay_allowed'] is False,
         'expected the original pre-exec authentication failure')
    claim = json.loads(raw_file(repo / CLAIM,32768,wanted=CLAIM_SHA))
    need(claim['request'] == REQUEST and claim['label'] == LABEL and claim['parent'] == PARENT
         and claim['placement'] == NODES and claim['replay_allowed'] is False, 'original published claim differs')
    return sha256(raw).hexdigest()


def namespace(base):
    folder = base / LABEL
    try:
        metadata = folder.lstat()
    except FileNotFoundError:
        return dict(present=False)
    result = dict(present=True, directory=stat.S_ISDIR(metadata.st_mode), original_intent_observed=False)
    if not result['directory'] or folder.resolve(strict=True) != folder:
        result['identity_error'] = 'unexpected_namespace_type'
        return result
    intent = folder / 'intent.json'
    try:
        intent.lstat()
    except FileNotFoundError:
        return result
    try:
        raw = raw_file(intent,65536)
        data = json.loads(raw)
        result['intent_SHA256'] = sha256(raw).hexdigest()
        result['original_intent_observed'] = data.get('request') == REQUEST and data.get('publication') == PUBLICATION
    except (OSError, ValueError, TypeError) as error:
        result['identity_error'] = type(error).__name__
    return result


def scan_accounting(raw):
    need(0 < len(raw) <= 2*1024**2, 'original Slurm accounting missing or too large')
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    need(all(len(row) == 7 for row in rows), 'unexpected original accounting columns')
    need(sum(row[0] == PARENT for row in rows) == 1, 'original parent not uniquely covered')
    expected = {}
    for row in rows:
        handle,label,state,_,_,node,_ = row
        if label not in LABELS:
            continue
        need(label not in expected and re.fullmatch(PARENT+r'\.\d+',handle) and state.split() and node == LABELS[label],
             'duplicate or foreign original Slurm phase')
        expected[label] = row
    handles = [row[0] for row in expected.values()]
    need(len(handles) == len(set(handles)), 'original phase labels share a numbered step')
    return dict(raw_SHA256=sha256(raw).hexdigest(), rows=expected)


def live_accounting():
    return subprocess.check_output(['sacct','-n','-P','-j',PARENT,'-S','2026-11-19T00:00:00','-o',
        'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'],timeout=8)


def live_membership(repo):
    ledger = json.loads(raw_file(repo / LEDGER,131072))
    parents = {job['JobId'] for job in ledger['jobs']}
    groups = {parent:{zone:group['allocations'][parent] for zone,group in ledger['groups'].items()} for parent in parents}
    report = snapshot(groups, owner_uid=os.getuid())
    need(report['distinct_nodes'] == 80 and report['allocated_DCU'] == 640 and all(
         parent['state'] == 'RUNNING' and parent['ledger_exact'] and parent['step_locations_exact']
         and parent['owned_by_observer'] is True for parent in report['parents'].values()),
         'original two-parent allocation is not verified')
    parent = report['parents'][PARENT]
    coverage = {node:item['state'] for group in parent['coverage_by_group_and_node'].values()
                for node,item in group.items() if node in NODES.values()}
    need(set(coverage) == set(NODES.values()), 'expected both original evaluation nodes')
    return dict(observed_epoch=report['observed_epoch'],node_membership=coverage)


def reconcile(repo, base, *, account=live_accounting, membership=live_membership):
    proof = pre_exec_receipt(repo)
    base = base.resolve(strict=True)
    anchor = base.stat()
    need(stat.S_ISDIR(anchor.st_mode), 'original results directory is unavailable')
    first = dict(namespace=namespace(base), accounting=scan_accounting(account()))
    members = membership(repo)
    last = dict(accounting=scan_accounting(account()), namespace=namespace(base))
    finished = base.stat()
    need((anchor.st_dev,anchor.st_ino) == (finished.st_dev,finished.st_ino), 'original results directory identity changed')
    ever_present = first['namespace']['present'] or last['namespace']['present']
    rows = first['accounting']['rows'] | last['accounting']['rows']
    absent = not ever_present and not rows
    coverage = members['node_membership']
    clear = set(coverage) == set(NODES.values()) and all(value == 'NO_NUMBERED_STEP' for value in coverage.values())
    if absent and clear:
        state = 'ABSENCE_VERIFIED_AFTER_PRE_EXEC_AUTH_FAILURE'
    elif absent:
        state = 'ABSENCE_VERIFIED_OTHER_STEP_OR_UNCLEAR_NODE_HOLD'
    elif not rows:
        state = 'ORIGINAL_NAMESPACE_PRESENT_ACCOUNTING_UNKNOWN'
    elif any(row[2].split()[0] not in TERMINAL for row in rows.values()):
        state = 'ORIGINAL_ACCOUNTING_NOT_TERMINAL'
    else:
        state = 'ORIGINAL_TERMINAL_ROWS_REQUIRE_INDEPENDENT_CLEANUP'
    return dict(state=state, request=REQUEST, publication=PUBLICATION, parent=PARENT, node_placement=NODES,
                original_replay_allowed=False, original_not_submitted_verified=absent,
                original_hold_eligible_for_removal=absent and clear, current_pci_KFD_port_admission=False,
                training_READY=False, pre_exec_receipt_SHA256=proof, membership=members, first=first,last=last)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    def expired(*_):
        raise TimeoutError('original Dev80 read-only reconciliation deadline elapsed')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,45)
    report = reconcile(REPO,RESULTS)
    print(json.dumps(report,allow_nan=False),flush=True)
    return int(not report['original_hold_eligible_for_removal'])


if __name__ == '__main__':
    raise SystemExit(main())
