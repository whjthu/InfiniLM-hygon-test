"""Recover only selected nodes proven absent from an already acknowledged standby envelope."""

import argparse
import ast
from base64 import b64decode, b64encode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
JOURNAL = LOCAL / "training80-1738-ordinary-renewal-dispatch"
FROZEN = LOCAL / "run-training80-2035-ordinary-renewal-once.py"
FROZEN_SHA256 = "62156f03554141703c55746165e4be8f7e41f4abb7b192d079bc764eb963e720"
EARLY_PUBLICATION = REPO / "tools/run_standby_renewal_with_pinned_publication.py"
EARLY_PUBLICATION_SHA256 = "e900334ac5be3a74fbde60f5c99457706dd9e802b7f749f0c375a1ecca48424e"
CONTROLLER_SHA256 = "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"
HELPER_SHA256 = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
PROCESS_LIMIT = 262144
HEX32 = re.compile(r"[a-f0-9]{32}")
HEX40 = re.compile(r"[a-f0-9]{40}")
HEX64 = re.compile(r"[a-f0-9]{64}")
NODE = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
STEP = re.compile(r"(?:1103606|1194900)\.[0-9]+")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
REMOTE = r'''
from base64 import b64encode
from hashlib import sha256
import json,re,sys
from pathlib import Path

request, expected, capture = sys.argv[1:]
assert re.fullmatch(r"[a-f0-9]{32}",request) and re.fullmatch(r"[a-f0-9]{64}",expected)
assert re.fullmatch(r"[a-f0-9]{32}",capture)
results=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
root=results/("training80-standby-compute-"+request)
assert root.is_dir() and not root.is_symlink()
records={}
for filename,limit in (("intent.json",8192),("entry.py",131072),("bridge.py",32768)):
    path=root/filename
    assert not path.is_symlink()
    with path.open("rb") as file: data=file.read(limit+1)
    assert 0<len(data)<=limit
    records[filename]=b64encode(data).decode("ascii")
    if filename=="entry.py": assert sha256(data).hexdigest()==expected
print(json.dumps({"request":request,"capture":capture,"files":records},sort_keys=True),flush=True)
'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read(path, limit):
    need(not path.is_symlink(), f"unexpected link: {path.name}")
    with path.open("rb") as file:
        value = file.read(limit + 1)
    need(0 < len(value) <= limit, f"unexpected size: {path.name}")
    return value


def exact_json(path, limit=65536):
    raw = read(path, limit)
    return json.loads(raw), digest(raw)


def identity_set(names, *, label):
    need(isinstance(names, list) and 1 <= len(names) <= 8
         and all(isinstance(node, str) and NODE.fullmatch(node) for node in names)
         and len(names) == len(set(names)), f"invalid {label} node list")
    return set(names)


def load_attempt(journal, local, wrapper, batch, expected_outcome):
    need(HEX32.fullmatch(wrapper) and HEX64.fullmatch(expected_outcome), "invalid wrapper or outcome binding")
    outcome, outcome_sha = exact_json(journal / f"{wrapper}-outcome.json")
    need(outcome_sha == expected_outcome and outcome['batch'] == batch and outcome['wrapper_invocation'] == wrapper
         and outcome['refill_rc'] == 0 and outcome['refill_attempted'] is True
         and outcome['no_GPU_send_confirmed'] is False and outcome['transport_observation'] == 'REFILL_PROCESS_RETURNED',
         "original wrapper is not one exact acknowledged compute-bridge call")
    stdout = read(journal / f"{wrapper}-stdout.private", 65536)
    need(digest(stdout) == outcome['stdout_SHA256'], "original complete stdout digest changed")
    admission, ack = [json.loads(line) for line in stdout.splitlines()]
    summary = admission['admission']
    admission_id, request = summary['request'], ack['request']
    need(HEX32.fullmatch(admission_id) and HEX32.fullmatch(request) and summary['fresh'] is True
         and ack['bridge_launch_pending'] is True
         and ack['control_label'] == f"training80-standby-compute-{request[:12]}",
         "original admission and bridge acknowledgements differ")
    prior_nodes = identity_set(outcome['terminal_ready_nodes'], label="reserved")
    parent = local / f"training80-replacement-admission-{admission_id}"
    intent, intent_sha = exact_json(parent / 'standby-intent.json')
    route, route_sha = exact_json(parent / 'bridge-route.json')
    original_admission, admission_sha = exact_json(parent / 'stdout.json', 2 * 1024 * 1024)
    installed_receipt, receipt_sha = exact_json(parent / 'receipt.json', 2 * 1024 * 1024)
    recorded = installed_receipt.pop('local_receipt')
    admission_payload_sha = digest(json.dumps(original_admission, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=False, allow_nan=False).encode())
    need(installed_receipt == original_admission and recorded['raw_response_SHA256'] == admission_sha
         and recorded['payload_SHA256'] == admission_payload_sha and recorded['request_id'] == admission_id,
         "original worker's canonical admission payload differs from the raw authenticated response")
    expected_nodes = identity_set(intent['nodes'], label="envelope")
    need(intent['request'] == route['request'] == request and intent['receipt_request'] == admission_id
         and intent['publication'] == route['publication'] and HEX40.fullmatch(intent['publication'])
         and route['standby_source'] == intent['payload_SHA256'] and HEX64.fullmatch(intent['payload_SHA256'])
         and HEX64.fullmatch(route['bridge_SHA256']) and intent['replay_allowed'] is False and intent['synthetic'] is True
         and expected_nodes < prior_nodes and intent['requested_cards'] == ack['requested_cards'] == 8 * len(expected_nodes),
         "original local attempt is not a bound strict subset of its reserved nodes")
    need(admission_sha == summary['snapshot_SHA256'] and original_admission['request_id'] == admission_id
         and original_admission['complete'] is True and original_admission['device_probe_scope'] == 'PRIORITY_ONLY'
         and identity_set(original_admission['requested_priority_nodes'], label="original admitted scope") == prior_nodes,
         "original complete device receipt is not the exact acknowledged target scope")
    return dict(wrapper=wrapper,outcome_SHA256=outcome_sha,stdout_SHA256=outcome['stdout_SHA256'],
                request=request,admission_id=admission_id,intent=intent,intent_SHA256=intent_sha,
                bridge_SHA256=route['bridge_SHA256'],bridge_route_SHA256=route_sha,
                original_admission=original_admission,admission_SHA256=admission_payload_sha,raw_admission_SHA256=admission_sha,
                original_receipt_SHA256=receipt_sha,
                terminal_receipt=outcome['terminal_receipt'],prior_nodes=prior_nodes,expected_nodes=expected_nodes)


def bound_exclusions(attempt, evidence, capture):
    need(evidence['rc'] == 0 and evidence['authenticated'] is True and evidence['command_sent'] is True,
         "original envelope was not read from the authenticated remote host")
    need(evidence['original_request'] == attempt['request'] and evidence['capture'] == capture,
         "read-only receipt belongs to a different capture or request")
    stdout = b64decode(evidence['stdout'], validate=True)
    need(0 < len(stdout) <= PROCESS_LIMIT and digest(stdout) == evidence['stdout_SHA256'],
         "complete authenticated read stdout digest differs")
    packet = json.loads(stdout)
    files = packet['files']
    need(packet['request'] == attempt['request'] and packet['capture'] == capture
         and set(files) == {'intent.json', 'entry.py', 'bridge.py'},
         "authenticated read crossed original bridge scope")
    remote_intent, entry, bridge = (b64decode(files[name], validate=True)
                                    for name in ('intent.json','entry.py','bridge.py'))
    original = json.loads(remote_intent)
    local = attempt['intent']
    need(original['request'] == attempt['request'] and original['entry_SHA256'] == digest(entry) == local['payload_SHA256']
         and original['bridge_SHA256'] == attempt['bridge_SHA256'] == digest(bridge)
         and original['parent'] == '1103606' and original['node'] == 'f12r2n13' and original['replay_allowed'] is False,
         "original bridge intent and acknowledged executable envelope differ")
    declaration, separator, controller = entry.partition(b'\n')
    need(separator and declaration.startswith(b'INPUT=') and digest(controller) == CONTROLLER_SHA256,
         "remote input is not the original finite controller envelope")
    values = ast.literal_eval(declaration.removeprefix(b'INPUT=').decode('ascii'))
    nodes = values['nodes']
    need(isinstance(nodes, dict) and set(nodes) == attempt['expected_nodes']
         and values['request'] == attempt['request'] and values['publication'] == local['publication']
         and values['admission_SHA256'] == attempt['admission_SHA256'] and values['seconds'] == 10800,
         "original literal GPU node map or controller budget differs")
    for node, allocation in nodes.items():
        actual = attempt['original_admission']['nodes'][node]
        need(set(allocation) == {'parent', 'pci'} and allocation['parent'] in {'1103606', '1194900'}
             and len(allocation['pci']) == len(set(allocation['pci'])) == 8
             and all(PCI.fullmatch(pci) for pci in allocation['pci']) and actual['parent'] == allocation['parent']
             and actual['PCIs'] == allocation['pci'] and actual['device_probe_observed'] is True
             and actual['membership_valid'] is True and actual['active_steps'] == []
             and actual['KFD_clean'] is True and actual['PID_clean'] is True and actual['ports_clean'] is True
             and actual['kfd_process_count'] == actual['own_device_fd_process_count'] == 0,
             "original GPU allocation or PCI identity differs from authenticated device admission")
    return attempt['prior_nodes'] - set(nodes)


def capture_once(attempt, capture, path):
    helper = LOCAL / 'run-minicpm-sdpa-target-asset-preflight-original.py'
    need(digest(read(helper, 131072)) == HELPER_SHA256, "existing authenticated helper differs")
    auth = runpy.run_path(str(helper))['ssh'](REPO.parent)
    command = shlex.join(['/usr/local/python3.12/bin/python3','-I','-S','-B','-c',REMOTE,
                          attempt['request'], attempt['intent']['payload_SHA256'], capture])
    with path.open('xb') as out:
        try:
            p = subprocess.run([auth[0], '-v', *auth[1:], command], stdin=subprocess.DEVNULL, capture_output=True, timeout=25)
            rc, stdout, stderr = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as error:
            rc, stdout, stderr = None, error.stdout or b'', error.stderr or b''
        need(len(stdout) <= PROCESS_LIMIT, "oversized original envelope reply")
        data = (json.dumps(dict(rc=rc, original_request=attempt['request'], capture=capture,
                    authenticated=b'Authenticated to' in stderr, command_sent=b'Sending command:' in stderr,
                    stdout=b64encode(stdout).decode('ascii'), stdout_SHA256=digest(stdout), stderr_SHA256=digest(stderr),
                    observed_epoch=time.time()),sort_keys=True) + '\n').encode()
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return digest(data), rc


def bind_terminal_markers(attempt, rows, batch, journal, local, frozen):
    name = attempt['terminal_receipt']
    need(re.fullmatch(rf'training80-2035-ordinary-terminal-batch-{batch}-[a-f0-9]{{32}}\.json', name),
         "original terminal receipt outside exact wave scope")
    terminal, terminal_sha = exact_json(local / name)
    need(attempt['prior_nodes'] <= {row['node'] for row in frozen['released_nodes'](rows, terminal)},
         "original terminal receipt does not qualify every original reserved node")
    for row in rows:
        node, step = row['node'], row['previous']
        if node not in attempt['prior_nodes']:
            continue
        original = attempt['original_admission']['nodes'][node]
        need(original['parent'] == row['parent'] and original['membership_valid'] is True,
             "original current target-parent identity differs from the frozen predecessor")
        matching = []
        for slot in range(3):
            path = journal / f'{node}-{step}-{slot}.json'
            if path.is_file():
                marker, _ = exact_json(path)
                if marker['wrapper_invocation'] == attempt['wrapper']:
                    matching.append(marker)
        need(len(matching) == 1 and matching[0]['terminal_receipt_SHA256'] == terminal_sha
             and matching[0]['terminal_receipt'] == name and matching[0]['batch'] == batch
             and matching[0]['node'] == node and matching[0]['predecessor'] == step,
             "original per-node journal marker and terminal proof differ")
    return terminal_sha


def reserve_proven(journal, node, step, intent, proof, store, did_not_send):
    if node not in proof['excluded'] or proof['predecessors'].get(node) != step:
        return False
    permitted = False
    for attempt in range(3):
        current = journal / f"{node}-{step}-{attempt}.json"
        if permitted:
            try:
                store(current, dict(intent, original_envelope_exclusion=proof['binding']))
            except FileExistsError:
                pass
            else:
                return True
        if not current.is_file():
            return False
        marker, _ = exact_json(current)
        prior = marker['wrapper_invocation']
        if not HEX32.fullmatch(prior) or marker['node'] != node or marker['predecessor'] != step:
            return False
        outcome_file = journal / f"{prior}-outcome.json"
        if not outcome_file.is_file():
            return False
        outcome, outcome_sha = exact_json(outcome_file)
        if (outcome['wrapper_invocation'] != prior or node not in outcome['terminal_ready_nodes']
                or outcome['terminal_receipt'] != marker['terminal_receipt'] or outcome['batch'] != marker['batch']):
            return False
        if prior == proof['binding']['wrapper_invocation']:
            if outcome_sha != proof['binding']['original_outcome_SHA256'] or permitted:
                return False
            permitted = True
        else:
            if outcome['no_GPU_send_confirmed'] is not True:
                return False
            stdout = read(journal / f"{prior}-stdout.private", 65536)
            if digest(stdout) != outcome['stdout_SHA256'] or not did_not_send(stdout, outcome['refill_rc']):
                return False
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--wrapper', required=True)
    parser.add_argument('--outcome-sha', required=True)
    parser.add_argument('--capture-id', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--capture', action='store_true')
    action.add_argument('--evidence-sha')
    parser.add_argument('--dispatch', action='store_true')
    args = parser.parse_args()
    need(not args.dispatch or args.evidence_sha is not None, "capture and GPU send must be separate actions")
    need(HEX32.fullmatch(args.capture_id), "independently unique capture id required")
    need(digest(read(FROZEN, 65536)) == FROZEN_SHA256, "frozen wave wrapper differs")
    base = runpy.run_path(str(FROZEN))
    batches = json.loads(base['pinned'](base['PLAN'], base['PLAN_SHA']))['candidate_batches_in_chronological_budget_order']
    need(1 <= args.batch <= len(batches), "batch is outside the frozen plan")
    rows = batches[args.batch-1]['rows']
    needed = {row['node']:row['previous'] for row in rows}
    attempt = load_attempt(JOURNAL, LOCAL, args.wrapper, args.batch, args.outcome_sha)
    need(attempt['prior_nodes'] <= set(needed) and all(STEP.fullmatch(needed[n]) for n in attempt['prior_nodes']),
         "original wrapper nodes are outside this exact wave batch")
    terminal_sha = bind_terminal_markers(attempt, rows, args.batch, JOURNAL, LOCAL, base)
    evidence_path = LOCAL / f"training80-ordinary-remote-envelope-{attempt['request']}-{args.capture_id}.json"
    if args.capture:
        evidence_pin, rc = capture_once(attempt, args.capture_id, evidence_path)
        print(json.dumps(dict(file=evidence_path.name, SHA256=evidence_pin, rc=rc, new_GRES=0)))
        raise SystemExit(int(rc != 0))
    evidence, actual = exact_json(evidence_path, 400000)
    need(actual == args.evidence_sha, "original read evidence bytes differ")
    excluded = bound_exclusions(attempt, evidence, args.capture_id)
    proof = dict(excluded=excluded,predecessors=needed,binding=dict(wrapper_invocation=args.wrapper,
         original_outcome_SHA256=args.outcome_sha,original_stdout_SHA256=attempt['stdout_SHA256'],
         original_entry_SHA256=attempt['intent']['payload_SHA256'], original_terminal_SHA256=terminal_sha,
         original_device_admission_SHA256=attempt['admission_SHA256'],original_device_raw_SHA256=attempt['raw_admission_SHA256'],
         original_local_receipt_SHA256=attempt['original_receipt_SHA256'], capture_id=args.capture_id, authenticated_read_SHA256=actual))
    if not args.dispatch:
        print(json.dumps(dict(excluded_nodes=sorted(excluded),reserved_nodes=sorted(attempt['prior_nodes']),
             in_original_GPU_envelope=sorted(attempt['expected_nodes']),replay_allowed=False,new_GRES=0),sort_keys=True))
        return
    need(digest(read(EARLY_PUBLICATION, 65536)) == EARLY_PUBLICATION_SHA256, "pinned early publication route differs")
    run = runpy.run_path(str(EARLY_PUBLICATION))['run_batch']
    reserve = lambda journal, node, step, intent: reserve_proven(
        journal, node, step, intent, proof, base['store'], base['did_not_send'])
    print(json.dumps(run(args.batch, True, reserve_override=reserve),sort_keys=True))


if __name__ == '__main__':
    main()
