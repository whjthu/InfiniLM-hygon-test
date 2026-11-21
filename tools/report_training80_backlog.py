"""Report 80-node coverage and prioritize evidence-bound work for the dispatcher."""

import argparse
import ast
from collections import Counter
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from tuning_queue import ACTIVE, SNAPSHOT_MAX_AGE_S, TRAINING80_LEDGER, digest, training80_inventory
from verify_training80_manifest import verify_package_members


REPO = Path(__file__).resolve().parents[1]
BACKLOG = REPO / 'coordination' / 'training' / 'training80-backlog-v1.json'
QUEUE_DB = REPO / 'local-only' / 'tuning' / 'queue.sqlite3'
KINDS = ('training', 'validation', 'inference', 'operator', 'data')
STATES = ('blocked', 'reconcile', 'ready', 'submitted', 'terminal')
PRIORITY = {kind: index for index, kind in enumerate(KINDS)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def hex_value(value, length):
    return isinstance(value, str) and re.fullmatch(f'[0-9a-f]{{{length}}}', value) is not None


def file_bytes(repo, relative, prefixes):
    need(isinstance(relative, str) and relative.startswith(prefixes) and '\\' not in relative,
         'evidence is outside its permitted repository directories')
    path = (repo / relative).resolve()
    need(any(path.is_relative_to(repo / prefix) for prefix in prefixes) and path.is_file(),
         'evidence did not resolve to a known repository file')
    need(path.stat().st_size <= 1024 * 1024, 'evidence document exceeds the bounded size')
    return path.read_bytes()


def pin_file(repo, reference, prefixes):
    raw = file_bytes(repo, reference['path'], prefixes)
    need(hex_value(reference['SHA256'], 64) and sha256(raw).hexdigest() == reference['SHA256'],
         'candidate evidence SHA256 differs')
    return raw


def invocation(repo, job):
    """Check entry-point syntax while leaving imports and transport to the coordinator."""
    repo = repo.resolve()
    launch = job['launch']
    path = launch['launcher']['path']
    raw = pin_file(repo, launch['launcher'], ('code/bw1000/', 'tools/'))
    tree = ast.parse(raw, filename=path)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    if len(functions) != 1:
        return None
    signature = functions[0].args
    if signature.posonlyargs or signature.kwonlyargs or signature.vararg or signature.kwarg:
        return None
    arguments = tuple(arg.arg for arg in signature.args)
    if arguments == ('publication', 'transport') and not signature.defaults:
        return dict(kind='coordinator_callback', entry='main', arguments=[launch['publication_commit']],
                    injected_keyword='transport')
    if (arguments == ('publication', 'request', 'transport') and len(signature.defaults) == 1 and
            isinstance(signature.defaults[0], ast.Constant) and signature.defaults[0].value is None):
        return dict(kind='coordinator_callback', entry='main',
                    arguments=[launch['publication_commit'], launch['request']], injected_keyword='transport')
    if arguments == ('publication', 'claim_sha256', 'request') and not signature.defaults:
        flags = ('--publication', '--claim-sha256', '--request-id')
        cli_args = {node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' and node.args and
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
        if cli_args == set(flags):
            return dict(kind='authenticated_host_python', arguments=[flags[0], launch['publication_commit'],
                        flags[1], launch['claim']['SHA256'], flags[2], launch['request']])
    return None


def fleet_view(snapshot, now, boot_anchor, ledger_path):
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    covered = {node: (zone, parent) for zone, item in ledger['groups'].items()
               for parent, nodes in item['allocations'].items() for node in nodes}
    is_device_receipt = snapshot.get('snapshot_kind') == 'training80-dispatch-v1'
    membership = snapshot['membership'] if is_device_receipt else snapshot
    need(membership.get('ledger_SHA256') in (None, sha256(ledger_raw).hexdigest()),
         'membership ledger SHA256 differs')
    expected_parents = {job['JobId'] for job in ledger['jobs']}
    need(set(membership['parents']) == expected_parents and
         len(covered) == membership['distinct_nodes'] == 80 and membership['allocated_DCU'] == 640,
         'expected the exact two-parent, 80-node allocation')
    nodes = {}
    for parent, observation in membership['parents'].items():
        groups = observation['coverage_by_group_and_node']
        for zone, rows in groups.items():
            for node, row in rows.items():
                need(node not in nodes and covered.get(node) == (zone, parent), 'membership node or zone differs')
                nodes[node] = dict(zone=zone, parent=parent, membership=row['state'], admission='unverified',
                                   numbered_step_assigned_DCU_bounds=row['numbered_step_assigned_DCU_bounds'])
    need(set(nodes) == set(covered), 'missing membership nodes')
    receipt = snapshot.get('local_receipt')
    fresh = False
    if is_device_receipt and receipt:
        need(snapshot.get('complete') is True and snapshot.get('total_cards') == 640 and
             hex_value(receipt.get('raw_response_SHA256'), 64) and receipt.get('request_id') == snapshot.get('request_id') and
             digest({key: value for key, value in snapshot.items() if key != 'local_receipt'}) == receipt['payload_SHA256'],
             'collector identity or payload differs')
        clocks = [receipt[key] for key in ('received_monotonic', 'roundtrip_s', 'boot_epoch_anchor')]
        need(all(type(clock) in (float, int) and math.isfinite(clock) for clock in clocks),
             'collector receipt clock is invalid')
        received, roundtrip, anchor = clocks
        fresh = (abs(boot_anchor - anchor) < 2 and now >= received and roundtrip >= 0 and
                 now - received + roundtrip <= SNAPSHOT_MAX_AGE_S)
        if fresh:
            placements = training80_inventory(snapshot)
            for node, (zone, parent, slurm_clear) in placements.items():
                probe = snapshot['nodes'][node]
                pci = probe['PCIs']
                need(probe['zone'] == zone and probe['parent'] == parent and len(pci) == len(set(pci)) == 8,
                     'collector PCI or node placement differs')
                if slurm_clear and probe['active_steps'] == [] and all(probe.get(key) is True for key in
                        ('device_probe_observed', 'KFD_clean', 'PID_clean', 'membership_valid')):
                    nodes[node]['admission'] = 'clear'
    return membership, nodes, fresh


def verify_launch(repo, job):
    launch = job['launch']
    need(hex_value(launch['publication_commit'], 40) and hex_value(launch['request'], 32) and
         hex_value(launch['workload_SHA256'], 64), 'invalid launch identity')
    claim = json.loads(pin_file(repo, launch['claim'], ('code/bw1000/', 'coordination/tasks/')))
    need(claim.get('request', claim.get('request_id')) == launch['request'] and
         claim.get(launch['authorization_key']) is True and
         launch['authorization_key'] in ('GPU_GO', 'operator_GO', 'training_GO', 'Model_GO', 'model_GO'),
         'published claim does not authorize this exact request')
    pin_file(repo, launch['launcher'], ('code/bw1000/', 'tools/'))
    package = Path(launch['manifest']['path']).parent
    package_refs = [ref for ref in (launch['claim'], launch['launcher']) if Path(ref['path']).is_relative_to(package)]
    verify_package_members(repo, launch['manifest'], required_files=package_refs)
    bindings = {
        'publication_receipt': dict(state='PUBLISHED_MATCHED', publication_commit=launch['publication_commit'],
                                    claim_SHA256=launch['claim']['SHA256'], manifest_SHA256=launch['manifest']['SHA256'],
                                    launcher_SHA256=launch['launcher']['SHA256']),
        'prerequisite_receipt': dict(state='TARGET_PREREQUISITES_VERIFIED', workload_SHA256=launch['workload_SHA256'],
                                     placement=job['placement'], required_ports=launch['required_ports'],
                                     min_updates=job.get('min_updates')),
        'original_request_receipt': dict(state='REMOTE_UNSTARTED_VERIFIED', original_namespace_absent=True,
                                         accounting_label_matches=0, prior_invocation=False),
    }
    for field, expected in bindings.items():
        raw = pin_file(repo, launch[field], ('analysis/bw1000/', 'results/bw1000/', 'local-only/'))
        expected['request'] = launch['request']
        value = json.loads(raw)
        need(all(key in value and type(value[key]) is type(item) and value[key] == item
                 for key, item in expected.items()), f'{field} does not close the exact dispatch prerequisite')
    need(type(launch['attempt_receipts']) is list and bool(launch['attempt_receipts']), 'missing exact attempted-run sentinels')
    for name in launch['attempt_receipts']:
        need(isinstance(name, str) and name.startswith('local-only/') and '\\' not in name,
             'attempt receipt must be an exact local-only file')
        marker = (repo / name).resolve()
        need(marker.is_relative_to(repo / 'local-only'), 'attempt receipt escaped its repository scope')
        try:
            marker.stat()
        except FileNotFoundError:
            continue
        return 'reconcile'
    return 'ready'


def read_queue_holds(path):
    path = path.resolve(strict=True)
    marks = ','.join('?' for _ in ACTIVE)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        rows = connection.execute(f'SELECT id,state,nodes FROM runs WHERE state IN ({marks})', ACTIVE).fetchall()
    result = []
    for run, state, raw_nodes in rows:
        nodes = json.loads(raw_nodes)
        need(isinstance(nodes, list) and all(isinstance(node, str) for node in nodes), 'queue nodes malformed')
        result.append(dict(run=run, state=state, nodes=nodes))
    return result


def plan(snapshot, backlog, *, repo=REPO, ledger_path=TRAINING80_LEDGER, holds=(), now=None, boot_anchor=None):
    repo = repo.resolve()
    now = time.monotonic() if now is None else now
    boot_anchor = time.time() - time.monotonic() if boot_anchor is None else boot_anchor
    membership, nodes, fresh = fleet_view(snapshot, now, boot_anchor, ledger_path)
    minimum = backlog['minimum_ready_training']
    need(type(minimum) is int and minimum >= 5, 'minimum ready training must be at least five')
    identities, workload_ids, request_ids, rows, proposed = set(), set(), set(), [], []
    occupied = set()
    held = {node for item in holds for node in item['nodes'] if node in nodes}
    ordered = sorted(backlog['items'], key=lambda job: (PRIORITY[job['kind']], job['priority'], job['id']))
    for job in ordered:
        ident, kind, state = (job[key] for key in ('id', 'kind', 'state'))
        need(ident not in identities and state in STATES and type(job['priority']) is int and
             job['priority'] >= 0 and isinstance(job['next_action'], str) and job['next_action'].strip(),
             'duplicate candidate or invalid state/priority/next action')
        identities.add(ident)
        placement = job['placement']
        wanted = set(placement)
        need(all(name in nodes and type(cards) is int and 0 < cards <= 8
                 for name, cards in placement.items()) and wanted, 'candidate requires explicit known nodes and device counts')
        row = dict(id=ident, case=job.get('case'), kind=kind, candidate_state=state, placement=placement,
                   requested_DCU=sum(placement.values()), next_action=job['next_action'])
        if kind == 'training':
            need(isinstance(job.get('case'), str) and job['case'].startswith('T') and
                 type(job['min_updates']) is int and job['min_updates'] >= 1,
                 'only positive planned optimizer updates qualify as training')
        if 'launch' in job:
            launch = job['launch']
            work, request = launch['workload_SHA256'], launch['request']
            need(work not in workload_ids and request not in request_ids, 'duplicate semantic scope or launch request')
            workload_ids.add(work)
            request_ids.add(request)
            row['request'] = request
        try:
            if 'evidence' in job:
                row['evidence'] = job['evidence']
                pin_file(repo, job['evidence'], ('coordination/tasks/', 'analysis/bw1000/', 'results/bw1000/', 'local-only/'))
            if state == 'ready':
                state = verify_launch(repo, job)
                if state == 'ready':
                    command = invocation(repo, job)
                    if command is None:
                        state = 'manual_launcher_binding_required'
                    else:
                        row['invocation'] = command
        except (OSError, ValueError, SyntaxError) as exc:
            state = 'invalid_launch_evidence'
            row['launch_evidence_error'] = str(exc)
        row['candidate_state'] = state
        row['dispatch_state'] = state
        if state == 'ready':
            probe = snapshot.get('nodes', {})
            ports = set(job['launch']['required_ports'])
            need(all(type(port) is int and 1 <= port <= 65535 for port in ports), 'invalid requested socket port')
            if not fresh:
                row['dispatch_state'] = 'awaiting_fresh_device_receipt'
            elif any(nodes[name]['admission'] != 'clear' or not isinstance(probe[name]['ports_checked'], list) or
                     not isinstance(probe[name]['occupied_ports'], list) or
                     not ports <= set(probe[name]['ports_checked']) or
                     ports.intersection(probe[name]['occupied_ports']) for name in wanted):
                row['dispatch_state'] = 'awaiting_device_or_port_clearance'
            elif held.intersection(wanted):
                row['dispatch_state'] = 'awaiting_original_queue_release'
            elif occupied.intersection(wanted):
                row['dispatch_state'] = 'awaiting_nonoverlapping_nodes'
            else:
                row['dispatch_state'] = 'proposed_for_root'
                occupied.update(wanted)
                proposed.append(ident)
        rows.append(row)
    training = [row for row in rows if row['kind'] == 'training']
    training_states = Counter(row['candidate_state'] for row in training)
    dispatch_states = Counter(row['dispatch_state'] for row in training)
    ready_training = training_states['ready']
    ready_to_dispatch = dispatch_states['proposed_for_root']
    training_pipeline = dict(ready_to_dispatch=ready_to_dispatch,
        ready_waiting_for_resources=ready_training - ready_to_dispatch,
        needs_preparation=training_states['blocked'] + training_states['manual_launcher_binding_required'],
        needs_reconciliation=training_states['reconcile'], invalid_evidence=training_states['invalid_launch_evidence'],
        recorded_submitted=training_states['submitted'], recorded_terminal=training_states['terminal'])
    groups = {}
    for node, value in sorted(nodes.items()):
        group = groups.setdefault(value['zone'], {}).setdefault(value['parent'], {})
        state = 'proposed_for_root' if node in occupied else 'queue_held' if node in held else value['admission']
        group.setdefault(state, []).append(node)
    return dict(observed_epoch=membership['observed_epoch'], fleet_nodes=80, allocated_DCU=640,
                numbered_step_membership_counts=dict(Counter(node['membership'] for node in nodes.values())),
                minimum_numbered_step_assigned_DCU=membership['minimum_numbered_step_assigned_DCU'],
                maximum_numbered_step_assigned_DCU=membership['maximum_numbered_step_assigned_DCU'],
                measured_busy_percent=None, fresh_device_receipt=fresh, admission_clear_nodes=sum(
                    node['admission'] == 'clear' for node in nodes.values()),
                outstanding_training_ready=ready_training, minimum_training_ready=minimum,
                training_pipeline=training_pipeline,
                training_ready_deficit=max(0, minimum - ready_training),
                blocked_or_reconciliation_training=training_states['blocked'] + training_states['reconcile'],
                proposed_for_root=proposed, proposed_DCU=sum(row['requested_DCU'] for row in rows
                    if row['dispatch_state'] == 'proposed_for_root'),
                proposed_nodes=sorted(occupied), queue_held_nodes=sorted(held), admission_by_zone_and_parent=groups,
                actual_submitted_DCU=0, advisory=True, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, help='original membership or authenticated collector receipt')
    parser.add_argument('--backlog', type=Path, default=BACKLOG)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_bytes())
    holds = read_queue_holds(args.queue_db)
    result = plan(snapshot, json.loads(args.backlog.read_bytes()), holds=holds)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
