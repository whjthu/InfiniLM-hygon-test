"""Produce priority, hold and port arguments for the existing training80 collector."""

import argparse
import json
from pathlib import Path

from prepare_training80_reconnect_worklist import UNVERIFIED_ATTEMPT_MARKER, WORKLIST, prepare
from report_training80_backlog import (BACKLOG, PRIORITY, QUEUE_DB, REPO, TRAINING80_LEDGER,
                                      direct_launch_state, direct_request_states, direct_workloads, hex_value,
                                      invocation, need, pin_file, prior_workload_claims, read_original_holds, read_queue_holds,
                                      verify_launch)
from tuning_queue import DIRECT_CLOSED_STATES


def probe_arguments(repo, backlog, ledger, holds, preflight=(), *, priority_only=False):
    repo = repo.resolve()
    ownership = {node: parent for zone in ledger['groups'].values()
                 for parent, nodes in zone['allocations'].items() for node in nodes}
    need(len(ownership) == 80, 'expected the frozen 80-node inventory')
    queue_held = {node for row in holds for node in row['nodes'] if node in ownership}
    direct_states = direct_request_states(holds)
    workloads = direct_workloads(holds)
    original_holds, errors, resolved = read_original_holds(repo, backlog['items'], ownership, holds)
    priority = {}
    ports = set()
    verified = []
    blocked = {}
    prior_claims = {}
    excluded_requests = set()

    def select(nodes, checked_ports):
        need(all(node in ownership for node in nodes), 'priority candidate is outside the node ledger')
        need(all(type(port) is int and 1 <= port <= 65535 for port in checked_ports), 'invalid frozen port')
        priority.update(dict.fromkeys(nodes))
        ports.update(checked_ports)

    jobs = sorted(backlog['items'], key=lambda job: (PRIORITY[job['kind']], job['priority'], job['id']))
    for job in jobs:
        ident = job['id']
        nodes = sorted(job['placement'])
        try:
            if direct_launch_state(job, direct_states) is not None:
                continue
            if job['state'] in ('reconcile', 'submitted'):
                pin_file(repo, job['evidence'], ('coordination/tasks/', 'results/bw1000/', 'analysis/bw1000/', 'local-only/'))
            if job['state'] != 'ready' or ident in original_holds:
                continue
            prior = prior_workload_claims(job, workloads)
            if prior:
                prior_claims[ident] = prior
                excluded_requests.add(job['launch']['request'])
                continue
            state = verify_launch(repo, job)
            if state != 'ready':
                need(state == 'reconcile' and all(node in ownership for node in nodes), 'invalid original placement')
                original_holds[ident] = dict(state=state, nodes=nodes)
            elif invocation(repo, job) is not None:
                verified.append(job)
            else:
                errors[ident] = 'manual launcher binding required'
        except (KeyError, OSError, TypeError, ValueError, SyntaxError) as error:
            errors[ident] = str(error)

    preflight = tuple(preflight)
    marker_holds = {}
    for row in preflight:
        if row['target_preflight_state'] != UNVERIFIED_ATTEMPT_MARKER:
            continue
        request, node, parent = (row[key] for key in ('request', 'node', 'parent'))
        need(hex_value(request, 32) and node in ownership and parent == ownership[node] and request not in marker_holds,
             'unverified original attempt must bind one exact request and ledger node')
        if direct_states.get(request) not in DIRECT_CLOSED_STATES:
            marker_holds[request] = dict(case=row['case'], parent=parent, nodes=[node], error=row['local_attempt_error'])
    held = queue_held | {node for record in (*original_holds.values(), *marker_holds.values()) for node in record['nodes']}
    ready_device_jobs = []
    for job in verified:
        ident = job['id']
        nodes = sorted(job['placement'])
        overlapping = sorted(held.intersection(nodes))
        if overlapping:
            blocked[ident] = overlapping
        else:
            select(nodes, job['launch']['required_ports'])
            if any(job['placement'].values()):
                ready_device_jobs.append(ident)

    preflight_priority = []
    cpu_backlog = {job['case']: job for job in jobs if job.get('cpu_prerequisite') is True}
    for row in sorted(preflight, key=lambda row: (not bool(row['requested_DCU']), row['case'])):
        if row['case'] in cpu_backlog:
            job = cpu_backlog[row['case']]
            need(row['requested_DCU'] == 0 and job['placement'] == {row['node']: 0}
                 and job['attempt_receipts'] == [row['attempt_receipt']],
                 'CPU worklist and persistent node hold differ')
        if (row['target_preflight_state'] != 'WAITING_AUTHENTICATED_TARGET' or row.get('request') in excluded_requests
                or direct_states.get(row.get('request')) in DIRECT_CLOSED_STATES):
            continue
        node = row['node']
        if node in held:
            blocked[row['case']] = [node]
        else:
            port = row['frozen_rendezvous_port']
            select([node], [] if port is None else [port])
            preflight_priority.append(node)

    by_parent = {}
    for node in priority:
        by_parent.setdefault(ownership[node], []).append(node)
    fields = [('port', sorted(ports)), ('priority-node', priority), ('held-node', sorted(held))]
    arguments = [token for option, items in fields for item in items for token in ('--' + option, str(item))]
    if priority_only and ready_device_jobs:
        arguments.append('--priority-only')
    return dict(collector_arguments=arguments, priority_by_parent=by_parent, required_ports=sorted(ports),
                queue_held_nodes=sorted(queue_held), held_nodes=sorted(held),
                reconcile_candidates={ident: record['nodes'] for ident, record in original_holds.items()
                                      if record['state'] in ('reconcile', 'submitted')},
                original_launch_holds=original_holds, preflight_attempt_marker_holds=marker_holds,
                verified_original_non_submissions=resolved,
                prior_workload_claims=prior_claims,
                blocked_candidate_held_nodes=blocked, invalid_or_manual_launches=errors,
                prepared_preflight_priority_nodes=preflight_priority,
                priority_only_requested=priority_only,
                priority_only_enabled=priority_only and bool(ready_device_jobs),
                ready_device_jobs=ready_device_jobs,
                verified_training_ready=sum(job['kind'] == 'training' for job in verified),
                training_ready_without_original_hold=sum(job['kind'] == 'training' and job['id'] not in blocked
                                                         for job in verified),
                current_device_eligible_DCU=None, measured_busy_percent=None, actual_submissions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backlog', type=Path, default=BACKLOG)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    parser.add_argument('--worklist', type=Path, default=WORKLIST)
    parser.add_argument('--priority-only', action='store_true',
                        help='use a priority-only scan when there is verified GPU launch work')
    args = parser.parse_args()
    ledger_raw = TRAINING80_LEDGER.read_bytes()
    preflight = prepare(REPO, json.loads(args.worklist.read_bytes()), ledger_raw)['candidates']
    report = probe_arguments(REPO, json.loads(args.backlog.read_bytes()), json.loads(ledger_raw),
                             read_queue_holds(args.queue_db), preflight, priority_only=args.priority_only)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
