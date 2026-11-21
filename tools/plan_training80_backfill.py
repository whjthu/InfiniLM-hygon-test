"""Read existing admission and queue evidence to suggest backfill or cooperative stop."""

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from report_training80_backlog import BACKLOG, QUEUE_DB, REPO, hex_value, pin_file, plan, queue_holds
from prepare_training80_dispatch_handoff import zone_coverage
from tuning_queue import DIRECT_STAGE, digest


LOW_PRIORITY = {'validation', 'inference', 'operator'}


def original_request(row):
    return row.get('direct_request', row.get('cpu_request', row.get('launch', {}).get('request')))


def active_direct_claims(connection):
    rows = connection.execute('SELECT id,handle,binding FROM runs WHERE stage=? AND state=?',
                              (DIRECT_STAGE, 'SUBMITTED'))
    return {run: (json.loads(handle), json.loads(binding)) for run, handle, binding in rows}


def stop_for_claim(repo, run, handle, binding, parents, held, legacy, observed):
    """Match the durable exact numbered step and the stop policy from its frozen claim."""
    request, parent, step_id = (handle.get(key) for key in ('request_id', 'parent', 'gpu_step'))
    ticket = binding.get('handoff_SHA256')
    if (not hex_value(request, 32) or request != binding.get('request') or
            run != digest(dict(stage=DIRECT_STAGE, request=request)) or
            not hex_value(ticket, 64) or ticket != handle.get('handoff_SHA256') or
            ticket != digest({key: value for key, value in binding.items() if key != 'handoff_SHA256'}) or
            not hex_value(binding.get('workload_SHA256'), 64) or parent not in parents or
            not isinstance(step_id, str) or not re.fullmatch(re.escape(parent)+r'\.[0-9]+', step_id) or
            handle.get('direct_kind') != binding.get('kind') or binding['kind'] not in LOW_PRIORITY):
        return None
    placement = binding.get('placement_by_parent', {})
    if set(placement) != {parent} or placement != handle.get('placement_by_parent'):
        return None
    counts = placement[parent]
    if not counts or any(type(value) is not int or not 1 <= value <= 8 for value in counts.values()):
        return None
    nodes = sorted(counts)
    step = parents[parent]['active_steps'].get(step_id, {})
    if (handle.get('nodes') != nodes or step.get('state') != 'RUNNING' or step.get('member_nodes') != nodes or
            step.get('allocated_DCU') != sum(counts.values()) or step.get('membership_qualified') is not True or
            any(held[node] != {run} or legacy[node] - {request} or observed[node] != {step_id} for node in nodes)):
        return None
    ref = binding.get('claim')
    if not isinstance(ref, dict):
        return None
    try:
        claim = json.loads(pin_file(repo, ref, ('code/bw1000/', 'coordination/tasks/')))
    except (KeyError, OSError, ValueError):
        return None
    stop = claim.get('backfill_stop') if isinstance(claim, dict) else None
    if not isinstance(stop, dict):
        return None
    grace = stop.get('grace_seconds')
    if (claim.get('request') != request or stop.get('method') != 'cooperative_signal' or
            stop.get('signal') not in ('USR1', 'TERM') or stop.get('save_partial_results') is not True or
            type(grace) is not int or grace <= 0):
        return None
    return dict(run=run, request=request, parent=parent, step=step_id, nodes=nodes, allocated_DCU=sum(counts.values()),
                placement_DCU_by_node=counts, signal=stop['signal'], grace_seconds=grace, claim_SHA256=ref['SHA256'])


def plan_backfill(snapshot, backlog, *, repo=REPO, holds=(), submitted=(), now=None, boot_anchor=None):
    """Suggest work on clear nodes and exact stops for conflicting low-priority steps."""
    repo = repo.resolve()
    report = plan(snapshot, backlog, repo=repo, holds=holds, now=now, boot_anchor=boot_anchor)
    work = {row['id']: row for row in backlog['items']}
    fresh = report['fresh_device_receipt']
    group_evidence, owners = zone_coverage(snapshot, report)
    groups, clear = [], set()
    for zone, parents in report['admission_by_zone_and_parent'].items():
        for parent, states in parents.items():
            group_clear = states.get('clear', []) + states.get('proposed_for_root', [])
            clear.update(group_clear)
            groups.append(dict(zone=zone, parent=parent, clean_nodes=sorted(group_clear),
                               clean_DCU_lower_bound=len(group_clear)*8 if fresh else None,
                               queue_held_nodes=states.get('queue_held', []),
                               original_held_nodes=states.get('original_launch_held', []),
                               device_holders_observed=group_evidence[zone][parent]['device_holders_observed'],
                               device_unverified=group_evidence[zone][parent]['device_unverified'],
                               active_numbered_step=group_evidence[zone][parent]['active_numbered_step']))
    launches = []
    for row in report['rows']:
        if row['dispatch_state'] != 'proposed_for_root':
            continue
        placement = {}
        for node, devices in row['placement'].items():
            placement.setdefault(owners[node], {})[node] = devices
        launches.append(dict(candidate=row['id'], kind=row['kind'], placement_by_parent=placement,
                             request=row['request'], requested_DCU=row['requested_DCU']))

    stops = []
    if fresh:
        parents = snapshot['membership']['parents']
        held, legacy, observed = defaultdict(set), defaultdict(set), defaultdict(set)
        for item in holds:
            for node in item['nodes']:
                held[node].add(item['run'])
        for candidate, record in report['original_launch_holds'].items():
            for node in record['nodes']:
                legacy[node].add(original_request(work[candidate]))
        for parent in parents.values():
            for step_id, step in parent['active_steps'].items():
                for node in step['member_nodes']:
                    observed[node].add(step_id)
        allowed = {}
        for run, (handle, binding) in dict(submitted).items():
            stop = stop_for_claim(repo, run, handle, binding, parents, held, legacy, observed)
            if stop is not None:
                allowed.update(dict.fromkeys(stop['nodes'], stop))

        occupied = {node for job in launches if job['kind'] == 'training'
                    for nodes in job['placement_by_parent'].values() for node in nodes}
        needed_for_training = set()
        for row in report['rows']:
            if row['kind'] != 'training' or row['dispatch_state'] != 'awaiting_original_queue_release':
                continue
            wanted = set(row['placement'])
            waiting = wanted - clear
            if not waiting or not waiting <= allowed.keys() or occupied.intersection(wanted):
                continue
            ports = set(work[row['id']]['launch']['required_ports'])
            probes = snapshot['nodes']
            if any(not isinstance(probes[node].get('ports_checked'), list) or
                   not isinstance(probes[node].get('occupied_ports'), list) or
                   not ports <= set(probes[node]['ports_checked']) or
                   ports.intersection(probes[node]['occupied_ports']) for node in wanted & clear):
                continue
            matched = {allowed[node]['step']: allowed[node] for node in waiting}
            if any(set(stop['nodes']) - waiting for stop in matched.values()):
                continue
            occupied.update(wanted)
            needed_for_training.update(wanted)
            stops.append(dict(candidate=row['id'], requested_DCU=row['requested_DCU'],
                              original_snapshot_request=snapshot['request_id'],
                              original_snapshot_SHA256=snapshot['local_receipt']['raw_response_SHA256'],
                              original_steps=[matched[step] for step in sorted(matched)],
                              requires_terminal_and_independent_cleanup=True, requires_new_device_admission=True))
        launches = [job for job in launches if job['kind'] == 'training' or not any(
            needed_for_training.intersection(nodes) for nodes in job['placement_by_parent'].values())]

    return dict(advisory=True, allocation_DCU=report['allocated_DCU'], fresh_device_receipt=fresh,
                observation_epoch=report['observed_epoch'], measured_busy_percent=None,
                training_ready=report['outstanding_training_ready'], training_ready_deficit=report['training_ready_deficit'],
                device_groups=groups, clean_node_dispatches=launches, cooperative_stops=stops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--backlog', type=Path, default=BACKLOG)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    args = parser.parse_args()
    uri = args.queue_db.resolve(strict=True).as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute('BEGIN')
        holds, submitted = queue_holds(db), active_direct_claims(db)
    result = plan_backfill(json.loads(args.snapshot.read_bytes()), json.loads(args.backlog.read_bytes()),
                           holds=holds, submitted=submitted)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
