"""Build or claim a frozen launcher handoff from an original 80-node receipt."""

import argparse
import json
from pathlib import Path
import time

from report_training80_backlog import (BACKLOG, QUEUE_DB, REPO, hex_value, need, plan, queue_holds,
                                      read_queue_holds)
from tuning_queue import DIRECT_STAGE, Queue, SNAPSHOT_MAX_AGE_S, canonical, digest


def zone_coverage(snapshot, report):
    lookup = snapshot.get('membership', snapshot)['parents']
    probes = snapshot.get('nodes', {}) if report['fresh_device_receipt'] else {}
    result = {}
    owners = {}
    for zone, parents in report['admission_by_zone_and_parent'].items():
        result[zone] = {}
        for parent, groups in parents.items():
            membership = lookup[parent]['coverage_by_group_and_node'][zone]
            idle = []
            active = []
            other = []
            for node, observation in membership.items():
                owners[node] = parent
                state = observation['state']
                bounds = observation['numbered_step_assigned_DCU_bounds']
                if state == 'NO_NUMBERED_STEP' and bounds == [0, 0]:
                    idle.append(node)
                elif state in ('FULLY_ASSIGNED', 'PARTIALLY_ASSIGNED', 'CPU_STEP_ONLY'):
                    active.append(node)
                else:
                    other.append(node)
            idle = set(idle)
            unverified = idle.intersection(groups.get('unverified', []))
            blocked = {node for node in unverified if probes.get(node, {}).get('device_probe_observed') is True
                       and any(type(probes[node].get(key)) is int and probes[node][key] > 0
                               for key in ('kfd_process_count', 'own_device_fd_process_count'))}
            result[zone][parent] = dict(slurm_no_numbered_step=sorted(idle),
                device_clear_unplaced=sorted(idle.intersection(groups.get('clear', []))),
                device_unverified=sorted(unverified - blocked), device_holders_observed=sorted(blocked),
                queue_held=groups.get('queue_held', []), original_launch_held=groups.get('original_launch_held', []),
                proposed=groups.get('proposed_for_root', []),
                active_numbered_step=sorted(active), other_membership=sorted(other))
    return result, owners


def build_handoff(snapshot, backlog, *, repo=REPO, holds=(), selected=None, now=None, boot_anchor=None):
    repo = repo.resolve()
    now = time.monotonic() if now is None else now
    report = plan(snapshot, backlog, repo=repo, holds=holds, now=now, boot_anchor=boot_anchor)
    zones, owners = zone_coverage(snapshot, report)
    jobs = {job['id']: job for job in backlog['items']}
    rows = {row['id']: row for row in report['rows']}
    proposed = report['proposed_for_root']
    if selected is not None:
        need(selected in proposed, 'selected candidate has no current device-admitted nonoverlapping proposal')
        proposed = [selected]
    packets = []
    unsupported = [row['id'] for row in report['rows']
                   if row['dispatch_state'] == 'manual_launcher_binding_required']
    invalid_evidence = {row['id']: row['launch_evidence_error'] for row in report['rows']
                        if row['dispatch_state'] == 'invalid_launch_evidence'}
    if proposed:
        receipt = snapshot['local_receipt']
        expires = receipt['received_monotonic'] + SNAPSHOT_MAX_AGE_S - receipt['roundtrip_s']
        held_digest = digest(sorted(holds, key=lambda row: row['run']))
        for ident in proposed:
            job = jobs[ident]
            launch = job['launch']
            placement = {}
            for node, count in job['placement'].items():
                placement.setdefault(owners[node], {})[node] = count
            packet = dict(candidate=ident, kind=job['kind'], case=job.get('case'), request=launch['request'],
                          publication_commit=launch['publication_commit'], workload_SHA256=launch['workload_SHA256'],
                          snapshot_request=receipt['request_id'], snapshot_payload_SHA256=receipt['payload_SHA256'],
                          snapshot_raw_SHA256=receipt['raw_response_SHA256'], queue_holds_SHA256=held_digest,
                          original_launch_holds_SHA256=digest(report['original_launch_holds']),
                          verified_original_non_submissions_SHA256=digest(report['verified_original_non_submissions']),
                          expires_local_monotonic=expires, boot_epoch_anchor=receipt['boot_epoch_anchor'],
                          placement_by_parent=placement, requested_DCU=sum(job['placement'].values()),
                          claim=launch['claim'], manifest=launch['manifest'], launcher=launch['launcher'],
                          attempt_receipts=launch['attempt_receipts'], invocation=rows[ident]['invocation'])
            packet['handoff_SHA256'] = digest(packet)
            packets.append(packet)
    return dict(advisory=True, actual_dispatched_DCU=0, allocation_DCU=report['allocated_DCU'],
                observed_epoch=report['observed_epoch'], fleet_nodes=report['fleet_nodes'],
                fresh_device_receipt=report['fresh_device_receipt'], measured_busy_percent=None,
                device_probe_scope=report['device_probe_scope'],
                device_admitted_unheld_DCU_lower_bound=report['device_admitted_unheld_DCU_lower_bound'],
                minimum_numbered_step_assigned_DCU=report['minimum_numbered_step_assigned_DCU'],
                maximum_numbered_step_assigned_DCU=report['maximum_numbered_step_assigned_DCU'],
                numbered_step_membership_counts=report['numbered_step_membership_counts'],
                outstanding_training_ready=report['outstanding_training_ready'],
                minimum_training_ready=report['minimum_training_ready'],
                training_pipeline=report['training_pipeline'],
                training_ready_deficit=report['training_ready_deficit'], coverage_by_zone_and_parent=zones,
                blocked_training=[{key: row[key] for key in ('id', 'case', 'candidate_state', 'next_action')}
                                  for row in report['rows'] if row['kind'] == 'training' and
                                  row['candidate_state'] in ('blocked', 'reconcile')],
                coordinator_handoffs=packets, manual_launcher_binding_required=unsupported,
                original_launch_holds=report['original_launch_holds'],
                verified_original_non_submissions=report['verified_original_non_submissions'],
                prior_workload_claims=report['prior_workload_claims'],
                invalid_launch_evidence=invalid_evidence)


def claim_handoff(queue, snapshot, backlog, packet, intent, *, repo=REPO, now=None, boot_anchor=None):
    [claim] = claim_handoffs(queue, snapshot, backlog, [(packet, intent)], repo=repo, now=now, boot_anchor=boot_anchor)
    return claim


def claim_handoffs(queue, snapshot, backlog, requests, *, repo=REPO, now=None, boot_anchor=None):
    """Reserve disjoint handoffs from one original snapshot in one write transaction."""
    need(0 < len(requests) <= 80, 'at least one handoff is required and at most one per node')
    need(all(isinstance(packet, dict) and isinstance(packet.get('candidate'), str) and
             hex_value(packet.get('request'), 32) for packet, _ in requests), 'invalid original handoff identity')
    candidates = [packet['candidate'] for packet, _ in requests]
    runs = [digest(dict(stage=DIRECT_STAGE, request=packet['request'])) for packet, _ in requests]
    intents = [intent for _, intent in requests]
    need(all(isinstance(intent, str) for intent in intents) and len(set(intents)) == len(requests) and
         len(set(candidates)) == len(requests) and len(set(runs)) == len(requests),
         'batch must have distinct original requests, candidates and intent paths')
    current_monotonic = time.monotonic if now is None else lambda: now
    db = queue.db
    db.execute('BEGIN IMMEDIATE')
    try:
        need(all(db.execute('SELECT 1 FROM runs WHERE id=?', (run,)).fetchone() is None for run in runs),
             'original direct request already claimed; reconcile without replay')
        report = build_handoff(snapshot, backlog, repo=repo, holds=queue_holds(db),
                               now=current_monotonic(), boot_anchor=boot_anchor)
        current = {packet['candidate']: packet for packet in report['coordinator_handoffs']}
        need(all(candidate in current for candidate in candidates),
             'selected candidate has no current device-admitted nonoverlapping proposal')
        for packet, intent in requests:
            rebuilt = current[packet['candidate']]
            need(packet == rebuilt, 'handoff drift; rebuild from the original device receipt')
            need(intent in rebuilt['attempt_receipts'], 'intent path differs from original request scope')
        claims = [reserve_handoff(queue, run, current[candidate], intent)
                  for run, candidate, intent in zip(runs, candidates, intents)]
        claim_time = current_monotonic()
        anchor = time.time() - claim_time if boot_anchor is None else boot_anchor
        need(all(abs(anchor - packet['boot_epoch_anchor']) < 2 and claim_time <= packet['expires_local_monotonic']
                 for packet, _ in requests), 'handoff expired before durable claim')
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    return claims


def reserve_handoff(queue, run, rebuilt, intent):
    placements = rebuilt['placement_by_parent']
    nodes = sorted(node for selected in placements.values() for node in selected)
    handle = dict(request_id=rebuilt['request'], intent=intent, nodes=nodes,
                  direct_kind=rebuilt['kind'], placement_by_parent=placements,
                  handoff_SHA256=rebuilt['handoff_SHA256'])
    if len(placements) == 1:
        handle['parent'] = next(iter(placements))
    candidate = digest(dict(stage=DIRECT_STAGE, request=rebuilt['request'], case=rebuilt['case'],
                            workload_SHA256=rebuilt['workload_SHA256']))
    queue.db.execute('INSERT INTO candidates(id,spec) VALUES(?,?)',
                     (candidate, canonical(dict(case=rebuilt['case'], kind=rebuilt['kind'],
                                                workload_SHA256=rebuilt['workload_SHA256'])).decode()))
    queue.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
                     (run, candidate, DIRECT_STAGE, 'DISPATCHING', canonical(rebuilt).decode(),
                      canonical(nodes).decode(), canonical(handle).decode(), None, time.time()))
    reservation = dict(dispatch_ticket=rebuilt['handoff_SHA256'], nodes=nodes,
                       snapshot_SHA256=rebuilt['snapshot_payload_SHA256'])
    if 'parent' in handle:
        reservation['parent'] = handle['parent']
    queue.event(run, 'reserved', reservation)
    queue.event(run, 'dispatch_claimed', dict(dispatch_ticket=reservation['dispatch_ticket'], handle=handle))
    return dict(run=run, state='DISPATCHING', handle=handle, invocation=rebuilt['invocation'],
                launcher=rebuilt['launcher'], expires_local_monotonic=rebuilt['expires_local_monotonic'],
                actual_slurm_submissions=0, actual_dispatched_DCU=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--backlog', type=Path, default=BACKLOG)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    parser.add_argument('--select', help='re-evaluate one candidate at the handoff boundary')
    claims = parser.add_mutually_exclusive_group()
    claims.add_argument('--claim-packet', type=Path, help='claim one exact previously generated packet')
    claims.add_argument('--claim-batch', type=Path, help='claim a JSON list of packet and intent objects from one receipt')
    parser.add_argument('--intent', help='the frozen intent path corresponding to the selected launcher')
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_bytes())
    backlog = json.loads(args.backlog.read_bytes())
    if args.claim_packet is not None or args.claim_batch is not None:
        queue = Queue(args.queue_db.resolve(strict=True))
        try:
            if args.claim_batch is None:
                packet = json.loads(args.claim_packet.read_bytes())
                need(args.intent is not None and args.select == packet['candidate'],
                     'claiming requires the selected candidate and its exact intent path')
                result = claim_handoff(queue, snapshot, backlog, packet, args.intent)
            else:
                need(args.select is None and args.intent is None, 'batch contains its own frozen candidates and intents')
                raw = json.loads(args.claim_batch.read_bytes())
                need(isinstance(raw, list) and all(isinstance(item, dict) and set(item) == {'packet', 'intent'}
                                                   for item in raw), 'invalid handoff batch shape')
                claimed = claim_handoffs(queue, snapshot, backlog, [(item['packet'], item['intent']) for item in raw])
                result = dict(claims=claimed, actual_slurm_submissions=0, actual_dispatched_DCU=0)
        finally:
            queue.db.close()
    else:
        need(args.intent is None, '--intent requires a frozen claim packet')
        result = build_handoff(snapshot, backlog, holds=read_queue_holds(args.queue_db), selected=args.select)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
