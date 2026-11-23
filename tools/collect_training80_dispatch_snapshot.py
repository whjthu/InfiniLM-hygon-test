"""One-shot Slurm and CPU-only device admission for a verified allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from read_training_slurm_membership import snapshot
from sample_training80_busy import require_pins
from training_dispatch_allocation import DEVICE_RECEIPT_KINDS, allocation
from tuning_queue import receive_snapshot, require


HOST_PYTHON = '/usr/local/python3.12/bin/python3'
PROC = Path('/proc')
KFD_PROCESSES = Path('/sys/class/kfd/kfd/proc')
DRM = Path('/sys/class/drm')
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
NODES_PER_PROBE = 8
PARALLEL_PROBES = 4
PROBE_TIMEOUT_S = 10
PROBE_TERM_GRACE_S = 2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_device_probe(argv, *, capture_output, timeout, check):
    """Give srun time to relay TERM before killing only its local launcher."""
    require(capture_output and not check, 'unexpected probe subprocess options')
    probe = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = probe.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as expired:
        probe.terminate()
        try:
            _, _ = probe.communicate(timeout=PROBE_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.wait(timeout=PROBE_TERM_GRACE_S)
        raise expired
    finally:
        probe.stdout.close()
        probe.stderr.close()
    return subprocess.CompletedProcess(argv, probe.returncode, stdout, stderr)


def validate_ports(ports):
    require(type(ports) is list and all(type(p) is int and 1 <= p <= 65535 for p in ports),
            'invalid requested TCP/UDP ports')
    return sorted(set(ports))


def validate_scope_nodes(nodes, owners):
    nodes = list(nodes)
    require(all(isinstance(node, str) and node in owners for node in nodes) and len(set(nodes)) == len(nodes),
            'priority and held nodes must be unique ledger members')
    return nodes


def probe_batches(candidates, priority_nodes, *, isolate_priority=False):
    """Schedule priority probes first, optionally with independent steps per node."""
    priorities = set(priority_nodes)
    batches = []
    for parent, names in candidates.items():
        available = set(names)
        segments = ([name for name in priority_nodes if name in available],
                    [name for name in names if name not in priorities])
        index = 0
        for tier, segment in enumerate(segments):
            batch_size = 1 if isolate_priority and tier == 0 else NODES_PER_PROBE
            for offset in range(0, len(segment), batch_size):
                batches.append((tier, offset // batch_size, parent, index,
                                segment[offset:offset + batch_size]))
                index += 1
    return sorted(batches, key=lambda item: item[:3])


def device_status(ports):
    pci = sorted((entry / 'device').resolve(strict=True).name.lower()
                 for entry in DRM.iterdir() if re.fullmatch(r'renderD[0-9]+', entry.name))
    processes = {entry.name for entry in KFD_PROCESSES.iterdir()}
    require(all(pid.isdecimal() for pid in processes), 'unknown KFD process inventory format')
    own_device_fds = set()
    for entry in PROC.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            for fd in (entry / 'fd').iterdir():
                try:
                    target = os.readlink(fd).removesuffix(' (deleted)')
                except FileNotFoundError:
                    continue
                if target == '/dev/kfd' or target.startswith('/dev/dri/'):
                    own_device_fds.add(entry.name)
        except FileNotFoundError:
            continue
    occupied_ports = set()
    if ports:
        for table in ('tcp', 'tcp6', 'udp', 'udp6'):
            lines = (PROC / 'net' / table).read_text().splitlines()
            require(lines and 'local_address' in lines[0], 'unknown kernel socket table format')
            occupied_ports.update(int(line.split()[1].rsplit(':', 1)[1], 16) for line in lines[1:])
    return dict(PCIs=pci, kfd_process_count=len(processes), own_device_fd_process_count=len(own_device_fds),
                occupied_requested_ports=sorted(set(ports) & occupied_ports))


def worker(request, parent, ports):
    node = socket.gethostname().split('.', 1)[0]
    step = os.environ['SLURM_STEP_ID']
    require(os.environ['SLURM_JOB_ID'] == parent and step.isdecimal(), 'wrong CPU-only probe step')
    row = dict(request_id=request, parent=parent, step=parent + '.' + step, node=node,
               ports_checked=ports, observer_uid=os.getuid())
    try:
        row.update(device_status(ports))
    except (OSError, ValueError, IndexError) as error:
        row['probe_error'] = type(error).__name__
    print(json.dumps(row, allow_nan=False), flush=True)


def read_worker_rows(raw, request, parent, nodes, ports, uid):
    require(len(raw) <= MAX_OUTPUT_BYTES, 'CPU-only probe output exceeded bounds')
    rows = {}
    steps = set()
    for line in raw.splitlines():
        row = json.loads(line)
        require(isinstance(row, dict), 'invalid CPU probe output record')
        name = row['node']
        step = row['step']
        require(row['request_id'] == request and row['parent'] == parent and row['observer_uid'] == uid and
                row['ports_checked'] == ports and re.fullmatch(re.escape(parent) + r'\.[0-9]+', step) and
                name in nodes and name not in rows, 'foreign, duplicate or mismatched CPU probe evidence')
        rows[name] = row
        steps.add(step)
    require(len(steps) <= 1, 'multiple CPU-only probe steps under one parent')
    return rows


def collect(request, ledger_raw, pins, ports, *, priority_nodes=(), held_nodes=(), priority_only=False,
            run=None, clock=time.monotonic):
    require(re.fullmatch(r'[0-9a-f]{32}', request), 'request ID must be the original 32 hex digest')
    kind, groups, owners = allocation(ledger_raw)
    require_pins(owners, pins)
    ports = validate_ports(ports)
    priority_nodes = validate_scope_nodes(priority_nodes, owners)
    held_nodes = set(validate_scope_nodes(held_nodes, owners))
    uid = os.getuid()
    query_run = subprocess.run if run is None else run
    probe_run = run_device_probe if run is None else run
    started = clock()
    deadline = started + 48
    hostlists = {}

    def query(*args):
        # Hostlist expansion depends only on its expression. Reuse it between the two live reads.
        is_expansion = args[:3] == ('scontrol', 'show', 'hostnames')
        if is_expansion and args in hostlists:
            return hostlists[args]
        remaining = deadline - clock()
        require(remaining > 0, 'Slurm observation budget expired')
        raw = query_run(args, capture_output=True, timeout=min(8, remaining), check=True).stdout
        require(len(raw) <= MAX_OUTPUT_BYTES, 'Slurm output exceeded bounds')
        lines = raw.decode('utf-8').strip().splitlines()
        if is_expansion:
            hostlists[args] = lines
        return lines

    initial = snapshot(groups, owner_uid=uid, run_query=query)
    require(all(parent['state'] == 'RUNNING' and parent['owned_by_observer'] and parent['step_locations_exact']
                for parent in initial['parents'].values()), 'initial Slurm ownership or step locations unverified')
    candidates = {parent: [] for parent in groups}
    for row in initial['dispatch_candidates']['ordered_node_checks']:
        if row['state'] == 'NO_NUMBERED_STEP' and row['node'] not in held_nodes:
            candidates[row['parent']].append(row['node'])

    def probe(parent, index, nodes):
        batch_request = digest(f'{request}:{parent}:{index}'.encode('ascii'))[:32]
        receipt = dict(request_id=batch_request, requested_nodes=nodes)
        remaining = deadline - clock() - 12
        cleanup_budget = 2 * PROBE_TERM_GRACE_S
        if remaining <= cleanup_budget:
            return {}, dict(receipt, state='BUDGET_EXPIRED')
        count = str(len(nodes))
        args = ['srun', '--threads=1', '--jobid=' + parent, '--job-name=training80-admit-' + batch_request,
                '--exact', '--immediate=5', '--nodes=' + count, '--ntasks=' + count,
                '--ntasks-per-node=1', '--nodelist=' + ','.join(nodes), '--cpus-per-task=1',
                '--mem=256M', '--gres=none', '--time=00:01:00', '--kill-on-bad-exit=0', HOST_PYTHON,
                '-B', str(Path(__file__).resolve()), 'node', batch_request, parent, json.dumps(ports)]
        try:
            result = probe_run(args, capture_output=True, timeout=min(PROBE_TIMEOUT_S, remaining - cleanup_budget),
                               check=False)
            rows = read_worker_rows(result.stdout, batch_request, parent, nodes, ports, uid)
            state = 'SRUN_FAILED_WITHOUT_ROWS' if result.returncode and not rows else 'RETURNED'
            return rows, dict(receipt, state=state, returncode=result.returncode, received_nodes=sorted(rows),
                              stdout_SHA256=digest(result.stdout), stderr_SHA256=digest(result.stderr))
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            return {}, dict(receipt, state=type(error).__name__)

    # A priority-only observation serves the next dispatch. Each candidate node needs its
    # own step because one unavailable node can prevent an entire multi-node step starting.
    batch_specs = probe_batches(candidates, priority_nodes, isolate_priority=priority_only)
    scope = 'FULL_FLEET'
    deferred = set()
    if priority_only:
        priority_batches = [batch for batch in batch_specs if batch[0] == 0]
        if priority_batches:
            scope = 'PRIORITY_ONLY'
            deferred = {node for tier, _, _, _, names in batch_specs if tier != 0 for node in names}
            batch_specs = priority_batches
        else:
            scope = 'FULL_FLEET_FALLBACK'
    reports = {parent: dict(requested_nodes=[node for node in names if node not in deferred],
                            received_nodes=[], batches=[])
               for parent, names in candidates.items()}
    observed_rows, absent_rows = {}, {node: 'LOCAL_QUEUE_HELD' for node in held_nodes}
    absent_rows.update(dict.fromkeys(deferred, 'DEFERRED_BY_PRIORITY_ONLY'))
    with ThreadPoolExecutor(max_workers=PARALLEL_PROBES) as executor:
        futures = [(tier, parent, executor.submit(probe, parent, index, nodes))
                   for tier, _, parent, index, nodes in batch_specs]
        for tier, parent, future in futures:
            rows, receipt = future.result()
            receipt['priority'] = tier == 0
            reports[parent]['batches'].append(receipt)
            reports[parent]['received_nodes'].extend(rows)
            observed_rows.update(rows)
            for name in set(receipt['requested_nodes']).difference(rows):
                absent_rows[name] = 'ROW_MISSING' if receipt['state'] == 'RETURNED' else receipt['state']
    for report in reports.values():
        report['received_nodes'].sort()
    membership = snapshot(groups, owner_uid=uid, run_query=query)
    membership['ledger_SHA256'] = digest(ledger_raw)
    nodes = {}
    for name, (parent, zone) in owners.items():
        observation = membership['parents'][parent]
        row = observed_rows.get(name, {})
        seen_pci = row.get('PCIs')
        observed = (type(seen_pci) is list and all(isinstance(pci, str) for pci in seen_pci) and
                    sorted(seen_pci) == sorted(pins[name]) and 'probe_error' not in row)
        checked_ports = row.get('occupied_requested_ports')
        port_status_known = (observed and type(checked_ports) is list and
                             all(type(p) is int and p in ports for p in checked_ports))
        kfd_count = row.get('kfd_process_count')
        fd_count = row.get('own_device_fd_process_count')
        probe_error = row.get('probe_error', absent_rows.get(name))
        if row and not observed and probe_error is None:
            probe_error = 'PCI_PIN_MISMATCH'
        active = [step for step, record in observation['active_steps'].items() if name in record['member_nodes']]
        nodes[name] = dict(zone=zone, parent=parent, PCIs=pins[name], device_probe_observed=observed,
                           KFD_clean=observed and type(kfd_count) is int and kfd_count == 0,
                           PID_clean=observed and type(fd_count) is int and fd_count == 0,
                           kfd_process_count=kfd_count if type(kfd_count) is int and kfd_count >= 0 else None,
                           own_device_fd_process_count=fd_count if type(fd_count) is int and fd_count >= 0 else None,
                           ports_clean=port_status_known and checked_ports == [],
                           ports_checked=ports if port_status_known else None,
                           occupied_ports=checked_ports if port_status_known else None,
                           membership_valid=(observation['state'] == 'RUNNING' and observation['owned_by_observer']
                                             and observation['step_locations_exact']), active_steps=sorted(active),
                           device_probe_step=row.get('step'), probe_error=probe_error)
    require(clock() <= deadline, 'observation budget expired')
    return dict(request_id=request, observed_epoch=membership['observed_epoch'], complete=True, total_cards=len(owners) * 8,
                membership=membership, nodes=nodes, probes=reports,
                requested_priority_nodes=priority_nodes, requested_held_nodes=sorted(held_nodes),
                device_probe_scope=scope, deferred_device_probe_nodes=sorted(deferred),
                measured_busy_percent=None, snapshot_kind=kind)


def receive(raw, request_started_monotonic, request_id):
    """Call once with the authenticated transport's original response and monotonic start."""
    require(len(raw) <= MAX_OUTPUT_BYTES, 'dispatch snapshot exceeded bounds')
    payload = json.loads(raw)
    require(isinstance(payload, dict) and payload.get('snapshot_kind') in DEVICE_RECEIPT_KINDS and
            payload.get('request_id') == request_id,
            'wrong dispatch snapshot request')
    return receive_snapshot(payload, request_started_monotonic, request_id, digest(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    host = modes.add_parser('collect')
    host.add_argument('request')
    host.add_argument('ledger', type=Path)
    host.add_argument('pci_pins', type=Path)
    host.add_argument('pci_pins_sha256')
    host.add_argument('--port', type=int, action='append', default=[])
    host.add_argument('--priority-node', action='append', default=[])
    host.add_argument('--held-node', action='append', default=[])
    host.add_argument('--priority-only', action='store_true')
    node = modes.add_parser('node')
    node.add_argument('request')
    node.add_argument('parent')
    node.add_argument('ports')
    args = parser.parse_args()
    if args.mode == 'node':
        worker(args.request, args.parent, validate_ports(json.loads(args.ports)))
    else:
        raw = args.pci_pins.read_bytes()
        require(digest(raw) == args.pci_pins_sha256, 'PCI inventory bytes differ from the pinned source')
        print(json.dumps(collect(args.request, args.ledger.read_bytes(), json.loads(raw), args.port,
                     priority_nodes=args.priority_node, held_nodes=args.held_node,
                     priority_only=args.priority_only), allow_nan=False))


if __name__ == '__main__':
    main()
