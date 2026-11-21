"""Read pinned training80 cases and prepare a shared reconnect admission check."""

import argparse
import ast
import json
from pathlib import Path

from report_training80_backlog import (
    REPO, TRAINING80_LEDGER, has_original_intent, hex_value, invocation, need, pin_file,
)
from verify_training80_manifest import verify_package_members


WORKLIST = REPO / 'coordination/training/training80-reconnect-worklist-v1.json'


def source_ref(task, source, name):
    if name == 'claim':
        return task.get('claim') or dict(path=source + '/CLAIM.json', SHA256=task['claim_SHA256'])
    return (task.get('source_manifest') or task.get('manifest') or
            dict(path=source + '/SOURCE_FILES.json', SHA256=task['source_manifest_SHA256']))


def pinned_target_probe(repo, probe):
    if probe is not None:
        pin_file(repo, probe['entry'], ('tools/',))
        for dependency in probe['dependencies']:
            pin_file(repo, dependency, ('tools/', 'coordination/training/', 'code/bw1000/', 'results/bw1000/'))
    return probe


def target_cpu_controller(repo, task, candidate):
    resources = task['resources']
    need(type(resources['dcu']) is int and resources['dcu'] == 0 and
         type(resources['cpus']) is int and resources['cpus'] > 0 and
         type(resources['memory_MiB']) is int and resources['memory_MiB'] > 0 and
         task['exact_plan']['optimizer_updates'] == 0 and task['training_READY'] is False,
         'a CPU prerequisite cannot reserve accelerators or count as training')
    entry = task['controller']
    raw = pin_file(repo, entry, ('tools/',))
    tree = ast.parse(raw, filename=entry['path'])
    cli = {node.args[0].value for node in ast.walk(tree)
           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr == 'add_argument' and node.args
           and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
    need(cli == {'--framework-root', '--publication'}, 'frozen CPU controller argument binding changed')
    for member in task['dependencies']:
        pin_file(repo, member, ('tools/', 'code/bw1000/', 'coordination/training/'))
    pin_file(repo, task['source_predecessor'], ('coordination/tasks/',))
    command = dict(kind='authenticated_host_python', arguments=[
        '--framework-root', '<verified_framework_root>', '--publication', '<verified_publication>',
    ])
    claim = dict(request=task['request'], parent=resources['parent'], node=resources['node'],
                 cpus=resources['cpus'], device_count=0, optimizer_updates=0)
    return claim, entry, candidate['task'], command


def packaged_candidate(repo, task, candidate):
    source = candidate['source']
    claim_ref = source_ref(task, source, 'claim')
    manifest_ref = source_ref(task, source, 'manifest')
    launcher = candidate['launcher']
    need(Path(claim_ref['path']).parent == Path(manifest_ref['path']).parent == Path(source),
         'worklist source package differs from task')
    manifest = verify_package_members(repo, manifest_ref, required_files=[claim_ref, launcher])
    claim = json.loads(pin_file(repo, claim_ref, ('code/bw1000/',)))
    command = invocation(repo, dict(launch=dict(launcher=launcher, publication_commit='<verified_publication>',
                                              request=claim['request'], claim=claim_ref)))
    need(command is not None, 'frozen candidate requires a manual launcher binding')
    return claim, launcher, manifest, command


def prepare(repo, config, ledger_raw):
    repo = repo.resolve()
    ledger = json.loads(ledger_raw)
    placement = {node: parent for zone in ledger['groups'].values()
                 for parent, nodes in zone['allocations'].items() for node in nodes}
    need(len(placement) == 80, 'expected the frozen 80-node ledger')
    result = []
    checked_requests = set()
    ports = set()
    required_nodes = set()
    for candidate in config['candidates']:
        task = json.loads(pin_file(repo, candidate['task'], ('coordination/tasks/',)))
        need(task['state'] == candidate['expected_task_state'], 'task status changed; rebuild worklist from the original')
        mode = candidate.get('mode', 'package')
        need(mode in ('package', 'target_cpu_controller'), 'unsupported reconnect candidate kind')
        selector = target_cpu_controller if mode == 'target_cpu_controller' else packaged_candidate
        claim, launcher, manifest, command = selector(repo, task, candidate)
        request = claim['request']
        need(hex_value(request, 32) and request == task['request'] and request not in checked_requests,
             'missing, repeated or mismatched task request')
        checked_requests.add(request)
        node, parent = claim['node'], claim['parent']
        need(placement.get(node) == parent, 'candidate node differs from its assigned Slurm parent')
        cards = claim.get('device_count', claim.get('GRES'))
        need(type(cards) is int and 0 <= cards <= 8, 'invalid candidate accelerator count')
        port = claim.get('master_port')
        need(port is None or type(port) is int and 1 <= port <= 65535, 'invalid frozen rendezvous port')
        gates = candidate['gates']
        need(isinstance(gates, list) and gates and all(isinstance(gate, str) and gate.strip() for gate in gates),
             'candidate needs concrete reconnect controls')
        attempted = has_original_intent(repo, [candidate['attempt_receipt']])
        predecessors = []
        for prior in candidate.get('blocking_original_requests', []):
            payload = json.loads(pin_file(repo, prior, ('results/bw1000/',)))
            need(payload.get('request') == prior['request'] and hex_value(prior['request'], 32) and
                 payload.get('replay_allowed') is False and prior['request'] != request,
                 'original request reference permits replay or does not bind its receipt')
            predecessor = dict(request=prior['request'], receipt_SHA256=prior['SHA256'], replay_allowed=False)
            probe = pinned_target_probe(repo, prior.get('target_reconciliation_probe'))
            if probe is not None:
                predecessor['target_reconciliation_probe'] = probe
            predecessors.append(predecessor)
        target_probe = pinned_target_probe(repo, candidate.get('target_read_only_preflight'))
        completion_reader = pinned_target_probe(repo, candidate.get('target_completion_reader'))
        updates = claim.get('optimizer_updates', claim.get('planned_updates', 0))
        need(type(updates) is int and updates >= 0 and (cards != 0 or updates == 0), 'invalid planned optimizer updates')
        if not attempted:
            required_nodes.add(node)
            if port is not None:
                ports.add(port)
        result.append(dict(case=task['case'], request=request, parent=parent, node=node, requested_DCU=cards,
                           requested_CPUs=claim['cpus'], planned_optimizer_updates=updates,
                           target_preflight_state='RECONCILE_ORIGINAL_ATTEMPT' if attempted else 'WAITING_AUTHENTICATED_TARGET',
                           current_GPU_GO=claim.get('GPU_GO') is True, current_CPU_GO=claim.get('CPU_GO') is True,
                           candidate_ready_verified=False, frozen_rendezvous_port=port,
                           gates=gates, blocking_original_requests=predecessors, manifest=manifest,
                           launcher=launcher, frozen_invocation_after_gates=None if attempted else command,
                           attempt_receipt=candidate['attempt_receipt'],
                           target_read_only_preflight=None if attempted else target_probe,
                           target_completion_reader=completion_reader))
    flags = [value for port in sorted(ports) for value in ('--port', str(port))]
    return dict(collector='tools/collect_training80_dispatch_snapshot.py', collector_port_arguments=flags,
                candidate_nodes_needing_current_probe=sorted(required_nodes),
                configured_nodes=len(placement), configured_DCU=len(placement)*8,
                current_allocation_DCU=None, current_available_DCU=None, measured_busy_percent=None,
                newly_verified_training_ready=0, actual_submissions=0, candidates=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worklist', type=Path, default=WORKLIST)
    args = parser.parse_args()
    plan = prepare(REPO, json.loads(args.worklist.read_bytes()), TRAINING80_LEDGER.read_bytes())
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
