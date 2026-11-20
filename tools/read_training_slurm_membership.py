"""Read live Slurm allocation and step membership against the frozen training ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def query(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip().splitlines()


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def allocated_dcu(value):
    """Read a DCU TRES or return unknown for an unreported device allocation."""
    if value in (None, '', '(null)', 'N/A'):
        return None
    resources = {}
    for entry in value.split(','):
        key, separator, amount = entry.partition('=')
        if not separator or not key or not amount or key in resources:
            raise ValueError(f'invalid Slurm TRES inventory: {value}')
        resources[key] = amount
    generic = resources.get('gres/dcu')
    typed = [amount for key, amount in resources.items() if key.startswith('gres/dcu:')]
    if any(not re.fullmatch(r'[0-9]+', amount) for amount in typed + ([generic] if generic is not None else [])):
        raise ValueError(f'invalid Slurm DCU count: {value}')
    if generic is not None:
        total = int(generic)
        if typed and sum(map(int, typed)) != total:
            raise ValueError(f'inconsistent generic and typed Slurm DCU counts: {value}')
        return total
    if typed:
        return sum(map(int, typed))
    if 'cpu' in resources and not any(key.startswith('gres/') for key in resources):
        return 0
    return None


def completed_numbered_step(columns, values, expected_step, *, scope):
    """Read a successful numbered step from its actual accounting column layout."""
    if not (isinstance(columns, list) and isinstance(values, list) and len(columns) == len(values)
            and all(isinstance(column, str) for column in columns)):
        raise ValueError(f'invalid {scope} terminal accounting row')
    keys = [column.split('%', 1)[0] for column in columns]
    if len(set(keys)) != len(keys) or not {'JobIDRaw', 'State', 'ExitCode', 'AllocTRES'} <= set(keys):
        raise ValueError(f'unexpected {scope} terminal accounting columns')
    row = dict(zip(keys, values, strict=True))
    step = row['JobIDRaw']
    if not (isinstance(step, str) and re.fullmatch(r'[0-9]+\.[0-9]+', step)
            and step == expected_step and row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0'):
        raise ValueError(f'{scope} evidence has no successful Slurm step matching the run')
    return row


def numbered_step_usage(steps, parent_nodes, cards_per_node):
    """Bound the number of distinct assigned cards using Slurm membership only."""
    groups = []
    covered = set()
    all_step_nodes = set()
    unknown = set()
    cpu_only = set()
    node_bounds = {node: [0, 0] for node in parent_nodes}
    for step in steps.values():
        nodes = set(step['member_nodes'])
        all_step_nodes.update(nodes)
        count = step['allocated_DCU']
        if count == 0:
            cpu_only.update(nodes)
            continue
        if count is None:
            unknown.update(nodes)
        covered.update(nodes)
        counts = [count]
        capacity = cards_per_node * len(nodes)
        upper = capacity if count is None else count
        node_lower = 0 if count is None else max(0, count - capacity + cards_per_node)
        node_upper = cards_per_node if count is None else min(count, cards_per_node)
        for node in nodes & parent_nodes:
            lower_bound, upper_bound = node_bounds[node]
            node_bounds[node] = [max(lower_bound, node_lower),
                                 min(cards_per_node, upper_bound + node_upper)]
        remaining = []
        # Steps sharing a node may share physical devices. Union their bounds.
        for member_nodes, member_counts, member_upper in groups:
            if member_nodes & nodes:
                nodes.update(member_nodes)
                counts.extend(member_counts)
                upper += member_upper
            else:
                remaining.append((member_nodes, member_counts, member_upper))
        remaining.append((nodes, counts, upper))
        groups = remaining
    if not all_step_nodes <= parent_nodes:
        return dict(minimum_assigned_DCU=None, maximum_assigned_DCU=None,
                    nodes_without_numbered_DCU_step=None, unknown_DCU_allocation_nodes=sorted(unknown),
                    assigned_DCU_bounds_by_node=None)
    lower = sum(max((count for count in counts if count is not None), default=0)
                for _, counts, _ in groups)
    upper = sum(min(bound, cards_per_node * len(nodes)) for nodes, _, bound in groups)
    lower = max(lower, sum(bounds[0] for bounds in node_bounds.values()))
    upper = min(upper, sum(bounds[1] for bounds in node_bounds.values()))
    return dict(minimum_assigned_DCU=lower, maximum_assigned_DCU=upper,
                nodes_without_numbered_DCU_step=sorted(parent_nodes - covered),
                unknown_DCU_allocation_nodes=sorted(unknown),
                nodes_with_only_CPU_numbered_steps=sorted((cpu_only - covered) & parent_nodes),
                assigned_DCU_bounds_by_node={node: node_bounds[node] for node in sorted(parent_nodes)})


def group_coverage(expected, usage, cards_per_node, valid_parent):
    """Report which nodes need follow-up before any new step can use them."""
    bounds = usage['assigned_DCU_bounds_by_node'] if valid_parent else None
    unknown = set(usage['unknown_DCU_allocation_nodes'])
    cpu_only = set(usage.get('nodes_with_only_CPU_numbered_steps', []))
    groups = {}
    for node, name in sorted(expected.items()):
        nodes = groups.setdefault(name, {})
        if bounds is None:
            nodes[node] = {'numbered_step_assigned_DCU_bounds': None, 'state': 'UNVERIFIED_PARENT'}
            continue
        lower, upper = bounds[node]
        if lower == cards_per_node:
            state = 'FULLY_ASSIGNED'
        elif node in unknown:
            state = 'UNREPORTED_DCU_ALLOCATION'
        elif upper == 0:
            state = 'CPU_STEP_ONLY' if node in cpu_only else 'NO_NUMBERED_STEP'
        elif upper < cards_per_node:
            state = 'PARTIALLY_ASSIGNED'
        else:
            state = 'ASSIGNMENT_UNCERTAIN'
        nodes[node] = {'numbered_step_assigned_DCU_bounds': [lower, upper], 'state': state}
    return groups


def dispatch_candidates(parents):
    """Order node checks before backfill, using Slurm bounds only."""
    rank = {'NO_NUMBERED_STEP': 0, 'CPU_STEP_ONLY': 1, 'PARTIALLY_ASSIGNED': 2}
    candidates = []
    states = {}
    for parent, observation in parents.items():
        for group, nodes in observation['coverage_by_group_and_node'].items():
            for node, row in nodes.items():
                state = row['state']
                states[state] = states.get(state, 0) + 1
                if state in rank:
                    lower, upper = row['numbered_step_assigned_DCU_bounds']
                    candidates.append(dict(parent=parent, group=group, node=node, state=state,
                                           numbered_step_assigned_DCU_bounds=[lower, upper]))
    # Start full-node checks first. For partial nodes, inspect the largest gaps first.
    candidates.sort(key=lambda row: (rank[row['state']], row['numbered_step_assigned_DCU_bounds'][1],
                                     row['parent'], row['group'], row['node']))
    return dict(nodes_by_state=states, ordered_node_checks=candidates,
                physical_admission_checked=False, busy_measurement=None)


def snapshot(groups, *, owner_uid=None, run_query=None):
    if run_query is None:
        run_query = query

    def expand(expression):
        return run_query('scontrol', 'show', 'hostnames', expression)

    started = time.monotonic()
    parents = {}
    all_nodes = set()
    for parent, assigned in groups.items():
        expected = {node: group for group, nodes in assigned.items() for node in nodes}
        if len(expected) != sum(map(len, assigned.values())):
            raise ValueError(f'duplicate ledger node in {parent}')

        lines = run_query('squeue', '-h', '-j', parent, '-o', '%i|%T|%D|%N')
        if len(lines) != 1:
            raise ValueError(f'expected one live allocation for {parent}')
        handle, state, count, expression = lines[0].split('|')
        nodes = expand(expression)
        if handle != parent or len(set(nodes)) != len(nodes) or len(nodes) != int(count):
            raise ValueError(f'allocation identity or node count differs for {parent}')
        parent_nodes = set(nodes)
        overlap = all_nodes.intersection(parent_nodes)
        if overlap:
            raise ValueError('Slurm parents share allocated nodes')
        all_nodes.update(parent_nodes)

        job_rows = run_query('scontrol', 'show', 'job', parent, '--oneliner')
        if len(job_rows) != 1:
            raise ValueError(f'expected one live job record for {parent}')
        job = fields(job_rows[0])
        owner = re.fullmatch(r'[^()]+\(([0-9]+)\)', job.get('UserId', ''))
        owned = None if owner_uid is None else bool(owner and int(owner[1]) == owner_uid)
        dcu = allocated_dcu(job.get('AllocTRES'))
        if job.get('JobId') != parent or job.get('JobState') != state or dcu is None or dcu != len(nodes) * 8:
            raise ValueError(f'parent identity, state or DCU allocation differs for {parent}')
        steps = {}
        step_rows = run_query('scontrol', 'show', 'step', parent, '--oneliner')
        if not step_rows:
            raise ValueError(f'live step inventory unavailable for {parent}')
        for line in step_rows:
            step = fields(line)
            step_id = step.get('StepId', '')
            if step_id in (parent + '.batch', parent + '.extern'):
                continue
            if not re.fullmatch(re.escape(parent) + r'\.[0-9]+', step_id):
                raise ValueError(f'unrecognized live step under {parent}')
            # Nodes is a count. NodeList is the actual Slurm host expression.
            members = expand(step['NodeList'])
            if len(members) != int(step['Nodes']) or len(set(members)) != len(members):
                raise ValueError(f'step node count differs for {step_id}')
            if step_id in steps:
                raise ValueError(f'duplicate live step {step_id}')
            step_groups = sorted({expected[node] for node in members if node in expected})
            outside_ledger = sorted(set(members).difference(expected))
            outside_parent = sorted(set(members).difference(parent_nodes))
            resources = step.get('TRES')
            if resources in (None, '', '(null)', 'N/A'):
                resources = step.get('AllocTRES', resources)
            step_dcu = allocated_dcu(resources)
            if step_dcu is not None and step_dcu > len(members) * 8:
                raise ValueError(f'impossible DCU placement for {step_id}')
            steps[step_id] = dict(
                state=step['State'], nodes=len(members), member_nodes=sorted(members),
                groups=step_groups, outside_ledger_nodes=outside_ledger,
                outside_parent_nodes=outside_parent, allocated_TRES=resources, allocated_DCU=step_dcu,
                membership_qualified=(len(step_groups) == 1 and not outside_ledger and not outside_parent),
            )
        usage = numbered_step_usage(steps, parent_nodes, cards_per_node=8)
        ledger_exact = parent_nodes == set(expected)
        step_locations_exact = (ledger_exact and all(
            step['member_nodes'] and not step['outside_parent_nodes'] for step in steps.values()
        ))
        parents[parent] = dict(
            state=state, nodes=len(nodes), allocated_DCU=dcu,
            ledger_exact=ledger_exact, owned_by_observer=owned, step_locations_exact=step_locations_exact,
            active_steps=steps, numbered_step_usage=usage,
            coverage_by_group_and_node=group_coverage(expected, usage, cards_per_node=8,
                                                      valid_parent=state == 'RUNNING' and step_locations_exact),
        )
    membership_qualified = all(
        parent['state'] == 'RUNNING' and parent['ledger_exact']
        and all(step['membership_qualified'] for step in parent['active_steps'].values())
        for parent in parents.values()
    )
    usage = [row['numbered_step_usage'] for row in parents.values()]

    def combined(field):
        counts = [row[field] for row in usage]
        return None if None in counts else sum(counts)

    return dict(parents=parents, distinct_nodes=len(all_nodes),
                allocated_DCU=sum(parent['allocated_DCU'] for parent in parents.values()),
                minimum_numbered_step_assigned_DCU=combined('minimum_assigned_DCU'),
                maximum_numbered_step_assigned_DCU=combined('maximum_assigned_DCU'),
                dispatch_candidates=dispatch_candidates(parents),
                busy_measurement=None, membership_qualified=membership_qualified,
                observed_epoch=time.time(), observation_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    ledger = json.loads(raw)
    groups = {parent: {name: group['allocations'][parent] for name, group in ledger['groups'].items()}
              for parent in ('1087656', '1103606')}
    result = snapshot(groups, owner_uid=os.getuid() if hasattr(os, 'getuid') else None)
    result['ledger_SHA256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
