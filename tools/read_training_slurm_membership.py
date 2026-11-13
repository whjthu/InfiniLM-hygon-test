"""Read live Slurm allocation and step membership against the frozen training ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def query(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip().splitlines()


def expand(expression):
    return query('scontrol', 'show', 'hostnames', expression)


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def snapshot(groups):
    parents = {}
    all_nodes = set()
    for parent, assigned in groups.items():
        expected = {node: group for group, nodes in assigned.items() for node in nodes}
        if len(expected) != sum(map(len, assigned.values())):
            raise ValueError(f'duplicate ledger node in {parent}')

        lines = query('squeue', '-h', '-j', parent, '-o', '%i|%T|%D|%N')
        if len(lines) != 1:
            raise ValueError(f'expected one live allocation for {parent}')
        handle, state, count, expression = lines[0].split('|')
        nodes = expand(expression)
        if handle != parent or len(set(nodes)) != len(nodes) or len(nodes) != int(count):
            raise ValueError(f'allocation identity or node count differs for {parent}')
        overlap = all_nodes.intersection(nodes)
        if overlap:
            raise ValueError('Slurm parents share allocated nodes')
        all_nodes.update(nodes)

        job = fields(query('scontrol', 'show', 'job', parent, '--oneliner')[0])
        dcu = int(dict(row.split('=', 1) for row in job['AllocTRES'].split(','))['gres/dcu'])
        steps = {}
        for line in query('scontrol', 'show', 'step', parent, '--oneliner'):
            step = fields(line)
            step_id = step.get('StepId', '')
            if not step_id.startswith(parent + '.') or not step_id.rsplit('.', 1)[-1].isdigit():
                continue
            # Nodes is a count. NodeList is the actual Slurm host expression.
            members = expand(step['NodeList'])
            if len(members) != int(step['Nodes']) or len(set(members)) != len(members):
                raise ValueError(f'step node count differs for {step_id}')
            if step_id in steps:
                raise ValueError(f'duplicate live step {step_id}')
            steps[step_id] = dict(state=step['State'], nodes=len(members),
                                  groups=sorted({expected.get(node, 'OUTSIDE_LEDGER') for node in members}),
                                  allocated_TRES=step.get('TRES'))
        parents[parent] = dict(state=state, nodes=len(nodes), allocated_DCU=dcu,
                               ledger_exact=set(nodes) == set(expected), active_steps=steps)
    return dict(parents=parents, distinct_nodes=len(all_nodes), observed_epoch=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    ledger = json.loads(raw)
    groups = {parent: {name: group['allocations'][parent] for name, group in ledger['groups'].items()}
              for parent in ('1087656', '1103606')}
    result = snapshot(groups)
    result['ledger_SHA256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
