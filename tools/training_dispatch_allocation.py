"""Read the original or surviving Slurm allocation for dispatch admission."""

from hashlib import sha256
import json
from pathlib import Path

from sample_training80_busy import node_map, require


FULL_KIND = 'training80-dispatch-v1'
SURVIVOR_KIND = 'training40-dispatch-v1'
SURVIVOR_LEDGER = (Path(__file__).resolve().parents[1] / 'coordination' / 'training' /
                   'training40-survivor1103606-node-task-map-v1.json')
SURVIVOR_LEDGER_SHA256 = '52ef543081bf351b70da3b50a68e84fdfed9a2522f3e26f582e3eb6f6623181e'
SURVIVOR_PARENT = '1103606'


def allocation(raw):
    ledger = json.loads(raw)
    if ledger.get('state') == 'ONE_LIVE_PARENT_DCU320':
        require(sha256(raw).hexdigest() == SURVIVOR_LEDGER_SHA256, 'surviving allocation ledger bytes differ')
        require(ledger['parent'] == SURVIVOR_PARENT, 'wrong surviving Slurm parent')
        groups = {SURVIVOR_PARENT: {zone: row['nodes'] for zone, row in ledger['groups'].items()}}
        owners = {node: (SURVIVOR_PARENT, zone) for zone, names in groups[SURVIVOR_PARENT].items()
                  for node in names}
        require(len(owners) == sum(map(len, groups[SURVIVOR_PARENT].values())) == ledger['node_count'] == 40
                and set(owners) == set(ledger['nodes']) and ledger['card_count'] == 320,
                'surviving node roster differs')
        return SURVIVOR_KIND, groups, owners
    owners = node_map(ledger)
    groups = {job['JobId']: {zone: row['allocations'][job['JobId']]
                            for zone, row in ledger['groups'].items()} for job in ledger['jobs']}
    require(set(groups) == {parent for parent, _ in owners.values()} and len(ledger['jobs']) == len(groups),
            'parent job records differ from the node mapping')
    return FULL_KIND, groups, owners


def survivor_placements(membership, probes, owners):
    parent = membership['parents'][SURVIVOR_PARENT]
    require(parent['state'] == 'RUNNING' and parent['ledger_exact'] is True and parent['nodes'] == 40
            and parent['allocated_DCU'] == 320 and parent['owned_by_observer'] is True
            and parent['step_locations_exact'] is True, 'surviving parent ownership or allocation differs')
    active_nodes = set()
    for step in parent['active_steps'].values():
        members = set(step['member_nodes'])
        require(members and members <= owners.keys(), 'step escaped the surviving allocation')
        active_nodes.update(members)
    coverage = parent['coverage_by_group_and_node']
    require(set(probes) == owners.keys() and set(coverage) == {zone for _, zone in owners.values()},
            'surviving device or group coverage differs')
    return {node: (zone, job, node not in active_nodes and coverage[zone][node]['state'] == 'NO_NUMBERED_STEP'
                   and coverage[zone][node]['numbered_step_assigned_DCU_bounds'] == [0, 0])
            for node, (job, zone) in owners.items()}
