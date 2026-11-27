"""Read a frozen Slurm allocation for dispatch admission."""

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from read_training_slurm_membership import allocated_dcu, completed_numbered_step
from sample_training80_busy import node_map, require


FULL_KIND = 'training80-dispatch-v1'
SURVIVOR_KIND = 'training40-dispatch-v1'
REPLACEMENT_KIND = 'training80-replacement1194900-dispatch-v1'
DEVICE_RECEIPT_KINDS = (FULL_KIND, SURVIVOR_KIND, REPLACEMENT_KIND)
SURVIVOR_LEDGER = (Path(__file__).resolve().parents[1] / 'coordination' / 'training' /
                   'training40-survivor1103606-node-task-map-v1.json')
REPLACEMENT_LEDGER = SURVIVOR_LEDGER.with_name('training80-replacement1194900-node-task-map-v1.json')
REPLACEMENT_LEDGER_SHA256 = '11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d'
SURVIVOR_LEDGER_SHA256 = '52ef543081bf351b70da3b50a68e84fdfed9a2522f3e26f582e3eb6f6623181e'
SURVIVOR_PARENT = '1103606'
STALE_STEP_PROBE_SCOPE = {
    '1103606.1644': ('f13r4n06', 'training80-dcu-standby-af719c997048-f13r4n06', 'cpu=8,gres/dcu=8,mem=16G,node=1'),
    '1103606.1646': ('f13r4n09', 'training80-dcu-standby-af719c997048-f13r4n09', 'cpu=8,gres/dcu=8,mem=16G,node=1'),
    '1194900.188': ('f12r3n03', 'training80-dcu-standby-c5db5fbaf6cc-f12r3n03', 'cpu=8,gres/dcu=8,mem=16G,node=1'),
    '1194900.189': ('f13r4n03', 'training80-admit-b0c43a4cae196c1f4f0a4a9ee71c0950', 'cpu=1,mem=256M,node=1'),
}
STALE_STEP_PROBE_SCHEMA = 'training80-four-numbered-timeouts-physical-probe-only-v1'
STALE_STEP_TERMINAL_SCHEMA = 'training80-timeout-physical-probe-with-accounting-v2'
PROBE_COLUMNS = 'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList%100,AllocTRES%100,Start,End'.split(',')


def successful_probe_accounting(values, step, node, label):
    row = completed_numbered_step(PROBE_COLUMNS, values, step, scope='stale-step physical probe')
    require(row['NodeList'] == node and row['JobName'] == label and allocated_dcu(row['AllocTRES']) == 0 and
            set(row['AllocTRES'].split(',')) == {'cpu=1', 'mem=256M', 'node=1'} and
            len(row['AllocTRES'].split(',')) == 3 and row['ElapsedRaw'].isascii() and
            row['ElapsedRaw'].isdecimal() and 0 <= int(row['ElapsedRaw']) <= 60 and
            all(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', row[k]) for k in ('Start', 'End')) and
            row['Start'] <= row['End'], 'stale-step probe accounting resource scope differs')
    return row


def stale_step_candidate(membership, handle):
    """Require exact singleton TIMEOUT in an otherwise empty node's raw inventory."""
    node, _, tres = STALE_STEP_PROBE_SCOPE[handle]
    parent = membership['parents'].get(handle.split('.')[0], {})
    if not (parent.get('state') == 'RUNNING' and parent.get('ledger_exact') is True and
            parent.get('owned_by_observer') is True and parent.get('step_locations_exact') is True):
        return False
    local = {step: row for step, row in parent['active_steps'].items() if node in row['member_nodes']}
    step = local.get(handle, {})
    return (len(local) == 1 and step.get('member_nodes') == [node] and step.get('state') == 'TIMEOUT' and
            step.get('membership_qualified') is True and step.get('allocated_TRES') == tres)


def confirm_stale_step(run_query, handle):
    """Return a fail-closed read-only witness using rc0 and quiet stderr at the call site."""
    node, label, tres = STALE_STEP_PROBE_SCOPE[handle]
    parent = handle.split('.')[0]
    proof = dict(controller_exact_timeout=False, queue_clear=False, statistics_empty=False)
    try:
        matches = []
        for line in run_query('scontrol', 'show', 'step', handle, '--oneliner'):
            terms = [part.split('=', 1) for part in line.split() if '=' in part]
            fields = dict(terms)
            require(re.fullmatch(re.escape(parent) + r'\.(?:[0-9]+|batch|extern)', fields.get('StepId', '')),
                    'unexpected exact-step controller row')
            if fields['StepId'] == handle:
                require(len(terms) == len(fields), 'duplicate exact-step controller field')
                matches.append(fields)
        proof['controller_exact_timeout'] = len(matches) == 1 and all(
            matches[0].get(key) == value for key, value in
            dict(StepId=handle, Name=label, NodeList=node, State='TIMEOUT', TRES=tres).items())
        proof['queue_clear'] = True
        seen = set()
        for line in run_query('squeue', '--steps', '-h', '-j', parent, '-o', '%i|%j|%N'):
            step, name, members = line.split('|')
            require(re.fullmatch(re.escape(parent) + r'\.(?:[0-9]+|batch|extern)', step) and name and step not in seen,
                    'unexpected or duplicate live step in queue')
            seen.add(step)
            if step.endswith(('.batch', '.extern')):
                continue
            nodes = ([members] if re.fullmatch(r'f[0-9]+r[0-9]+n[0-9]+', members) else
                     run_query('scontrol', 'show', 'hostnames', members))
            require(nodes and all(re.fullmatch(r'f[0-9]+r[0-9]+n[0-9]+', name) for name in nodes),
                    'unproven live queue node list')
            if step == handle or node in nodes:
                proof['queue_clear'] = False
        proof['statistics_empty'] = not run_query('sstat', '-nP', '-j', handle, '-o', 'JobID,NTasks,MaxRSS')
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        proof['query_complete'] = False
        proof['error_type'] = type(error).__name__
    else:
        proof['query_complete'] = True
    return proof


def stale_step_verified(proof):
    return isinstance(proof, dict) and all(proof.get(key) is True for key in
        ('controller_exact_timeout', 'queue_clear', 'statistics_empty', 'query_complete', 'raw_singleton_timeout'))


def stale_step_projection(fleet):
    """Read dispatch-side metadata without promoting any projected node to launch clearance."""
    projection = fleet.get('stale_step_probe_only')
    if projection is None:
        return {}
    schema = projection.get('schema')
    require(fleet.get('snapshot_kind') == REPLACEMENT_KIND and schema in (STALE_STEP_PROBE_SCHEMA, STALE_STEP_TERMINAL_SCHEMA),
            'stale-step probe scope differs from the active fleet')
    rows = projection['nodes']
    require({row['handle']: node for node, row in rows.items()} ==
            {handle: scope[0] for handle, scope in STALE_STEP_PROBE_SCOPE.items()} and len(rows) == len(STALE_STEP_PROBE_SCOPE),
            'stale-step probe witness escaped the frozen roster')
    for node, row in rows.items():
        handle = row['handle']
        complete = (row.get('probe_eligible') is True and row.get('probe_returned_zero') is True and
                    stale_step_verified(row.get('initial')) and stale_step_verified(row.get('final')))
        if schema == STALE_STEP_TERMINAL_SCHEMA:
            device = fleet['nodes'][node]
            terminal = row.get('probe_accounting')
            if terminal is not None:
                step = device['device_probe_step']
                request = row['probe_request']
                require(isinstance(step, str) and re.fullmatch(re.escape(handle.split('.')[0]) + r'\.[0-9]+', step) and
                        isinstance(request, str) and re.fullmatch(r'[a-f0-9]{32}', request), 'invalid probe identity')
                successful_probe_accounting(terminal, step, node, 'training80-admit-' + request)
            complete = complete and terminal is not None and bool(device['ports_checked'])
        require(row.get('physical_probe_witness_complete') is complete and
                (not complete or stale_step_candidate(fleet['membership'], handle)),
                'stale-step probe completion differs from raw membership or exact observations')
    return rows


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
    kind = FULL_KIND
    if ledger.get('state') == 'LIVE_REPLACEMENT_DCU640':
        require(sha256(raw).hexdigest() == REPLACEMENT_LEDGER_SHA256, 'replacement allocation ledger bytes differ')
        owners = node_map(ledger, {SURVIVOR_PARENT, '1194900'})
        kind = REPLACEMENT_KIND
    else:
        owners = node_map(ledger)
    groups = {job['JobId']: {zone: row['allocations'][job['JobId']]
                            for zone, row in ledger['groups'].items()} for job in ledger['jobs']}
    require(set(groups) == {parent for parent, _ in owners.values()} and len(ledger['jobs']) == len(groups),
            'parent job records differ from the node mapping')
    return kind, groups, owners


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
