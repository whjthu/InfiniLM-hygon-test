"""Finite, evidence-bound tuning queue; does not launch arbitrary experiment code.

Root dispatches reserved packets through the model's frozen launch_once adapter.
The dispatcher can retire a claimed direct request before transport starts.
Expiry, connection errors and local process exits are never terminal evidence.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from uuid import uuid4

from training_dispatch_allocation import REPLACEMENT_KIND, REPLACEMENT_LEDGER, allocation

ACTIVE = ('RESERVED', 'DISPATCHING', 'SUBMITTED', 'UNKNOWN')
DIRECT_STAGE = 'training80-direct'
DIRECT_CLOSED_STATES = ('TERMINAL', 'RETIRED_PRELAUNCH')
SNAPSHOT_MAX_AGE_S = 60
RUNG_PAIR_COUNTS = (3, 9, 27)
STAGES = ('source', 'CPU', 'component', 'model', 'quality', 'service')
TRAINING80_LEDGER = (Path(__file__).resolve().parents[1] / 'coordination' /
                     'training' / 'training80-active-node-task-map-v1.json')

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def expand(campaign):
    """Deterministic bounded Cartesian space; proposals confer no run permission."""
    out = []
    for family in campaign['families']:
        axes = family['space']
        require(all(isinstance(v, list) and v for v in axes.values()), 'empty parameter axis')
        require(math.prod(map(len, axes.values())) <= family['max_candidates'], 'space exceeds declared budget')
        require(set(family['baseline']) == set(axes), 'baseline axes differ')
        require(all(v in axes[k] for k, v in family['baseline'].items()), 'baseline outside search space')
        for values in itertools.product(*(axes[k] for k in sorted(axes))):
            params = dict(zip(sorted(axes), values))
            identity = dict(model=family['model'], method=family['method'],
                            source=family['source'], workload=family['workload'], parameters=params)
            row = dict(identity, id=digest(identity), family=family['id'],
                       baseline=params == family['baseline'],
                       initial_stage=family['initial_stage'],
                       prerequisites=family['prerequisites'],
                       hypothesis=family['hypothesis'],
                       zone=family['zone'], nodes_required=family['nodes_required'],
                       memory_floor_bytes=family['memory_floor_bytes'],
                       stage_budgets_s=family['stage_budgets_s'])
            out.append(row)
    require(len({r['id'] for r in out}) == len(out), 'duplicate semantic candidates')
    return sorted(out, key=lambda r: r['id'])


def receive_snapshot(fleet, request_started_monotonic, request_id, raw_response_SHA256):
    """Stamp the authenticated transport receipt once, never when rereading a file.

    Cluster wall timestamps are descriptive only. The request roundtrip bounds
    the oldest possible fresh remote observation; a local monotonic clock measures
    subsequent age. A boot/clock-step anchor prevents reuse across local reboots.
    """
    now=time.monotonic()
    require(0<=now-request_started_monotonic<=60, 'snapshot transport exceeded60s')
    require(bool(request_id)and bool(raw_response_SHA256), 'original transport identity required')
    require('local_receipt'not in fleet, 'do not restamp an old snapshot')
    result=dict(fleet)
    result['local_receipt']=dict(received_monotonic=now,roundtrip_s=now-request_started_monotonic,
        boot_epoch_anchor=time.time()-now,request_id=request_id,raw_response_SHA256=raw_response_SHA256,
        payload_SHA256=digest(fleet),clock='local monotonic; remote wall time never subtracted')
    return result


def dispatch_window_open(reservation, now, boot_anchor):
    return (abs(boot_anchor - reservation['boot_epoch_anchor']) < 2 and
            reservation['received_monotonic'] <= now <= reservation['claim_by_monotonic'])


def fit_training80_run(previous, pools, allowed, needed):
    """Keep one assignment per occupied set in a ledger group of at most eight nodes."""
    extended = {}
    for occupied, assignments in previous.items():
        fits = []
        for parent, nodes in pools:
            available = sorted(allowed.intersection(nodes).difference(occupied))
            if len(available) >= needed:
                fits.append((len(available), parent, available))
        for _, parent, available in sorted(fits):
            for chosen in itertools.combinations(available, needed):
                extended.setdefault(occupied.union(chosen), assignments + ((parent, chosen),))
    return extended


def training80_backfill_zones(binding, zones):
    if 'backfill_zones' not in binding:
        return ()
    donors = binding['backfill_zones']
    require(type(donors) is list and donors and
            all(type(zone) is str and zone in zones and zone != binding['zone'] for zone in donors) and
            len(set(donors)) == len(donors), 'invalid training80 backfill zones')
    deadline = binding.get('deadline_s')
    require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > 0,
            'training80 backfill requires a bounded run deadline')
    return donors


def refit_single_node_loans(waiting, free, selected, placements):
    """Fill remaining single-node loans by moving only provisional single-node loans."""
    fixed_nodes = {node for job in selected if len(job['nodes']) > 1 for node in job['nodes']}
    node_by_run = {job['run']: job['nodes'][0] for job in selected if len(job['nodes']) == 1}
    run_by_node = {node: run for run, node in node_by_run.items()}
    single = [entry for entry in waiting if entry[1]['nodes_required'] == 1]
    choices = {
        job['run']: [node for donor in donors for parent in parents
                     for node in sorted(free.get((donor, parent), ()))
                     if node in allowed and node not in fixed_nodes]
        for job, _, allowed, parents, donors in single
    }

    def find_node(run, visited):
        for node in choices[run]:
            if node in visited:
                continue
            visited.add(node)
            owner = run_by_node.get(node)
            if owner is None or find_node(owner, visited):
                run_by_node[node] = run
                node_by_run[run] = node
                return True
        return False

    new_loans = []
    for job, _, _, _, _ in single:
        run = job['run']
        if run not in node_by_run and find_node(run, set()):
            new_loans.append(job)
    for job, binding, _, _, _ in single:
        node = node_by_run.get(job['run'])
        if node is not None:
            donor, parent, _ = placements[node]
            job.update(parent=parent, nodes=[node],
                       backfill=dict(donor_zone=donor, borrower_zone=binding['zone'],
                                     deadline_s=binding['deadline_s'], return_condition='actual_terminal_and_post_clean'))
    return new_loans


def refit_provisional_loans(waiting, free, selected, *, maximum_choices=8192):
    """Admit additional loans without evicting a provisionally selected run."""
    current = {job['run']: (job['backfill']['donor_zone'], job['parent'], tuple(job['nodes']))
               for job in selected}
    remaining = maximum_choices
    if len(current) == len(waiting):
        return []
    priority = {entry[0]['run']: pos for pos, entry in enumerate(waiting)}
    candidate_pools = {}
    for job, binding, allowed, parents, donors in waiting:
        candidate_pools[job['run']] = (binding['nodes_required'], [
            (donor, parent, sorted(allowed.intersection(free.get((donor, parent), ()))))
            for donor in donors for parent in parents
        ])

    def options(run):
        nodes_required, pools = candidate_pools[run]
        for donor, parent, nodes in pools:
            for chosen in itertools.combinations(nodes, nodes_required):
                yield donor, parent, chosen

    def place(pending, fixed, assigned):
        nonlocal remaining
        if not pending:
            return assigned
        run, *tail = pending
        owners = {node: owner for owner, (_, _, nodes) in assigned.items() for node in nodes}
        for option in options(run):
            if remaining <= 0:
                return None
            remaining -= 1
            nodes = option[2]
            if not fixed.isdisjoint(nodes):
                continue
            conflicts = {owners[node] for node in nodes if node in owners}
            trial = {owner: existing for owner, existing in assigned.items() if owner not in conflicts}
            trial[run] = option
            moved = place((*tail, *sorted(conflicts, key=priority.__getitem__)), fixed.union(nodes), trial)
            if moved is not None:
                return moved
        return None

    new_loans = []
    for job, _, _, _, _ in waiting:
        run = job['run']
        if run in current:
            continue
        if remaining <= 0:
            break
        updated = place((run,), frozenset(), current)
        if updated is not None:
            current = updated
            new_loans.append(job)
    for job, binding, _, _, _ in waiting:
        if job['run'] in current:
            donor, parent, nodes = current[job['run']]
            job.update(parent=parent, nodes=list(nodes),
                       backfill=dict(donor_zone=donor, borrower_zone=binding['zone'],
                                     deadline_s=binding['deadline_s'], return_condition='actual_terminal_and_post_clean'))
    return new_loans


def training80_inventory(fleet):
    """Bind fresh Slurm coverage and device observations to the same parent and group."""
    ledger_path = REPLACEMENT_LEDGER if fleet.get('snapshot_kind') == REPLACEMENT_KIND else TRAINING80_LEDGER
    raw = ledger_path.read_bytes()
    membership = fleet.get('membership') or {}
    require(membership.get('ledger_SHA256') == hashlib.sha256(raw).hexdigest(),
            'training80 membership must bind the active node ledger')
    _, groups, _ = allocation(raw)
    expected_parents = set(groups)
    parents = membership['parents']
    require(len(expected_parents) == 2 and set(parents) == expected_parents and
            membership['distinct_nodes'] == 80 and membership['allocated_DCU'] == 640,
            'training80 Slurm inventory is incomplete')
    placements = {}
    for parent, observation in parents.items():
        require(observation['state'] == 'RUNNING' and observation['ledger_exact'] is True and
                observation['nodes'] == 40 and observation['allocated_DCU'] == 320,
                'training80 parent allocation differs from the active ledger')
        require(observation.get('owned_by_observer') is True and observation.get('step_locations_exact') is True,
                'training80 parent ownership or numbered step locations are unverified')
        expected_members = {name for names in groups[parent].values() for name in names}
        active_nodes = set()
        for step in observation['active_steps'].values():
            members = set(step['member_nodes'])
            require(members and members <= expected_members, 'training80 step escaped its parent allocation')
            active_nodes.update(members)
        coverage = observation['coverage_by_group_and_node']
        require(set(coverage) == set(groups[parent]), 'training80 group coverage is incomplete')
        for zone, expected_nodes in groups[parent].items():
            require(set(coverage[zone]) == set(expected_nodes), 'training80 group node coverage drift')
            for name in expected_nodes:
                require(name not in placements, 'duplicate training80 ledger node')
                state = coverage[zone][name]
                slurm_clear = (name not in active_nodes and state['state'] == 'NO_NUMBERED_STEP' and
                               state['numbered_step_assigned_DCU_bounds'] == [0, 0])
                placements[name] = (zone, parent, slurm_clear)
    require(len(placements) == 80 and set(placements) == set(fleet['nodes']),
            'training80 device inventory differs from Slurm node coverage')
    return placements


class Queue:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA foreign_keys=ON;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, spec TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
          stage TEXT NOT NULL, state TEXT NOT NULL, binding TEXT NOT NULL,
          nodes TEXT NOT NULL, handle TEXT, result TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, at REAL NOT NULL,
          run TEXT, kind TEXT NOT NULL, data TEXT NOT NULL);
        ''')

    def event(self, ident, kind, data):
        self.db.execute('INSERT INTO events(at,run,kind,data) VALUES(?,?,?,?)',
                        (time.time(), ident, kind, canonical(data).decode()))

    def reservation(self, ident):
        row = self.db.execute('SELECT data FROM events WHERE run=? AND kind="reserved" ORDER BY seq DESC LIMIT 1',
                              (ident,)).fetchone()
        return json.loads(row['data']) if row else {}

    def reserved_parent(self, ident):
        return self.reservation(ident).get('parent')

    def dispatch_state(self, ident):
        row = self.db.execute('''
            SELECT state, nodes, handle, (SELECT data FROM events WHERE run=runs.id AND kind='reserved'
                     ORDER BY seq DESC LIMIT 1) AS reservation FROM runs WHERE id=?
        ''', (ident,)).fetchone()
        require(row is not None, 'unknown run')
        reservation = json.loads(row['reservation']) if row['reservation'] else {}
        return dict(state=row['state'], nodes=json.loads(row['nodes']),
                    handle=json.loads(row['handle']) if row['handle'] is not None else None,
                    reservation=reservation if row['state'] in ACTIVE else None)

    def seed(self, rows):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for row in rows:
                old=self.db.execute('SELECT spec FROM candidates WHERE id=?',(row['id'],)).fetchone()
                require(old is None or old[0]==canonical(row).decode(), 'candidate policy changed; version campaign explicitly')
                self.db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?)', (row['id'], canonical(row).decode()))
            self.event(None, 'seed', {'candidates': len(rows)})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def prepare(self, binding, repo):
        """Verify published exact bytes, workload, floor, stage and explicit root signoff.

        A prepared run is a different object from an expanded parameter proposal.
        This never calls the published launcher or claims an idle device.
        """
        require(binding['root_reviewed'] is True, 'root review missing')
        row = self.db.execute('SELECT spec FROM candidates WHERE id=?', (binding['candidate'],)).fetchone()
        require(row is not None, 'unknown candidate')
        spec = json.loads(row[0])
        require(binding['stage'] in ('component','model','quality','service'), 'GPU queue accepts only device stages; source/CPU use CPU-only adapters')
        require(type(binding['rung']) is int and 0<=binding['rung']<=2, 'predeclared rung0..2 required')
        require(binding['workload_SHA256'] == digest(spec['workload']), 'workload drift')
        require(binding['memory_floor_bytes'] == spec['memory_floor_bytes'], 'floor drift')
        require(binding['deadline_s'] > 0 and binding['deadline_s'] <= spec['stage_budgets_s'][binding['stage']], 'deadline exceeds campaign')
        require(binding['nodes_required'] == spec['nodes_required'], 'resource shape drift')
        require(binding['zone'] == spec['zone'], 'zone drift')
        require(binding['prerequisites_closed'] == spec['prerequisites'], 'prerequisites not explicitly closed')
        if 'backfill_zones' in binding:
            ledger = json.loads(TRAINING80_LEDGER.read_bytes())
            require(binding['zone'] in ledger['groups'], 'training80 backfill requires a ledger borrower group')
            training80_backfill_zones(binding, ledger['groups'])
        require(bool(binding['prerequisite_evidence']), 'prerequisite evidence absent')
        require(len(binding['artifacts']) >= 3, 'source/deps/claim artifacts required')
        require({'source', 'deps', 'claim', 'launcher'} <= {a['role'] for a in binding['artifacts']}, 'incomplete run bundle')
        commit = binding['publication_commit']
        require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'exact publication commit required')
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.PIPE)
        git('merge-base', '--is-ancestor', commit, 'refs/remotes/origin/bw1000-experiments')
        for item in binding['artifacts']:
            name = item['path']
            require(name.startswith('code/bw1000/') and '..' not in Path(name).parts and '\\' not in name, 'artifact outside own source')
            require(hashlib.sha256(git('show', commit + ':' + name)).hexdigest() == item['SHA256'], 'published artifact drift')
            require(sha(Path(repo)/name) == item['SHA256'], 'local artifact drift')
        require(binding['claim_semantics_checked'] is True, 'model adapter must validate exact claim semantics')
        require(bool(binding['inputoutput_binding_SHA256']) and bool(binding['dependency_binding_SHA256']), 'missing exact input/dependency binding')
        ident = digest(binding)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            attempts=self.db.execute('SELECT binding,state,result FROM runs WHERE candidate=? AND stage=?',
                                     (binding['candidate'],binding['stage'])).fetchall()
            require(not any(json.loads(v['binding'])['rung']==binding['rung'] for v in attempts),
                    'same candidate/stage/rung already attempted; reconcile original run')
            if binding['rung']>0:
                prior=[v for v in attempts if json.loads(v['binding'])['rung']==binding['rung']-1]
                require(len(prior)==1 and prior[0]['state']=='TERMINAL' and
                        json.loads(prior[0]['result']).get('promote') is True,
                        'next rung requires previous actual promoted terminal')
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
                (ident, binding['candidate'], binding['stage'], 'READY', canonical(binding).decode(), '[]', None, None, time.time()))
            self.event(ident, 'prepared', {'publication_commit': commit})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        return ident

    def plan(self, fleet, now=None):
        self.db.execute('BEGIN')
        try:
            return self._plan(fleet, now)
        finally:
            self.db.execute('ROLLBACK')

    def _plan(self, fleet, now=None):
        now = time.monotonic() if now is None else now
        boot_anchor = time.time()-time.monotonic()
        receipt=fleet['local_receipt']
        require(digest({k:v for k,v in fleet.items()if k!='local_receipt'})==receipt['payload_SHA256'], 'snapshot payload drift')
        require(abs(boot_anchor-receipt['boot_epoch_anchor'])<2, 'local reboot/clock change requires fresh receipt')
        age=now-receipt['received_monotonic']
        require(0 <= age and 0<=receipt['roundtrip_s'] and age+receipt['roundtrip_s']<=SNAPSHOT_MAX_AGE_S,
                'fresh idle snapshot required, max60s from original request')
        cards = fleet['total_cards']
        require(fleet['complete'] is True and cards in (136, 640), 'incomplete cluster inventory')
        placements = training80_inventory(fleet) if cards == 640 else None
        seen, free = set(), {}
        for name, node in fleet['nodes'].items():
            require(len(node['PCIs']) == 8 and len(set(node['PCIs'])) == 8, 'need eight unique PCI/node')
            require(not (seen & {(name, p) for p in node['PCIs']}), 'duplicate identity')
            seen.update((name,p) for p in node['PCIs'])
            zone, parent, slurm_clear = (node['zone'], '', True) if placements is None else placements[name]
            require(node['zone'] == zone and (placements is None or node.get('parent') == parent),
                    'training80 device node parent or group drift')
            device_clear = all(node.get(k) is True for k in ('KFD_clean','PID_clean','membership_valid'))
            observation_clear = (node.get('ports_clean') is True if placements is None else
                                 node.get('device_probe_observed') is True)
            if slurm_clear and node['active_steps'] == [] and device_clear and observation_clear:
                free.setdefault((zone, parent), []).append(name)
        require(len(seen) == cards, 'PCI inventory incomplete')
        expired = {}
        for row in self.db.execute('''
            SELECT runs.id, events.seq, events.data FROM runs
            JOIN events ON events.seq = (SELECT MAX(seq) FROM events WHERE run=runs.id AND kind='reserved')
            WHERE runs.state='RESERVED' AND runs.handle IS NULL ORDER BY runs.id
        '''):
            reservation = json.loads(row['data'])
            if reservation.get('dispatch_ticket') and not dispatch_window_open(reservation, now, boot_anchor):
                expired[row['id']] = row['seq']
        active_placeholders = ','.join('?' for _ in ACTIVE)
        queue_held_nodes = set()
        for row in self.db.execute(f'SELECT id,nodes FROM runs WHERE state IN ({active_placeholders})', ACTIVE):
            if row['id'] in expired:
                continue
            for name in json.loads(row['nodes']):
                queue_held_nodes.add(name)
                for zone in free.values():
                    if name in zone:
                        zone.remove(name)
        ready = []
        for r in self.db.execute('SELECT * FROM runs WHERE state IN ("READY", "RESERVED") ORDER BY created,id'):
            if r['state'] != 'READY' and r['id'] not in expired:
                continue
            b = json.loads(r['binding'])
            ready.append((r, b))
        selected = []
        selected_by_zone = {}
        assignments_by_zone = {}
        backfill_waiting = []
        zones = {zone for zone, _, _ in placements.values()} if placements is not None else set()
        # FIFO across approved families; no stale guessed performance priority.
        for r, b in ready:
            needed = b['nodes_required']
            require(type(needed) is int and needed > 0, 'positive whole-node request required')
            allowed = set(b['allowed_nodes'])
            job = dict(run=r['id'], candidate=r['candidate'], stage=r['stage'],
                       binding_SHA256=digest(b), publication_commit=b['publication_commit'])
            if placements is not None:
                ports = b.get('required_ports')
                if type(ports) is not list or any(type(p) is not int or not 1 <= p <= 65535 for p in ports):
                    continue
                required_ports = set(ports)
                allowed = {name for name in allowed if name in fleet['nodes'] and
                           isinstance(fleet['nodes'][name].get('ports_checked'), list) and
                           isinstance(fleet['nodes'][name].get('occupied_ports'), list) and
                           required_ports <= set(fleet['nodes'][name]['ports_checked']) and
                           not required_ports.intersection(fleet['nodes'][name]['occupied_ports'])}
                zone = b['zone']
                parents = b.get('allowed_parents', [])
                donors = training80_backfill_zones(b, zones)
                pools = [(parent, free.get((zone, parent), ())) for parent in parents]
                previous = assignments_by_zone.get(zone, {frozenset(): ()})
                extended = fit_training80_run(previous, pools, allowed, needed)
                if not extended:
                    if donors:
                        backfill_waiting.append((job, b, allowed, parents, donors))
                    continue
                assignments_by_zone[zone] = extended
            else:
                chosen = sorted(allowed.intersection(free.get((b['zone'], ''), ())))[:needed]
                if len(chosen) < needed:
                    continue
                for name in chosen:
                    free[(b['zone'], '')].remove(name)
            if placements is not None:
                selected_by_zone.setdefault(zone, []).append(job)
            else:
                job['nodes'] = chosen
            selected.append(job)
        for zone, jobs in selected_by_zone.items():
            assignments = next(iter(assignments_by_zone[zone].values()))
            for job, (parent, chosen) in zip(jobs, assignments):
                job.update(parent=parent, nodes=list(chosen))
                for name in chosen:
                    free[(zone, parent)].remove(name)

        # A loan only uses nodes left after every group's primary placement.
        # Retain alternative assignments within each donor to fit later tasks.
        backfill_by_zone = {}
        assignments_by_donor = {}
        for job, binding, allowed, parents, donors in backfill_waiting:
            needed = binding['nodes_required']
            for donor in donors:
                pools = [(parent, free.get((donor, parent), ())) for parent in parents]
                previous = assignments_by_donor.get(donor, {frozenset(): ()})
                extended = fit_training80_run(previous, pools, allowed, needed)
                if not extended:
                    continue
                assignments_by_donor[donor] = extended
                backfill_by_zone.setdefault(donor, []).append(job)
                job['backfill'] = {'donor_zone': donor, 'borrower_zone': binding['zone'],
                                   'deadline_s': binding['deadline_s'],
                                   'return_condition': 'actual_terminal_and_post_clean'}
                selected.append(job)
                break
        for donor, jobs in backfill_by_zone.items():
            assignments = next(iter(assignments_by_donor[donor].values()))
            for job, (parent, chosen) in zip(jobs, assignments):
                job.update(parent=parent, nodes=list(chosen))
        if backfill_waiting:
            loans = [job for jobs in backfill_by_zone.values() for job in jobs]
            single_loans = refit_single_node_loans(backfill_waiting, free, loans, placements)
            selected.extend(single_loans)
            selected.extend(refit_provisional_loans(backfill_waiting, free, loans + single_loans))
        result = dict(snapshot_SHA256=digest(fleet), observed_epoch=fleet['observed_epoch'], jobs=selected,
                      selected_cards=sum(8*len(j['nodes']) for j in selected),
                      selected_cards_scope='exclusive reserved node capacity, not actual requested/submitted/busy cards',
                      actual_submitted_cards=0,
                      busy_percent=fleet.get('measured_busy_percent'), busy_is_allocation=False,
                      ready=len(ready), proposed_is_executable=False)
        if placements is not None:
            planned_nodes = {name for job in selected for name in job['nodes']}
            planned_runs = {job['run'] for job in selected}
            coverage = {}
            for name, (zone, parent, slurm_clear) in sorted(placements.items()):
                node = fleet['nodes'][name]
                if not slurm_clear or node['active_steps']:
                    state = 'slurm_step_present_or_unverified'
                elif name in queue_held_nodes:
                    state = 'queue_held'
                elif node.get('device_probe_observed') is not True or node.get('membership_valid') is not True:
                    state = 'admission_unverified'
                elif node.get('KFD_clean') is not True or node.get('PID_clean') is not True:
                    state = 'device_cleanliness_unconfirmed'
                elif name in planned_nodes:
                    state = 'planned'
                else:
                    state = 'available_without_placement'
                coverage.setdefault(zone, {}).setdefault(parent, {}).setdefault(state, []).append(name)
            result['node_coverage'] = coverage
            result['unplaced_ready_runs'] = [dict(run=row['id'], zone=binding['zone'],
                                                  nodes_required=binding['nodes_required'])
                                             for row, binding in ready if row['id'] not in planned_runs]
        if expired:
            result['expired_unclaimed_reservations'] = expired
        result['plan_SHA256'] = digest(result)
        return result

    def reserve(self, plan, fleet, *, claimable=False):
        tickets = {}
        self.db.execute('BEGIN IMMEDIATE')
        try:
            actual = self._plan(fleet)
            require(actual == plan, 'plan changed, replan before dispatch')
            for ident, seq in actual.get('expired_unclaimed_reservations', {}).items():
                self.db.execute('UPDATE runs SET state="READY",nodes="[]" WHERE id=?', (ident,))
                self.event(ident, 'expired_unclaimed', {'reservation_seq': seq,
                                                       'snapshot_SHA256': plan['snapshot_SHA256']})
            for job in plan['jobs']:
                require(claimable or not self.reservation(job['run']).get('dispatch_ticket'),
                        'previously claimable run must use a new dispatch ticket')
                self.db.execute('UPDATE runs SET state="RESERVED",nodes=? WHERE id=?',
                                (canonical(job['nodes']).decode(),job['run']))
                evidence = {'snapshot_SHA256': plan['snapshot_SHA256'], 'nodes': job['nodes']}
                if 'parent' in job:
                    evidence['parent'] = job['parent']
                if 'backfill' in job:
                    evidence['backfill'] = job['backfill']
                if claimable:
                    ticket = uuid4().hex
                    tickets[job['run']] = ticket
                    receipt = fleet['local_receipt']
                    evidence.update(dispatch_ticket=ticket,
                                    received_monotonic=receipt['received_monotonic'],
                                    claim_by_monotonic=(receipt['received_monotonic'] + SNAPSHOT_MAX_AGE_S -
                                                        receipt['roundtrip_s']),
                                    boot_epoch_anchor=receipt['boot_epoch_anchor'])
                self.event(job['run'], 'reserved', evidence)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        return tickets

    def claim_dispatch(self, ident, ticket, handle):
        """Persist the original identity before the launcher may contact Slurm."""
        require(all(isinstance(handle.get(key), str) and handle[key].strip()
                    for key in ('request_id', 'intent')), 'original request and intent required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT state,nodes FROM runs WHERE id=?', (ident,)).fetchone()
            reservation = self.reservation(ident)
            require(row is not None and row['state'] == 'RESERVED' and ticket and
                    reservation.get('dispatch_ticket') == ticket, 'current claimable reservation ticket required')
            now = time.monotonic()
            require(dispatch_window_open(reservation, now, time.time()-now),
                    'dispatch admission expired; release unclaimed reservation and take a fresh snapshot')
            parent = reservation.get('parent')
            require(parent is None or handle.get('parent') == parent, 'dispatch parent differs from reservation')
            require(handle.get('nodes') == json.loads(row['nodes']), 'dispatch nodes differ from reservation')
            self.db.execute('UPDATE runs SET state="DISPATCHING",handle=? WHERE id=?',
                            (canonical(handle).decode(), ident))
            self.event(ident, 'dispatch_claimed', {'dispatch_ticket': ticket, 'handle': handle})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def release_unclaimed(self, ident, ticket):
        """Return a claimable reservation while no launcher has permission to execute it."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT state,handle FROM runs WHERE id=?', (ident,)).fetchone()
            reservation = self.reservation(ident)
            require(row is not None and row['state'] == 'RESERVED' and row['handle'] is None and ticket and
                    reservation.get('dispatch_ticket') == ticket, 'current unclaimed reservation required')
            self.db.execute('UPDATE runs SET state="READY",nodes="[]" WHERE id=?', (ident,))
            self.event(ident, 'released_unclaimed', {'dispatch_ticket': ticket})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def retire_ready(self, ident, reason, evidence_SHA256):
        """Retire a proven prelaunch source defect; never release reserved/unknown work."""
        require(isinstance(reason,str) and reason.strip(), 'prelaunch reason required')
        require(len(evidence_SHA256)==64 and all(c in '0123456789abcdef' for c in evidence_SHA256), 'exact prelaunch evidence SHA required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r=self.db.execute('SELECT state,nodes,handle FROM runs WHERE id=?',(ident,)).fetchone()
            require(r is not None and r['state']=='READY' and json.loads(r['nodes'])==[] and r['handle'] is None,
                    'only unreserved READY may retire; reconcile original active handle')
            result=dict(retired_before_reservation=True,reason=reason,evidence_SHA256=evidence_SHA256,
                        GPU_submission=False,measurement_terminal=False)
            self.db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",result=? WHERE id=?',(canonical(result).decode(),ident))
            self.event(ident,'retired_prelaunch',result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def retire_unattempted_direct(self, ident, handle, receipt_SHA256):
        """Record a local pre-transport rejection from the owning dispatcher."""
        require(isinstance(receipt_SHA256, str) and len(receipt_SHA256) == 64 and
                all(char in '0123456789abcdef' for char in receipt_SHA256), 'exact transport receipt SHA required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT stage,state,handle,nodes FROM runs WHERE id=?', (ident,)).fetchone()
            require(row is not None and row['stage'] == DIRECT_STAGE and row['state'] == 'DISPATCHING'
                    and json.loads(row['handle']) == handle and json.loads(row['nodes']) == handle['nodes']
                    and self.reservation(ident).get('dispatch_ticket') == handle.get('handoff_SHA256')
                    and bool(handle.get('handoff_SHA256')), 'original direct claim must still be before transport')
            result = dict(request_id=handle['request_id'], intent=handle['intent'],
                          receipt_SHA256=receipt_SHA256, transport_attempted=False,
                          measurement_terminal=False, replay_allowed=False)
            self.db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",nodes="[]",result=? WHERE id=?',
                            (canonical(result).decode(), ident))
            self.event(ident, 'direct_retired_unattempted', result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def retire_remote_prelaunch_direct(self, ident, proof):
        """Release a claim after a verified rejection before its first srun."""
        required = {'request_id', 'intent', 'failure_phase', 'launch_attempted',
                    'matching_accounting_rows', 'post_clean', 'node', 'evidence_SHA256'}
        phase = proof.get('failure_phase')
        detached = phase == 'DETACHED_PRE_SLURM_IMPORT'
        operator_admission = phase == 'OPERATOR_PRE_SLURM_ADMISSION'
        scopes = {'PRE_SLURM_IMPORT': ('validation', 0), 'PRE_SLURM_INTENT': ('training', 8),
                  'DETACHED_PRE_SLURM_IMPORT': ('training', 1),
                  'OPERATOR_PRE_SLURM_ADMISSION': ('operator', 1)}
        if detached:
            required |= {'controller', 'controller_stderr_SHA256'}
        else:
            required.add('transport_returncode')
        if operator_admission:
            required |= {'remote_intent_absent', 'admission_SHA256'}
        require(set(proof) == required and phase in scopes
                and proof['launch_attempted'] is False and type(proof['matching_accounting_rows']) is int
                and proof['matching_accounting_rows'] == 0 and proof['post_clean'] is True,
                'verified remote pre-srun rejection and fresh cleanup required')
        if detached:
            require(isinstance(proof['controller_stderr_SHA256'], str)
                    and re.fullmatch(r'[0-9a-f]{64}', proof['controller_stderr_SHA256'])
                    and isinstance(proof['controller'], dict) and proof['controller'].get('request') == proof['request_id']
                    and type(proof['controller'].get('controller_pid')) is int
                    and type(proof['controller'].get('controller_startticks')) is int,
                    'exact detached controller and its terminal import error are required')
        else:
            require(type(proof['transport_returncode']) is int and proof['transport_returncode'] > 0,
                    'failed foreground transport return code required')
        if operator_admission:
            require(proof['remote_intent_absent'] is True and isinstance(proof['admission_SHA256'], str)
                    and re.fullmatch(r'[0-9a-f]{64}', proof['admission_SHA256']),
                    'operator admission proof must precede its observably absent remote intent')
        evidence = proof['evidence_SHA256']
        require(isinstance(evidence, str) and re.fullmatch(r'[0-9a-f]{64}', evidence),
                'exact pre-srun reconciliation evidence SHA required')
        active_state = 'SUBMITTED' if detached else 'UNKNOWN'
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT stage,state,handle,nodes,binding FROM runs WHERE id=?',(ident,)).fetchone()
            require(row is not None and row['stage'] == DIRECT_STAGE and row['state'] == active_state,
                    'original direct claim state differs')
            handle = json.loads(row['handle'])
            reservation = self.reservation(ident)
            node = proof['node']
            kind, cards = scopes[proof['failure_phase']]
            require(handle['request_id'] == proof['request_id'] and handle['intent'] == proof['intent']
                    and handle['nodes'] == json.loads(row['nodes']) == [node]
                    and handle.get('direct_kind') == kind
                    and handle.get('placement_by_parent') == {reservation.get('parent'):{node:cards}}
                    and bool(handle.get('handoff_SHA256'))
                    and reservation.get('dispatch_ticket') == handle['handoff_SHA256'],
                    'original reservation or dispatch identity differs')
            if detached:
                require(handle.get('controller') == proof['controller'], 'different original detached controller')
            if operator_admission:
                require(json.loads(row['binding']).get('snapshot_payload_SHA256') == proof['admission_SHA256'],
                        'different original operator admission')
            result = dict(proof, transport_attempted=True, measurement_terminal=False, replay_allowed=False)
            self.db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",nodes="[]",result=? WHERE id=?',
                            (canonical(result).decode(),ident))
            self.event(ident,'direct_remote_presrun_rejected',result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def submission(self, ident, handle, unknown=False):
        require(bool(handle.get('request_id')) and bool(handle.get('intent')), 'original request and intent required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r = self.db.execute('SELECT state,handle FROM runs WHERE id=?', (ident,)).fetchone()
            reservation = self.reservation(ident)
            if reservation.get('dispatch_ticket'):
                require(r is not None and r['state'] == 'DISPATCHING', 'claim dispatch before contacting Slurm')
                claimed = json.loads(r['handle'])
                if 'direct_kind' in claimed:
                    require(all(handle.get(key, claimed[key]) == claimed[key]
                                for key in ('direct_kind', 'placement_by_parent', 'handoff_SHA256')),
                            'direct submission differs from its handoff')
                require(all(handle[k] == claimed[k] for k in ('request_id', 'intent')), 'different execution identity')
                require(handle.get('nodes', claimed['nodes']) == claimed['nodes'], 'submission nodes differ from claim')
                handle = {**claimed, **handle}
            else:
                require(r is not None and r['state'] == 'RESERVED', 'not exclusively reserved; no replay')
            parent = reservation.get('parent')
            require(parent is None or handle.get('parent') == parent, 'submission parent differs from reservation')
            self.db.execute('UPDATE runs SET state=?,handle=? WHERE id=?',
                            ('UNKNOWN' if unknown else 'SUBMITTED', canonical(handle).decode(), ident))
            self.event(ident, 'submission_unknown' if unknown else 'submitted', handle)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, ident, result):
        """Only actual terminal+post proof releases nodes. A failure remains a failure."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r = self.db.execute('SELECT * FROM runs WHERE id=?', (ident,)).fetchone()
            require(r is not None and r['state'] in ('DISPATCHING','SUBMITTED','UNKNOWN'),
                    'original run has no recorded dispatch identity')
            original = json.loads(r['handle'])
            require(result['request_id'] == original['request_id'] and result['intent'] == original['intent'], 'different execution identity')
            parent = self.reservation(ident).get('parent')
            require(parent is None or result.get('parent') == parent, 'terminal parent differs from reservation')
            require(result['terminal'] is True and result['post_clean'] is True, 'resource release requires actual terminal and post')
            require(bool(result['evidence_SHA256']) and result['planned'] > 0, 'original evidence and planned denominator required')
            counts = [result[k] for k in ('completed','failed','rejected','canceled','missing')]
            require(all(type(v) is int and v >= 0 for v in counts) and sum(counts) == result['planned'], 'denominator lost')
            if r['stage'] == DIRECT_STAGE:
                require(result['qualification'] == original['direct_kind']
                        and result.get('placement_by_parent') == original['placement_by_parent']
                        and not result.get('promote'), 'direct terminal scope differs from claimed placement or kind')
                evidence = result['evidence_SHA256']
                require(isinstance(evidence, str) and len(evidence) == 64
                        and all(char in '0123456789abcdef' for char in evidence),
                        'direct terminal requires a pinned post-clean result')
            else:
                require(result['qualification'] in STAGES, 'explicit qualification scope required')
            require(result['keeper_passed'] in (True, False, None), 'keeper outcome absent')
            if result.get('promote'):
                require(result['passed'] is True and result['keeper_passed'] is True
                        and result['missing'] == 0 and result['failed'] == 0, 'failure/unknown cannot promote')
                require(result['qualification'] == r['stage'], 'component result cannot promote model/service')
            self.db.execute('UPDATE runs SET state="TERMINAL",result=? WHERE id=?', (canonical(result).decode(),ident))
            self.event(ident, 'terminal', result)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def summary(self):
        return dict(candidates=self.db.execute('SELECT count(*) FROM candidates').fetchone()[0],
                    runs={r[0]:r[1] for r in self.db.execute('SELECT state,count(*) FROM runs GROUP BY state')},
                    launches_performed_by_this_tool=0)

def select_next_rung(rows, keep_fraction=1/3, min_pairs=3):
    """Within one frozen comparison, keep measured winners; never fill from failures.

    Deterministic paired bootstrap of warm per-request ratios (candidate/base).
    Source/CPU/component cohorts cannot be mixed with model/service cohorts.
    No decision on formal quality NI is inferred from this screening rank.
    """
    import random
    import statistics
    require(0 < keep_fraction <= 1/3, 'invalid retention: maximum one third')
    require(type(min_pairs) is int and min_pairs in RUNG_PAIR_COUNTS,
            'registered pair budget must be 3, 9 or 27')
    for r in rows:
        require({'model','hardware','precision','scope','workload_SHA256','weights_SHA256','metric'} <= set(r['comparison']), 'incomplete comparison identity')
        require(r['comparison']['scope'] in ('component','model','service'), 'build cannot rank as performance')
    keys={digest(r['comparison']) for r in rows}
    require(len(keys)<=1, 'different hardware/workload/precision/scope comparisons')
    candidates = [r['candidate'] for r in rows]
    require(all(isinstance(candidate, str) and candidate for candidate in candidates)
            and len(set(candidates)) == len(candidates), 'unique candidate IDs required')
    accepted, rejected=[],[]
    for r in rows:
        bad=[]
        counts=[r[k]for k in ('completed','failed','rejected','canceled','missing')]
        require(type(r['planned'])is int and r['planned']>0 and all(type(v)is int and v>=0 for v in counts)and sum(counts)==r['planned'], 'ranking denominator lost')
        if r['completed']<r['baseline_completed'] or r['rejected']>r['baseline_rejected'] or r['canceled']>r['baseline_canceled']:bad.append('completion/reject/cancel tradeoff requires explicit service objective, no automatic speedup promotion')
        if r['quality_correct']<r['baseline_quality_correct']:bad.append('quality loss requires prespecified statistical review')
        if not r['numeric_passed'] or not r['post_clean'] or r['keeper_passed'] is not True:bad.append('numeric/lifecycle/keeper')
        if r['missing'] or r['failed']:bad.append('incomplete/failed denominator')
        if not r['quality_screen_passed']:bad.append('quality screen')
        pairs=r['warm_per_user_ratios']
        if len(pairs) != min_pairs:
            bad.append(f'exactly {min_pairs} warm paired measurements required, observed {len(pairs)}')
        elif any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in pairs):
            bad.append('invalid warm paired measurement')
        if bad:
            rejected.append(dict(candidate=r['candidate'],reasons=bad));continue
        rng=random.Random(digest([r['candidate'],pairs]))
        samples=sorted(statistics.median(rng.choices(pairs,k=len(pairs))) for _ in range(2000))
        lo,hi=samples[49],samples[1949]
        # Throughput ratios >1 mean faster. A confidence interval spanning1 does
        # not become a win merely because this is the only surviving candidate.
        if lo<=1:
            rejected.append(dict(candidate=r['candidate'],reasons=['speedup not resolved above1'],ratio95=[lo,hi]));continue
        accepted.append(dict(candidate=r['candidate'],median_ratio=statistics.median(pairs),ratio95=[lo,hi]))
    accepted.sort(key=lambda r:(-r['ratio95'][0],r['candidate']))
    keep = min(len(rows) // 3, math.floor(len(rows) * keep_fraction))
    return dict(promoted=accepted[:keep],not_promoted=rejected+accepted[keep:],
                formal_NI_proven=False,goal_qualified=False,retention=keep_fraction,
                required_warm_pairs=min_pairs)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--db', default='local-only/tuning/queue.sqlite3')
    sub=a.add_subparsers(dest='command',required=True)
    for name, fields in [('expand',['campaign']),('rank',['results']),('seed',['campaign']),('prepare',['binding']),
                         ('plan',['snapshot']),('reserve',['plan','snapshot']),
                         ('dispatch-state',['run']),('claim-dispatch',['run','ticket','handle']),
                         ('release-unclaimed',['run','ticket']),
                         ('submitted',['run','handle']),('unknown',['run','handle']),('finish',['run','result']),('retire-ready',['run','proof']),('summary',[])]:
        p=sub.add_parser(name)
        for f in fields:p.add_argument(f)
        if name == 'rank':
            p.add_argument('--rung', type=int, choices=range(len(RUNG_PAIR_COUNTS)), default=0)
        if name == 'reserve':
            p.add_argument('--claimable', action='store_true')
    args=a.parse_args();read=lambda p:json.loads(Path(p).read_bytes())
    if args.command=='rank':out=select_next_rung(read(args.results), min_pairs=RUNG_PAIR_COUNTS[args.rung])
    elif args.command=='expand':out=expand(read(args.campaign))
    else:
        q=Queue(args.db)
        if args.command=='seed':q.seed(expand(read(args.campaign)));out=q.summary()
        elif args.command=='prepare':out={'run':q.prepare(read(args.binding),Path(__file__).resolve().parents[1])}
        elif args.command=='plan':out=q.plan(read(args.snapshot))
        elif args.command=='reserve':
            tickets = q.reserve(read(args.plan), read(args.snapshot), claimable=args.claimable)
            out = q.summary()
            if args.claimable:
                out['dispatch_tickets'] = tickets
        elif args.command=='dispatch-state':out=q.dispatch_state(args.run)
        elif args.command=='claim-dispatch':q.claim_dispatch(args.run,args.ticket,read(args.handle));out=q.dispatch_state(args.run)
        elif args.command=='release-unclaimed':q.release_unclaimed(args.run,args.ticket);out=q.dispatch_state(args.run)
        elif args.command in ('submitted','unknown'):q.submission(args.run,read(args.handle),args.command=='unknown');out=q.summary()
        elif args.command=='finish':q.finish(args.run,read(args.result));out=q.summary()
        elif args.command=='retire-ready':
            proof=read(args.proof);q.retire_ready(args.run,proof['reason'],proof['evidence_SHA256']);out=q.summary()
        else:out=q.summary()
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
