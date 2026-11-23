from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from report_training80_backlog import TRAINING80_LEDGER, plan, read_queue_holds
from training_dispatch_allocation import (REPLACEMENT_KIND, REPLACEMENT_LEDGER, SURVIVOR_KIND, SURVIVOR_LEDGER,
                                          SURVIVOR_PARENT, allocation)
from tuning_queue import DIRECT_STAGE, digest


class BacklogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.now = 1000.0
        self.anchor = 3000.0
        raw = TRAINING80_LEDGER.read_bytes()
        ledger = json.loads(raw)
        self.membership = dict(ledger_SHA256=sha256(raw).hexdigest(), observed_epoch=1234, distinct_nodes=80,
                               allocated_DCU=640, minimum_numbered_step_assigned_DCU=0,
                               maximum_numbered_step_assigned_DCU=0, parents={})
        probes = {}
        for parent in ('1087656', '1103606'):
            coverage = {}
            for zone, group in ledger['groups'].items():
                coverage[zone] = {}
                for node in group['allocations'][parent]:
                    coverage[zone][node] = dict(state='NO_NUMBERED_STEP', numbered_step_assigned_DCU_bounds=[0, 0])
                    probes[node] = dict(parent=parent, zone=zone, PCIs=[str(i) for i in range(8)],
                                        device_probe_observed=True, membership_valid=True, PID_clean=True,
                                        KFD_clean=True, active_steps=[], occupied_ports=[], ports_checked=[29907])
            self.membership['parents'][parent] = dict(state='RUNNING', ledger_exact=True, nodes=40, allocated_DCU=320,
                    owned_by_observer=True, step_locations_exact=True, coverage_by_group_and_node=coverage,
                    active_steps={})
        self.snapshot = dict(snapshot_kind='training80-dispatch-v1', complete=True, total_cards=640,
                             request_id='f' * 32, membership=self.membership, nodes=probes)

    def verified_snapshot(self, age=3):
        receipt = dict(received_monotonic=self.now-age, roundtrip_s=1.0, boot_epoch_anchor=self.anchor,
                       request_id=self.snapshot['request_id'], payload_SHA256=digest(self.snapshot),
                       raw_response_SHA256='a' * 64)
        return dict(self.snapshot, local_receipt=receipt)

    def put(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return dict(path=name, SHA256=sha256(raw).hexdigest())

    def trial(self, index=1, kind='training', node='a01r4n16', parent=None):
        request = format(index, '032x')
        work = format(index, '064x')
        commit = format(index, '040x')
        base = f'code/bw1000/fixture-{index}'
        prefix = f'local-only/trial-{index}'
        placement = {node: 8 if kind == 'training' else 1}
        minimum = 1 if kind == 'training' else None
        claim = self.put(f'{base}/CLAIM.json', dict(request=request, GPU_GO=True, parent=parent))
        launcher_path = self.repo / base / 'launch.py'
        launcher_bytes = b'def main(publication, request, transport=None): pass\n'
        launcher_path.write_bytes(launcher_bytes)
        launcher = dict(path=f'{base}/launch.py', SHA256=sha256(launcher_bytes).hexdigest())
        dependency = self.put(f'{base}/config.json', dict(context=4096))
        manifest = self.put(f'{base}/SOURCE_FILES.json', dict(files_SHA256={
            Path(ref['path']).name: ref['SHA256'] for ref in (claim, launcher, dependency)
        }))
        publication = self.put(f'{prefix}/publication.json', dict(
            request=request, state='PUBLISHED_MATCHED', publication_commit=commit, claim_SHA256=claim['SHA256'],
            manifest_SHA256=manifest['SHA256'], launcher_SHA256=launcher['SHA256']))
        prerequisites = self.put(f'{prefix}/requirements.json', dict(
            request=request, state='TARGET_PREREQUISITES_VERIFIED', workload_SHA256=work, min_updates=minimum,
            placement=placement, required_ports=[29907]))
        original = self.put(f'{prefix}/original.json', dict(request=request, state='REMOTE_UNSTARTED_VERIFIED',
            original_namespace_absent=True, accounting_label_matches=0, prior_invocation=False))
        return dict(id=f'trial-{index}', case='T01.sft', state='ready', kind=kind, priority=index,
                    next_action='fixture', placement=placement, min_updates=minimum,
                    launch=dict(request=request, publication_commit=commit, workload_SHA256=work,
                                claim=claim, manifest=manifest, launcher=launcher, authorization_key='GPU_GO',
                                publication_receipt=publication, prerequisite_receipt=prerequisites,
                                original_request_receipt=original, required_ports=[29907],
                                attempt_receipts=[f'{prefix}/intent.json']))

    def run_plan(self, *jobs, snapshot=None, anchor=None, holds=()):
        return plan(self.verified_snapshot() if snapshot is None else snapshot,
                    dict(minimum_ready_training=5, items=jobs), repo=self.repo,
                    holds=holds, now=self.now, boot_anchor=self.anchor if anchor is None else anchor)

    def test_membership_alone_or_expired_never_admits(self):
        for receipt in (self.membership, self.verified_snapshot(age=60)):
            result = self.run_plan(self.trial(), snapshot=receipt)
            self.assertEqual(result['outstanding_training_ready'], 1)
            self.assertEqual(result['training_ready_deficit'], 4)
            self.assertEqual(result['proposed_DCU'], 0)
            self.assertEqual(result['admission_clear_nodes'], 0)
            self.assertIsNone(result['measured_busy_percent'])

    def surviving_snapshot(self):
        proof = deepcopy(self.snapshot)
        proof['snapshot_kind'] = SURVIVOR_KIND
        proof['total_cards'] = 320
        proof['membership'].update(distinct_nodes=40, allocated_DCU=320,
                                   ledger_SHA256=sha256(SURVIVOR_LEDGER.read_bytes()).hexdigest())
        proof['membership']['parents'] = {SURVIVOR_PARENT: proof['membership']['parents'][SURVIVOR_PARENT]}
        proof['nodes'] = {name: row for name, row in proof['nodes'].items() if row['parent'] == SURVIVOR_PARENT}
        return proof

    def seal_surviving_snapshot(self, proof, age=3):
        return dict(proof, local_receipt=dict(received_monotonic=self.now-age, roundtrip_s=1.0,
            boot_epoch_anchor=self.anchor, request_id=proof['request_id'], payload_SHA256=digest(proof),
            raw_response_SHA256='a'*64))

    def test_surviving_allocation_keeps_historical_rows_and_proposes_only_live_nodes(self):
        live = self.trial(1, node='f13r1n10')
        failed = self.trial(2, node='a01r4n16')
        history = self.trial(3, node='a01r4n14')
        history.pop('launch')
        history['state'] = 'reconcile'
        proof = self.seal_surviving_snapshot(self.surviving_snapshot())
        result = self.run_plan(live, failed, history, snapshot=proof)
        self.assertEqual((result['fleet_nodes'], result['allocated_DCU']), (40, 320))
        self.assertEqual(result['proposed_for_root'], [live['id']])
        self.assertEqual(result['outstanding_training_ready'], 1)
        self.assertEqual(result['training_pipeline']['outside_live_allocation'], 1)
        self.assertEqual(result['original_launch_holds'][history['id']]['nodes'], ['a01r4n14'])
        self.assertIsNone(result['measured_busy_percent'])

    def test_surviving_snapshot_requires_freshness_and_exact_live_step_coverage(self):
        live = self.trial(1, node='f13r1n10')
        proof = self.surviving_snapshot()
        result = self.run_plan(live, snapshot=self.seal_surviving_snapshot(proof, age=60))
        self.assertEqual(result['proposed_DCU'], 0)
        self.assertFalse(result['fresh_device_receipt'])
        proof['membership']['parents'][SURVIVOR_PARENT]['active_steps']['1103606.4'] = {
            'member_nodes': ['f13r1n10']}
        result = self.run_plan(live, snapshot=self.seal_surviving_snapshot(proof))
        self.assertEqual(result['proposed_DCU'], 0)
        proof['membership']['parents'][SURVIVOR_PARENT]['active_steps']['1103606.4'] = {
            'member_nodes': ['a01r4n16']}
        with self.assertRaises(ValueError):
            self.run_plan(live, snapshot=self.seal_surviving_snapshot(proof))

    def test_surviving_receipt_cannot_claim_full_capacity(self):
        proof = self.surviving_snapshot()
        proof['total_cards'] = 640
        with self.assertRaises(ValueError):
            self.run_plan(snapshot=self.seal_surviving_snapshot(proof))

    def test_replacement_requires_new_parent_on_a_reused_hostname(self):
        raw = REPLACEMENT_LEDGER.read_bytes()
        kind, groups, owners = allocation(raw)
        proof = deepcopy(self.snapshot)
        proof.update(snapshot_kind=kind, nodes={})
        proof['membership'].update(ledger_SHA256=sha256(raw).hexdigest(), parents={})
        for parent, zones in groups.items():
            coverage = {zone: {node: dict(state='NO_NUMBERED_STEP', numbered_step_assigned_DCU_bounds=[0, 0])
                               for node in nodes} for zone, nodes in zones.items()}
            proof['membership']['parents'][parent] = {
                **self.membership['parents'][SURVIVOR_PARENT], 'coverage_by_group_and_node': coverage,
            }
        template = self.snapshot['nodes']['f13r1n10']
        for node, (parent, zone) in owners.items():
            proof['nodes'][node] = {**template, 'parent': parent, 'zone': zone}
        reused = 'f12r3n04'
        jobs = [self.trial(1, node=reused, parent='1087656'), self.trial(2, node=reused, parent='1194900')]
        result = self.run_plan(*jobs, snapshot=self.seal_surviving_snapshot(proof))
        self.assertEqual(kind, REPLACEMENT_KIND)
        self.assertEqual(result['allocated_DCU'], 640)
        self.assertEqual(result['proposed_for_root'], [jobs[1]['id']])
        self.assertEqual(result['training_pipeline']['outside_live_allocation'], 1)

    def test_fresh_priority_distinct_nodes_and_training_count(self):
        result = self.run_plan(self.trial(1, 'operator'), self.trial(2), self.trial(3, 'validation', 'a01r4n14'))
        self.assertEqual(result['outstanding_training_ready'], 1)
        self.assertEqual(result['proposed_for_root'], ['trial-2', 'trial-3'])
        self.assertEqual(result['proposed_DCU'], 9)
        self.assertEqual(result['actual_submitted_DCU'], 0)

    def test_attempt_receipt_always_requires_reconciliation(self):
        job = self.trial()
        self.put(job['launch']['attempt_receipts'][0], dict(remote_outcome='UNKNOWN'))
        result = self.run_plan(job)
        self.assertEqual(result['outstanding_training_ready'], 0)
        self.assertEqual(result['rows'][0]['dispatch_state'], 'reconcile')

    def test_durable_direct_training_hold_does_not_depend_on_marker_permissions(self):
        spec = self.trial()
        launch = spec.pop('launch')
        spec.update(state='blocked', direct_request=launch['request'],
                    attempt_receipts=launch['attempt_receipts'])
        hold = dict(run='e' * 64, state='UNKNOWN', direct_request=launch['request'],
                    nodes=list(spec['placement']))
        with patch('report_training80_backlog.has_original_intent', side_effect=PermissionError()):
            held = self.run_plan(spec, holds=[hold])
            unmatched = self.run_plan(spec, holds=[dict(hold, nodes=['f13r1n10'])])
        self.assertEqual(held['rows'][0]['dispatch_state'], 'reconcile')
        self.assertEqual(held['training_pipeline']['needs_reconciliation'], 1)
        self.assertEqual(held['training_pipeline']['invalid_evidence'], 0)
        self.assertEqual(held['queue_held_nodes'], list(spec['placement']))
        self.assertEqual(unmatched['rows'][0]['dispatch_state'], 'invalid_launch_evidence')

    def test_training_pipeline_keeps_dispatch_reserve_and_recorded_runs_distinct(self):
        ready = self.trial(1)
        waiting = self.trial(2)
        blocked = dict(self.trial(3), state='blocked')
        reconcile = dict(self.trial(4, node='f13r1n10'), state='reconcile')
        invalid = dict(self.trial(5), evidence={'path': 'local-only/missing.json', 'SHA256': '0' * 64})
        submitted = dict(self.trial(6, node='f13r1n11'), state='submitted')
        terminal = dict(self.trial(7), state='terminal')
        validation = self.trial(8, kind='validation', node='a01r4n14')
        result = self.run_plan(ready, waiting, blocked, reconcile, invalid, submitted, terminal, validation)
        self.assertEqual(result['training_pipeline'], dict(ready_to_dispatch=1,
            ready_waiting_for_resources=1, needs_preparation=1, needs_reconciliation=1, invalid_evidence=1,
            recorded_submitted=1, recorded_terminal=1))
        self.assertEqual(result['outstanding_training_ready'], 2)
        self.assertEqual(result['training_ready_deficit'], 3)
        self.assertIn(validation['id'], result['proposed_for_root'])

    def test_kfd_holder_or_occupied_port_prevents_placement(self):
        job = self.trial()
        probe = self.snapshot['nodes']['a01r4n16']
        for changes in (dict(KFD_clean=False, occupied_ports=[]), dict(KFD_clean=True, occupied_ports=[29907])):
            probe.update(changes)
            result = self.run_plan(job)
            self.assertEqual(result['proposed_DCU'], 0)
            self.assertEqual(result['rows'][0]['dispatch_state'], 'awaiting_device_or_port_clearance')

    def test_duplicate_workload_or_tampered_publication_rejected(self):
        first = self.trial()
        second = self.trial(2)
        second['launch']['workload_SHA256'] = first['launch']['workload_SHA256']
        with self.assertRaisesRegex(ValueError, 'duplicate semantic scope'):
            self.run_plan(first, second)
        self.put(first['launch']['publication_receipt']['path'], dict(incorrect=True))
        result = self.run_plan(first)
        self.assertEqual(result['rows'][0]['candidate_state'], 'invalid_launch_evidence')
        self.assertIn('evidence SHA256', result['rows'][0]['launch_evidence_error'])

    def test_frozen_manifest_without_its_original_member_bytes_is_rejected(self):
        job = self.trial()
        target = Path(job['launch']['manifest']['path']).parent / 'config.json'
        self.put(target, dict(context=2048))
        valid = self.trial(2)
        result = self.run_plan(job, valid)
        self.assertEqual(result['rows'][0]['candidate_state'], 'invalid_launch_evidence')
        self.assertIn('source member bytes differ: config.json', result['rows'][0]['launch_evidence_error'])
        self.assertEqual(result['outstanding_training_ready'], 1)
        self.assertEqual(result['proposed_for_root'], [valid['id']])

    def test_receipt_payload_and_boot_anchor_checked(self):
        job = self.trial()
        baseline = self.verified_snapshot()
        changed = deepcopy(baseline)
        changed['nodes']['a01r4n16']['KFD_clean'] = False
        with self.assertRaisesRegex(ValueError, 'collector identity or payload differs'):
            self.run_plan(job, snapshot=changed)
        result = self.run_plan(job, snapshot=baseline, anchor=self.anchor+3)
        self.assertFalse(result['fresh_device_receipt'])
        self.assertEqual(result['proposed_DCU'], 0)

    def test_unknown_queue_reservation_prevents_proposal_without_db_mutation(self):
        db = self.repo / 'test.sqlite3'
        connection = sqlite3.connect(db)
        with connection:
            connection.execute('create table runs (id text, state text, nodes text)')
            connection.executemany('insert into runs values (?, ?, ?)',
                [('original', 'UNKNOWN', '["a01r4n16"]'), ('ended', 'TERMINAL', '["f13r1n10"]')])
        connection.close()
        previous = db.read_bytes()
        result = self.run_plan(self.trial(), holds=read_queue_holds(db))
        self.assertEqual(result['queue_held_nodes'], ['a01r4n16'])
        self.assertEqual(result['rows'][0]['dispatch_state'], 'awaiting_original_queue_release')
        self.assertEqual(db.read_bytes(), previous)

    def test_known_historical_glm_terminal_remains_in_workload_index(self):
        db = self.repo / 'historical.sqlite3'
        request = '93148263aba346dda17b534d419d81c83'
        run = digest(dict(stage=DIRECT_STAGE, request=request))
        workload = 'c' * 64
        terminal = dict(request_id=request, terminal=True, post_clean=True,
                        evidence_SHA256='240dcb8986383f3d786264a96828f008062f977ecc72feba1343a0508c726628')
        connection = sqlite3.connect(db)
        self.addCleanup(connection.close)
        with connection:
            connection.execute('create table runs (id text, state text, nodes text, stage text, handle text, binding text, result text)')
            connection.execute('insert into runs values (?, ?, ?, ?, ?, ?, ?)',
                (run, 'TERMINAL', '["f12r2n13"]', DIRECT_STAGE, json.dumps(dict(request_id=request)),
                 json.dumps(dict(request=request, workload_SHA256=workload)), json.dumps(terminal)))
        self.assertEqual(read_queue_holds(db), [dict(run=run, state='TERMINAL', nodes=[],
            direct_request=request, direct_workload_SHA256=workload)])
        with connection:
            connection.execute('update runs set state = ?', ('UNKNOWN',))
        with self.assertRaisesRegex(ValueError, 'direct handoff request identity missing'):
            read_queue_holds(db)


if __name__ == '__main__':
    unittest.main()
