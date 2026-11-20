from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from report_training80_backlog import TRAINING80_LEDGER, plan, read_queue_holds
from tuning_queue import digest


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

    def trial(self, index=1, kind='training', node='a01r4n16'):
        request = format(index, '032x')
        work = format(index, '064x')
        commit = format(index, '040x')
        base = f'code/bw1000/fixture-{index}'
        prefix = f'local-only/trial-{index}'
        placement = {node: 8 if kind == 'training' else 1}
        minimum = 1 if kind == 'training' else None
        claim = self.put(f'{base}/CLAIM.json', dict(request=request, GPU_GO=True))
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


if __name__ == '__main__':
    unittest.main()
