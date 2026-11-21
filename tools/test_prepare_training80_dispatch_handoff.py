from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Barrier
import unittest
from unittest.mock import patch

from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import invocation, read_queue_holds
import test_report_training80_backlog as fixture
from tuning_queue import Queue, digest


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixture.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)

    def trial(self, source='def main(publication, request, transport=None): pass\n', index=1):
        job = self.fleet.trial(index)
        launch = job['launch']
        launcher = launch['launcher']
        payload = source.encode()
        (self.fleet.repo / launcher['path']).write_bytes(payload)
        launcher['SHA256'] = sha256(payload).hexdigest()
        manifest_ref = launch['manifest']
        manifest = json.loads((self.fleet.repo / manifest_ref['path']).read_bytes())
        manifest['files_SHA256']['launch.py'] = launcher['SHA256']
        launch['manifest'] = self.fleet.put(manifest_ref['path'], manifest)
        receipt = launch['publication_receipt']
        value = json.loads((self.fleet.repo / receipt['path']).read_bytes())
        value['launcher_SHA256'] = launcher['SHA256']
        value['manifest_SHA256'] = launch['manifest']['SHA256']
        launch['publication_receipt'] = self.fleet.put(receipt['path'], value)
        return job

    def handoff(self, jobs, *, snapshot=None, selected=None, holds=()):
        if snapshot is None:
            snapshot = self.fleet.verified_snapshot()
        return build_handoff(snapshot, dict(minimum_ready_training=5, items=jobs),
            repo=self.fleet.repo, selected=selected, holds=holds,
            now=self.fleet.now, boot_anchor=self.fleet.anchor)

    def direct_claim(self, queue, job, packet, *, intent=None, now=None, snapshot=None):
        return claim_handoff(queue, self.fleet.verified_snapshot() if snapshot is None else snapshot,
            dict(minimum_ready_training=5, items=[job]), packet,
            job['launch']['attempt_receipts'][0] if intent is None else intent,
            repo=self.fleet.repo, now=self.fleet.now if now is None else now, boot_anchor=self.fleet.anchor)

    def test_claim_persists_before_submission_and_cannot_repeat(self):
        job = self.trial()
        [packet] = self.handoff([job])['coordinator_handoffs']
        queue_path = self.fleet.repo / 'claim.sqlite'
        queue = Queue(queue_path)
        self.addCleanup(queue.db.close)
        result = self.direct_claim(queue, job, packet)
        self.assertEqual(result['actual_slurm_submissions'], 0)
        self.assertEqual(result['expires_local_monotonic'], packet['expires_local_monotonic'])
        self.assertEqual(queue.dispatch_state(result['run'])['state'], 'DISPATCHING')
        second = Queue(queue_path)
        self.addCleanup(second.db.close)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.direct_claim(second, job, packet)
        report = self.handoff([job], holds=read_queue_holds(queue_path))
        self.assertEqual(report['outstanding_training_ready'], 0)
        self.assertEqual(report['coordinator_handoffs'], [])

    def test_claim_refuses_changed_packet_wrong_intent_and_stale_receipt(self):
        job = self.trial()
        [packet] = self.handoff([job])['coordinator_handoffs']
        queue = Queue(self.fleet.repo / 'tampered.sqlite')
        self.addCleanup(queue.db.close)
        with self.assertRaisesRegex(ValueError, 'handoff drift'):
            self.direct_claim(queue, job, dict(packet, workload_SHA256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'intent path differs'):
            self.direct_claim(queue, job, packet, intent='local-only/unrelated/intent.json')
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.direct_claim(queue, job, packet, now=packet['expires_local_monotonic'] + 0.5)
        self.fleet.put(job['launch']['attempt_receipts'][0], {'remote_outcome': 'UNKNOWN'})
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.direct_claim(queue, job, packet)
        self.assertEqual(queue.summary()['runs'], {})

    def test_claim_rechecks_time_after_rebuilding_the_packet(self):
        job = self.trial()
        [packet] = self.handoff([job])['coordinator_handoffs']
        queue = Queue(self.fleet.repo / 'expired-at-commit.sqlite')
        self.addCleanup(queue.db.close)
        with patch('prepare_training80_dispatch_handoff.time.monotonic',
                   side_effect=[self.fleet.now, packet['expires_local_monotonic'] + 0.5]):
            with self.assertRaisesRegex(ValueError, 'expired before durable claim'):
                claim_handoff(queue, self.fleet.verified_snapshot(),
                    dict(minimum_ready_training=5, items=[job]), packet, job['launch']['attempt_receipts'][0],
                    repo=self.fleet.repo, boot_anchor=self.fleet.anchor)
        self.assertEqual(queue.summary()['runs'], {})

    def test_concurrent_claims_for_same_node_allow_one(self):
        jobs = [self.trial(index=i) for i in (1, 2)]
        packets = [self.handoff([job])['coordinator_handoffs'][0] for job in jobs]
        queue_path = self.fleet.repo / 'concurrent.sqlite'
        Queue(queue_path).db.close()
        barrier = Barrier(2)
        def attempt(item):
            job, packet = item
            queue = Queue(queue_path)
            try:
                barrier.wait(timeout=5)
                try:
                    self.direct_claim(queue, job, packet)
                except ValueError as exc:
                    self.assertIn('no current device-admitted', str(exc))
                    return False
                return True
            finally:
                queue.db.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(attempt, zip(jobs, packets))), [False, True])
        self.assertEqual(len(read_queue_holds(queue_path)), 1)

    def test_direct_terminal_keeps_original_identity_but_releases_owned_nodes(self):
        job = self.trial()
        [packet] = self.handoff([job])['coordinator_handoffs']
        path = self.fleet.repo / 'terminal.sqlite'
        queue = Queue(path)
        self.addCleanup(queue.db.close)
        claimed = self.direct_claim(queue, job, packet)
        handle = claimed['handle']
        with self.assertRaisesRegex(ValueError, 'differs from its handoff'):
            queue.submission(claimed['run'], dict(handle, placement_by_parent={'1103606': {'x': 8}}))
        queue.submission(claimed['run'], handle, unknown=True)
        result = dict(handle, qualification='training', terminal=True, post_clean=True,
                      evidence_SHA256='a' * 64, planned=1, completed=1, failed=0, rejected=0,
                      canceled=0, missing=0, keeper_passed=True)
        with self.assertRaisesRegex(ValueError, 'requires actual terminal and post'):
            queue.finish(claimed['run'], dict(result, post_clean=False))
        with self.assertRaisesRegex(ValueError, 'direct terminal scope differs'):
            queue.finish(claimed['run'], dict(result, qualification='inference'))
        with self.assertRaisesRegex(ValueError, 'pinned post-clean result'):
            queue.finish(claimed['run'], dict(result, evidence_SHA256='a'))
        queue.finish(claimed['run'], result)
        self.fleet.put(job['launch']['attempt_receipts'][0], {'retained': True})
        other = self.trial(index=2)
        report = self.handoff([job, other], holds=read_queue_holds(path))
        self.assertEqual(report['outstanding_training_ready'], 1)
        self.assertEqual([p['candidate'] for p in report['coordinator_handoffs']], [other['id']])
        self.assertEqual(read_queue_holds(path)[0]['nodes'], [])
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.direct_claim(queue, job, packet)

    def test_valid_packet_uses_pins_and_original_expiration(self):
        job = self.trial()
        result = self.handoff([job], selected=job['id'])
        self.assertEqual(result['training_ready_deficit'], 4)
        self.assertEqual(result['training_pipeline']['ready_to_dispatch'], 1)
        self.assertEqual(result['training_pipeline']['ready_waiting_for_resources'], 0)
        [packet] = result['coordinator_handoffs']
        self.assertEqual(packet['placement_by_parent'], {'1087656': {'a01r4n16': 8}})
        self.assertEqual(packet['expires_local_monotonic'], self.fleet.now + 56)
        self.assertEqual(packet['invocation'], dict(kind='coordinator_callback', entry='main',
            arguments=[job['launch']['publication_commit'], job['launch']['request']], injected_keyword='transport'))
        self.assertEqual(packet['handoff_SHA256'], digest({key: value for key, value in packet.items()
                                                         if key != 'handoff_SHA256'}))
        self.assertEqual(result['actual_dispatched_DCU'], 0)

    def test_interface_includes_vggt_callback_and_host_flags(self):
        vggt = self.trial('def main(publication, transport): pass\n')
        self.assertEqual(invocation(self.fleet.repo, vggt)['arguments'],
                         [vggt['launch']['publication_commit']])
        host = self.trial('''def main(publication, claim_sha256, request):
    pass
cli.add_argument('--publication')
cli.add_argument('--claim-sha256')
cli.add_argument('--request-id')
''', index=2)
        args = invocation(self.fleet.repo, host)
        self.assertEqual(args['kind'], 'authenticated_host_python')
        self.assertEqual(args['arguments'][3], host['launch']['claim']['SHA256'])
        self.assertEqual(args['arguments'][5], host['launch']['request'])

    def test_rereading_select_rejects_old_or_previously_attempted(self):
        job = self.trial()
        for snapshot in (self.fleet.verified_snapshot(age=60), self.fleet.membership):
            with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
                self.handoff([job], selected=job['id'], snapshot=snapshot)
        self.fleet.put(job['launch']['attempt_receipts'][0], {'rc': 255})
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.handoff([job], selected=job['id'])

    def test_original_cpu_prerequisite_prevents_gpu_handoff_on_same_node(self):
        job = self.trial()
        cpu = dict(id='cpu-input-audit', case='T01.input-audit', state='blocked',
                   kind='validation', priority=1, cpu_prerequisite=True, placement={'a01r4n16': 0},
                   attempt_receipts=['local-only/native-cpu/intent.json'], next_action='reconcile original')
        self.assertEqual(len(self.handoff([job, cpu])['coordinator_handoffs']), 1)
        self.fleet.put(cpu['attempt_receipts'][0], {'remote_outcome': 'UNKNOWN'})
        result = self.handoff([job, cpu])
        self.assertEqual(result['coordinator_handoffs'], [])
        self.assertEqual(result['original_launch_holds'][cpu['id']],
                         {'state': 'reconcile', 'nodes': ['a01r4n16']})
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.handoff([job, cpu], selected=job['id'])

    def test_zone_reports_actual_holders_without_placing_that_node(self):
        node = 'a01r4n16'
        self.fleet.snapshot['nodes'][node].update(KFD_clean=False, kfd_process_count=1)
        result = self.handoff([self.trial()])
        scope = result['coverage_by_zone_and_parent']['T01']['1087656']
        self.assertEqual(scope['slurm_no_numbered_step'], ['a01r4n14', 'a01r4n16'])
        self.assertEqual(scope['device_clear_unplaced'], ['a01r4n14'])
        self.assertEqual(scope['device_holders_observed'], [node])
        self.assertEqual(result['coordinator_handoffs'], [])

    def test_unsupported_launcher_stays_manual(self):
        job = self.trial('def main(publication, ssh_helper, bundle): pass\n')
        for snapshot in (self.fleet.verified_snapshot(), self.fleet.membership):
            result = self.handoff([job], snapshot=snapshot)
            self.assertEqual(result['manual_launcher_binding_required'], [job['id']])
            self.assertEqual(result['outstanding_training_ready'], 0)
            self.assertEqual(result['training_ready_deficit'], 5)
            self.assertEqual(result['coordinator_handoffs'], [])

    def test_manual_launcher_leaves_its_nodes_available_for_next_candidate(self):
        manual = self.trial('def main(publication, ssh_helper, bundle): pass\n')
        automatic = self.trial(index=2)
        result = self.handoff([manual, automatic])
        self.assertEqual(result['manual_launcher_binding_required'], [manual['id']])
        self.assertEqual([packet['candidate'] for packet in result['coordinator_handoffs']], [automatic['id']])
        self.assertEqual(result['outstanding_training_ready'], 1)

    def test_stale_supporting_evidence_rejects_only_its_candidate(self):
        stale = self.trial()
        current = self.trial(index=2)
        stale['evidence'] = self.fleet.put('coordination/tasks/stale.json', {'state': 'blocked'})
        self.fleet.put(stale['evidence']['path'], {'state': 'updated'})

        result = self.handoff([stale, current])

        self.assertEqual(result['invalid_launch_evidence'], {stale['id']: 'candidate evidence SHA256 differs'})
        self.assertEqual([packet['candidate'] for packet in result['coordinator_handoffs']], [current['id']])
        self.assertEqual(result['outstanding_training_ready'], 1)
        self.assertEqual(result['actual_dispatched_DCU'], 0)

    def test_five_candidates_including_unbound_launcher_leave_one_ready_deficit(self):
        jobs = [self.trial(index=index) for index in range(1, 5)]
        jobs.append(self.trial('def main(publication, bundle): pass\n', index=5))
        result = self.handoff(jobs, snapshot=self.fleet.membership)
        self.assertEqual(result['outstanding_training_ready'], 4)
        self.assertEqual(result['training_ready_deficit'], 1)
        self.assertEqual(result['coordinator_handoffs'], [])

    def test_queue_hold_prevents_selected_handoff(self):
        job = self.trial()
        holds = [dict(run='prior', state='UNKNOWN', nodes=['a01r4n16'])]
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.handoff([job], selected=job['id'], holds=holds)

    def test_new_original_marker_blocks_second_selection_on_same_node(self):
        replacement = self.trial(index=1)
        original = self.trial(index=2)
        fresh = self.fleet.verified_snapshot()
        [before] = self.handoff([replacement, original], snapshot=fresh)['coordinator_handoffs']
        self.assertEqual(before['candidate'], replacement['id'])
        self.fleet.put(original['launch']['attempt_receipts'][0], {'rc': 255, 'remote_outcome': 'UNKNOWN'})
        self.fleet.put('code/bw1000/fixture-2/config.json', {'invalidates_old_candidate_manifest': True})

        result = self.handoff([replacement, original], snapshot=fresh)
        self.assertEqual(result['original_launch_holds'], {original['id']: {
            'state': 'reconcile', 'nodes': ['a01r4n16']}})
        self.assertEqual(result['coverage_by_zone_and_parent']['T01']['1087656']['original_launch_held'],
                         ['a01r4n16'])
        self.assertEqual(result['device_admitted_unheld_DCU_lower_bound'], 632)
        self.assertIsNone(result['measured_busy_percent'])
        self.assertEqual(result['coordinator_handoffs'], [])
        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.handoff([replacement, original], selected=replacement['id'], snapshot=fresh)

    def test_missing_original_evidence_keeps_both_participating_nodes_held(self):
        original = self.trial(index=3)
        original['state'] = 'submitted'
        original['placement']['f13r1n10'] = 8
        original['evidence'] = self.fleet.put('local-only/prior-original.json', {'state': 'unknown'})
        (self.fleet.repo / original['evidence']['path']).unlink()
        replacement = self.fleet.trial(1, node='f13r1n10')
        independent = self.fleet.trial(2, node='f13r1n11')

        result = self.handoff([replacement, independent, original])
        self.assertEqual(result['original_launch_holds'], {original['id']: {
            'state': 'submitted', 'nodes': ['a01r4n16', 'f13r1n10']}})
        self.assertIn(original['id'], result['invalid_launch_evidence'])
        [packet] = result['coordinator_handoffs']
        self.assertEqual(packet['candidate'], independent['id'])
        self.assertEqual(packet['original_launch_holds_SHA256'], digest(result['original_launch_holds']))

    def test_unknown_attempt_marker_scope_keeps_candidate_placement_held(self):
        replacement = self.trial(index=1)
        original = self.trial(index=2)
        original['launch']['attempt_receipts'] = ['local-only/../coordination/original.json']
        result = self.handoff([replacement, original])
        self.assertEqual(result['original_launch_holds'][original['id']]['state'], 'unverified_attempt_markers')
        self.assertIn(original['id'], result['invalid_launch_evidence'])
        self.assertEqual(result['coordinator_handoffs'], [])

    def test_priority_device_window_reports_only_verified_lower_bound(self):
        clear_nodes = {'a01r4n16', 'f13r1n11'}
        self.fleet.snapshot['device_probe_scope'] = 'PRIORITY_ONLY'
        for node, probe in self.fleet.snapshot['nodes'].items():
            if node not in clear_nodes:
                probe.update(device_probe_observed=False, KFD_clean=False, PID_clean=False,
                             probe_error='DEFERRED_BY_PRIORITY_ONLY')

        result = self.handoff([], snapshot=self.fleet.verified_snapshot())
        self.assertEqual(result['allocation_DCU'], 640)
        self.assertEqual(result['minimum_numbered_step_assigned_DCU'], 0)
        self.assertEqual(result['maximum_numbered_step_assigned_DCU'], 0)
        self.assertEqual(result['device_probe_scope'], 'PRIORITY_ONLY')
        self.assertEqual(result['device_admitted_unheld_DCU_lower_bound'], 16)
        self.assertIsNone(result['measured_busy_percent'])
        stale = self.handoff([], snapshot=self.fleet.verified_snapshot(age=60))
        self.assertIsNone(stale['device_probe_scope'])
        self.assertIsNone(stale['device_admitted_unheld_DCU_lower_bound'])


if __name__ == '__main__':
    unittest.main()
