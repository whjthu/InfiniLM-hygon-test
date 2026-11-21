import json
import unittest

from plan_training80_backfill import active_direct_claims, plan_backfill
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue


class TrainingBackfillTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixtures.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.now = dict(now=self.fleet.now, boot_anchor=self.fleet.anchor)
        self.queue = Queue(self.fleet.repo/'dispatch.sqlite')
        self.addCleanup(self.queue.db.close)
        self.parent = '1103606'
        self.active = 'f13r1n10'
        self.peer = 'f13r1n11'
        self.other = 'a01r4n16'
        self.step = self.parent+'.103'

    def adjust(self, ref, **fields):
        document = json.loads((self.fleet.repo/ref['path']).read_bytes())
        return self.fleet.put(ref['path'], dict(document, **fields))

    def job(self, index, kind, nodes):
        row = self.fleet.trial(index, kind, nodes[0])
        row['placement'] = dict.fromkeys(nodes, 8 if kind == 'training' else 1)
        launch = row['launch']
        launch['prerequisite_receipt'] = self.adjust(launch['prerequisite_receipt'], placement=row['placement'])
        return row

    def register_operator(self, nodes, *, unknown=False, exact_step=True, cooperative=True):
        job = self.job(1, 'operator', nodes)
        launch = job['launch']
        if cooperative:
            launch['claim'] = self.adjust(launch['claim'], backfill_stop=dict(method='cooperative_signal', signal='USR1',
                save_partial_results=True, grace_seconds=10))
            manifest = json.loads((self.fleet.repo/launch['manifest']['path']).read_bytes())['files_SHA256']
            launch['manifest'] = self.adjust(launch['manifest'], files_SHA256=dict(manifest, **{
                'CLAIM.json': launch['claim']['SHA256']}))
            launch['publication_receipt'] = self.adjust(launch['publication_receipt'],
                claim_SHA256=launch['claim']['SHA256'], manifest_SHA256=launch['manifest']['SHA256'])
        board = dict(minimum_ready_training=5, items=[job])
        snap = self.fleet.verified_snapshot()
        report = build_handoff(snap, board, repo=self.fleet.repo, **self.now)
        [packet] = report['coordinator_handoffs']
        claimed = claim_handoff(self.queue, snap, board, packet, launch['attempt_receipts'][0],
                                repo=self.fleet.repo, **self.now)
        bound_step = self.step if exact_step else self.parent+'.104'
        self.queue.submission(claimed['run'], dict(claimed['handle'], gpu_step=bound_step), unknown=unknown)
        parent = self.fleet.membership['parents'][self.parent]
        parent['active_steps'][self.step] = dict(state='RUNNING', member_nodes=sorted(nodes),
            allocated_DCU=len(nodes), membership_qualified=True)
        for node in nodes:
            self.fleet.snapshot['nodes'][node].update(device_probe_observed=False, active_steps=[self.step],
                                                     ports_checked=None, occupied_ports=None)
            zone = self.fleet.snapshot['nodes'][node]['zone']
            parent['coverage_by_group_and_node'][zone][node] = dict(state='PARTIALLY_ASSIGNED',
                numbered_step_assigned_DCU_bounds=[1, 1])
        return job

    def plan(self, *jobs, age=3):
        return plan_backfill(self.fleet.verified_snapshot(age=age), dict(minimum_ready_training=5, items=list(jobs)),
            repo=self.fleet.repo, holds=queue_holds(self.queue.db), submitted=active_direct_claims(self.queue.db), **self.now)

    def test_exact_running_owner_step_and_disjoint_backfill_can_progress_together(self):
        running = self.register_operator([self.active])
        report = self.plan(running, self.job(2, 'training', [self.active]), self.job(3, 'validation', [self.other]))
        [stop] = report['cooperative_stops']
        [step] = stop['original_steps']
        self.assertEqual(step['step'], self.step)
        self.assertEqual(step['request'], running['launch']['request'])
        self.assertEqual(step['nodes'], [self.active])
        self.assertEqual(step['allocated_DCU'], 1)
        self.assertTrue(stop['requires_terminal_and_independent_cleanup'])
        self.assertTrue(stop['requires_new_device_admission'])
        self.assertEqual([row['candidate'] for row in report['clean_node_dispatches']], ['trial-3'])
        self.assertEqual(report['training_ready_deficit'], 4)
        self.assertEqual(self.queue.summary()['runs'], {'SUBMITTED': 1})

    def test_unknown_original_is_never_selected_for_stop_or_same_node_backfill(self):
        running = self.register_operator([self.active], unknown=True)
        report = self.plan(running, self.job(2, 'training', [self.active]), self.job(3, 'validation', [self.other]))
        self.assertEqual(report['cooperative_stops'], [])
        self.assertEqual([row['candidate'] for row in report['clean_node_dispatches']], ['trial-3'])

    def test_different_numbered_step_is_not_the_claimed_owner(self):
        running = self.register_operator([self.active], exact_step=False)
        self.assertEqual(self.plan(running, self.job(2, 'training', [self.active]))['cooperative_stops'], [])

    def test_no_frozen_cooperative_policy_keeps_natural_completion(self):
        running = self.register_operator([self.active], cooperative=False)
        self.assertEqual(self.plan(running, self.job(2, 'training', [self.active]))['cooperative_stops'], [])

    def test_overlapping_numbered_step_preserves_other_active_owner(self):
        running = self.register_operator([self.active])
        parent = self.fleet.membership['parents'][self.parent]
        parent['active_steps'][self.parent+'.104'] = dict(state='RUNNING', member_nodes=[self.active],
            allocated_DCU=0, membership_qualified=True)
        self.assertEqual(self.plan(running, self.job(2, 'training', [self.active]))['cooperative_stops'], [])

    def test_never_stop_a_group_to_reclaim_only_one_of_its_nodes(self):
        running = self.register_operator([self.active, self.peer])
        self.assertEqual(self.plan(running, self.job(2, 'training', [self.active]))['cooperative_stops'], [])
        [stop] = self.plan(running, self.job(3, 'training', [self.active, self.peer]))['cooperative_stops']
        self.assertEqual(stop['original_steps'][0]['nodes'], [self.active, self.peer])

    def test_ready_training_keeps_its_other_clear_node_out_of_low_priority_backfill(self):
        running = self.register_operator([self.active])
        report = self.plan(running, self.job(2, 'training', [self.active, self.peer]),
                           self.job(3, 'validation', [self.peer]), self.job(4, 'validation', [self.other]))
        [stop] = report['cooperative_stops']
        self.assertEqual(stop['candidate'], 'trial-2')
        self.assertEqual([row['candidate'] for row in report['clean_node_dispatches']], ['trial-4'])

    def test_stale_devices_cannot_suggest_stop_or_report_idle_capacity(self):
        running = self.register_operator([self.active])
        report = self.plan(running, self.job(2, 'training', [self.active]), age=60)
        self.assertEqual(report['cooperative_stops'], [])
        self.assertTrue(all(row['clean_DCU_lower_bound'] is None for row in report['device_groups']))


if __name__ == '__main__':
    unittest.main()
