import json
import unittest

from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from prepare_training80_dispatch_probe_args import probe_arguments
from report_training80_backlog import TRAINING80_LEDGER, read_queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue


class DurableWorkloadTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixtures.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.database = self.fleet.repo / 'direct.sqlite3'
        self.queue = Queue(self.database)
        self.addCleanup(self.queue.db.close)
        self.snapshot = self.fleet.verified_snapshot()
        self.timing = dict(repo=self.fleet.repo, now=self.fleet.now, boot_anchor=self.fleet.anchor)
        self.original = self.fleet.trial(1, node='a01r4n16')
        self.rekeyed = self.fleet.trial(2, node='f13r1n11')
        launch = self.rekeyed['launch']
        launch['workload_SHA256'] = self.original['launch']['workload_SHA256']
        path = launch['prerequisite_receipt']['path']
        prerequisite = json.loads((self.fleet.repo / path).read_bytes())
        prerequisite['workload_SHA256'] = launch['workload_SHA256']
        launch['prerequisite_receipt'] = self.fleet.put(path, prerequisite)
        self.variant = self.fleet.trial(3, node='f13r1n11')

    def backlog(self, *jobs):
        return dict(minimum_ready_training=5, items=jobs)

    def packet(self, job):
        [packet] = build_handoff(self.snapshot, self.backlog(job), **self.timing)['coordinator_handoffs']
        return packet

    def claim(self, job, packet):
        return claim_handoff(self.queue, self.snapshot, self.backlog(job), packet,
                             job['launch']['attempt_receipts'][0], **self.timing)

    def test_unknown_prior_workload_blocks_a_new_request_before_atomic_claim(self):
        stale_packet = self.packet(self.rekeyed)
        original = self.claim(self.original, self.packet(self.original))
        self.queue.submission(original['run'], original['handle'], unknown=True)

        with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
            self.claim(self.rekeyed, stale_packet)
        self.assertEqual(self.queue.summary()['runs'], {'UNKNOWN': 1})

    def test_prior_workload_exclusion_leaves_other_node_available_to_a_distinct_variant(self):
        original = self.claim(self.original, self.packet(self.original))
        holds = read_queue_holds(self.database)
        current = self.backlog(self.rekeyed, self.variant)
        handoff = build_handoff(self.snapshot, current, holds=holds, **self.timing)
        self.assertEqual([row['candidate'] for row in handoff['coordinator_handoffs']], [self.variant['id']])
        self.assertEqual(handoff['outstanding_training_ready'], 1)
        self.assertEqual(handoff['prior_workload_claims'][self.rekeyed['id']][0]['run'], original['run'])
        probe = probe_arguments(self.fleet.repo, current, json.loads(TRAINING80_LEDGER.read_bytes()), holds)
        self.assertEqual(probe['verified_training_ready'], 1)
        self.assertIn(self.rekeyed['id'], probe['prior_workload_claims'])
        self.assertEqual(probe['priority_by_parent'], {'1103606': ['f13r1n11']})

    def test_terminal_prior_workload_blocks_reentry_after_releasing_original_nodes(self):
        original = self.claim(self.original, self.packet(self.original))
        handle = original['handle']
        self.queue.submission(original['run'], handle)
        self.queue.finish(original['run'], dict(handle, qualification='training', terminal=True, post_clean=True,
                                                evidence_SHA256='a' * 64, keeper_passed=True, planned=1, completed=1,
                                                failed=0, rejected=0, canceled=0, missing=0))
        holds = read_queue_holds(self.database)
        handoff = build_handoff(self.snapshot, self.backlog(self.rekeyed), holds=holds, **self.timing)
        self.assertEqual(holds[0]['nodes'], [])
        self.assertEqual(handoff['coordinator_handoffs'], [])
        self.assertEqual(handoff['outstanding_training_ready'], 0)
        self.assertEqual(handoff['prior_workload_claims'][self.rekeyed['id']][0]['state'], 'TERMINAL')


if __name__ == '__main__':
    unittest.main()
