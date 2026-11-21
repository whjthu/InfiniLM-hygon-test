import json
import unittest

import prepare_training80_deep_c4_link_backfill as bridge
import test_report_training80_backlog as fixtures


class DeepGuardedBackfillTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.ledger = json.loads(bridge.TRAINING80_LEDGER.read_bytes())
        self.backlog = dict(minimum_ready_training=5, items=[])
        task = json.loads((bridge.REPO / bridge.TASK['path']).read_bytes())
        self.manifest_path = task['source'] + '/SOURCE_FILES.json'
        mf = json.loads((bridge.REPO / self.manifest_path).read_bytes())
        for name in (bridge.TASK['path'], bridge.ORIGINAL_BUILD['path'], self.manifest_path,
                     *(item['path'] for item in mf['files'].values())):
            dest = self.fixture.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((bridge.REPO / name).read_bytes())

    def prepare(self, holds=()):
        return bridge.prepare(self.fixture.repo, self.backlog, self.ledger, holds)

    def test_exact_sources_and_original_terminal_are_ready_for_cpu_preflight_only(self):
        details = self.prepare()['deep_c4_link_backfill']
        self.assertEqual(details['state'], 'WAITING_AUTHENTICATED_CPU_ADMISSION')
        self.assertEqual(details['verified_source_members'], 8)
        self.assertEqual(details['original_steps'], ['1103606.156', '1103606.487'])
        self.assertFalse(details['counts_toward_training_reserve'])
        self.assertFalse(details['gpu_eligible'])
        self.assertEqual(details['actual_DCU_started'], 0)

    def test_existing_overlay_blocks_gpu_selection_when_original_intent_appears(self):
        gpu = self.fixture.trial(1, 'operator', node=bridge.NODE)
        self.backlog['items'].append(gpu)
        overlay = self.prepare()
        before = self.fixture.run_plan(*overlay['items'])
        self.assertEqual(before['proposed_for_root'], [gpu['id']])
        self.fixture.put(bridge.INTENT, {'request': bridge.REQUEST, 'remote_outcome': 'UNKNOWN'})

        after = self.fixture.run_plan(*overlay['items'])
        self.assertEqual(after['training_ready_deficit'], 5)
        self.assertEqual(after['proposed_for_root'], [])
        self.assertEqual(after['original_launch_holds'][bridge.IDENT]['nodes'], [bridge.NODE])
        self.assertIsNone(self.prepare()['deep_c4_link_backfill']['invocation_after_existing_checks'])

    def test_original_and_sqlite_holds_both_block_the_cpu_launch_pointer(self):
        self.backlog['items'].append(dict(id='other-original', state='reconcile', placement={bridge.NODE: 1}))
        result = self.prepare(holds=[dict(run='sql-active', state='UNKNOWN', nodes=[bridge.NODE])])
        details = result['deep_c4_link_backfill']
        self.assertEqual(details['state'], 'WAITING_FOR_QUEUE_HOLD')
        self.assertEqual(details['before_run_queue_holds'], ['sql-active'])
        self.assertEqual(details['before_run_direct_holds'], ['other-original'])
        self.assertIsNone(details['invocation_after_existing_checks'])

    def test_changed_receipt_rejects_the_overlay(self):
        (self.fixture.repo / bridge.ORIGINAL_BUILD['path']).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'candidate evidence SHA256 differs'):
            self.prepare()

    def test_published_registration_is_idempotent_and_still_protects_default_planner(self):
        gpu = self.fixture.trial(1, 'operator', node=bridge.NODE)
        self.backlog['items'].append(gpu)
        self.backlog = self.prepare()
        original = self.backlog['items']
        self.assertEqual(self.prepare()['items'], original)
        self.fixture.put(bridge.INTENT, {'request': bridge.REQUEST, 'remote_outcome': 'UNKNOWN'})
        actual = self.fixture.run_plan(*original)
        self.assertEqual(actual['proposed_for_root'], [])
        self.assertEqual(actual['original_launch_holds'][bridge.IDENT]['nodes'], [bridge.NODE])
        original[-1]['placement'] = {bridge.NODE: 1}
        with self.assertRaisesRegex(ValueError, 'registered Deep CPU backfill scope differs'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
