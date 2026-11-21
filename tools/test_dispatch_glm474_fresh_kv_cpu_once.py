import json
import subprocess
from types import SimpleNamespace
import unittest

import dispatch_glm474_fresh_kv_cpu_once as launcher
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import plan, queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue


class GLMOriginalCPUIntentTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo.resolve()
        names = [launcher.MANIFEST['path'], launcher.TASK['path']]
        manifest = json.loads((launcher.REPO / launcher.MANIFEST['path']).read_bytes())
        names.extend(launcher.PACKAGE + '/' + name for name in manifest['files_SHA256'])
        claim = json.loads((launcher.REPO / launcher.CLAIM['path']).read_bytes())
        names.extend(claim['shared_files'])
        for name in names:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((launcher.REPO / name).read_bytes())
        (self.repo / 'local-only').mkdir()
        self.job = self.fixture.trial(1, 'operator', node=launcher.NODE)
        self.board = dict(minimum_ready_training=5, items=[launcher.backlog_row(), self.job])
        self.board_path = self.repo / 'board.json'
        self.write_board()
        self.database = self.repo / 'queue.sqlite3'
        self.queue = Queue(self.database)
        self.addCleanup(self.queue.db.close)
        self.intent = self.repo / launcher.INTENT

    def write_board(self):
        self.board_path.write_text(json.dumps(self.board), encoding='utf8')

    def dispatch(self, callback, root=None, publication=launcher.HOST_PUBLICATION):
        return launcher.dispatch(publication, str(launcher.PROJECT) if root is None else root, callback,
                                 repo=self.repo, backlog=self.board_path, database=self.database)

    def test_durable_intent_blocks_old_gpu_handoff_before_callback(self):
        snapshot = self.fixture.verified_snapshot()
        local_clock = dict(now=self.fixture.now, boot_anchor=self.fixture.anchor)
        [packet] = build_handoff(snapshot, self.board, repo=self.repo, **local_clock)['coordinator_handoffs']
        calls = []

        def transfer(argv, stdout, stderr, timeout):
            calls.append(argv)
            original = json.loads(self.intent.read_bytes())
            self.assertEqual(original['request'], launcher.REQUEST)
            self.assertEqual(original['placement'], launcher.PLACEMENT)
            self.assertEqual(original['remote_outcome'], 'UNKNOWN')
            [hold] = queue_holds(self.queue.db)
            self.assertEqual((hold['state'], hold['nodes'], hold['direct_request']),
                             ('DISPATCHING', [launcher.NODE], launcher.REQUEST))
            self.assertEqual(argv[-2:], ('--claim-sha256', launcher.CLAIM['SHA256']))
            self.assertEqual(timeout, 320)
            with self.assertRaisesRegex(ValueError, 'no current device-admitted'):
                claim_handoff(self.queue, snapshot, self.board, packet, self.job['launch']['attempt_receipts'][0],
                              repo=self.repo, **local_clock)
            stdout.write(b'private fixture result')
            return SimpleNamespace(returncode=0)

        result = self.dispatch(transfer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['transport_returncode'], 0)
        self.assertEqual(result['remote_outcome'], 'UNKNOWN')
        self.assertFalse(result['terminal_verified'])
        self.assertEqual((self.intent.parent / 'stdout.private').read_bytes(), b'private fixture result')

    def test_response_loss_preserves_intent_and_refuses_second_invocation(self):
        calls = []

        def timeout(*args):
            calls.append(args[0])
            raise subprocess.TimeoutExpired('fixture', 1)

        result = self.dispatch(timeout)
        self.assertEqual(result['transport_error'], 'TimeoutExpired')
        self.assertTrue(self.intent.is_file())
        self.assertEqual(json.loads((self.intent.parent / 'transport.json').read_bytes()), result)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.dispatch(timeout)
        self.assertEqual(len(calls), 1)

    def test_existing_sqlite_reservation_prevents_local_claim(self):
        self.queue.db.execute('INSERT INTO candidates VALUES(?,?)', ('other', '{}'))
        self.queue.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
            ('other', 'other', 'component', 'UNKNOWN', '{}', json.dumps([launcher.NODE]), None, None, 1))
        with self.assertRaisesRegex(ValueError, 'already holds'):
            self.dispatch(lambda *args: self.fail('call would overlap an active node'))
        self.assertFalse(self.intent.exists())

    def test_missing_registration_refuses_cross_end_call(self):
        self.board['items'].remove(launcher.backlog_row())
        self.write_board()
        with self.assertRaisesRegex(ValueError, 'register the exact'):
            self.dispatch(lambda *args: self.fail('node guard was not registered'))
        self.assertFalse(self.intent.exists())

    def test_outside_target_or_host_pin_drift_never_creates_intent(self):
        with self.assertRaisesRegex(ValueError, 'canonical target checkout'):
            self.dispatch(lambda *args: self.fail('target is outside project'), root='/tmp/other')
        (self.repo / launcher.HOST['path']).write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'source member bytes differ'):
            self.dispatch(lambda *args: self.fail('edited target entry'))
        self.assertFalse(self.intent.exists())

    def test_unrelated_publication_never_creates_intent(self):
        with self.assertRaisesRegex(ValueError, 'verified HOST publication'):
            self.dispatch(lambda *args: self.fail('wrong published commit'), publication='a'*40)
        self.assertFalse(self.intent.exists())

    def test_sqlite_hold_survives_disjoint_planning_and_releases_only_with_terminal_post(self):
        result = self.dispatch(lambda *args: SimpleNamespace(returncode=0))
        run = result['queue_run']
        self.assertEqual(self.queue.reserved_parent(run), launcher.PARENT)
        completion = dict(request_id=launcher.REQUEST, intent=launcher.INTENT, parent=launcher.PARENT,
                          terminal=True, post_clean=True, evidence_SHA256='a'*64, planned=1, completed=1,
                          failed=0, rejected=0, canceled=0, missing=0, qualification='validation', keeper_passed=None,
                          placement_by_parent={launcher.PARENT: {launcher.NODE: 0}})
        with self.assertRaisesRegex(ValueError, 'actual terminal and post'):
            self.queue.finish(run, dict(completion, post_clean=False))
        self.queue.finish(run, completion)
        snapshot = self.fixture.verified_snapshot()
        report = plan(snapshot, self.board, repo=self.repo, holds=queue_holds(self.queue.db),
                      now=self.fixture.now, boot_anchor=self.fixture.anchor)
        self.assertEqual(report['proposed_for_root'], [self.job['id']])
        self.assertTrue(self.intent.is_file())


if __name__ == '__main__':
    unittest.main()
