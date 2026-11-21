from contextlib import closing
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dispatch_training80_vggt_once as dispatcher
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import read_queue_holds
import test_prepare_training80_vggt_admission as admission_fixture
from tuning_queue import Queue


class GuardedVggtDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = admission_fixture.VggtAdmissionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        overlay = self.fixture.prepare()
        self.backlog = json.loads((self.repo/overlay['path']).read_bytes())
        self.snapshot = self.fixture.fleet.verified_snapshot()
        [self.packet] = self.fixture.report(self.snapshot, self.backlog)['coordinator_handoffs']
        self.now = self.fixture.fleet.now
        self.anchor = self.fixture.fleet.anchor
        self.database = self.repo/dispatcher.QUEUE_DB.relative_to(dispatcher.REPO)
        Queue(self.database).db.close()
        self.intent = self.repo/dispatcher.INTENT
        self.proof = self.intent.with_name(dispatcher.GUARD_RECEIPT)
        path = self.repo/dispatcher.PACKAGE/'CLAIM.json'
        claim = json.loads(path.read_bytes())
        self.last_blob = next(reversed(claim['sources'].values()))['path']
        self.options = dict(repo=self.repo, database=self.database)

    def published_files(self, args, *, timeout):
        self.assertGreater(timeout, 0)
        if args[-4:] == ['ls-remote', '--exit-code', 'origin', 'refs/heads/bw1000-experiments']:
            return (self.fixture.publication+' refs/heads/bw1000-experiments\n').encode()
        self.assertEqual(args[-3:-1], ['cat-file', 'blob'])
        publication, relative = args[-1].split(':', 1)
        self.assertEqual(publication, self.fixture.publication)
        self.assertIn(relative, self.fixture.members)
        return (self.repo/relative).read_bytes()

    def invoke(self, transport, *, publication_response=None):
        with patch('subprocess.check_output', side_effect=publication_response or self.published_files), \
             patch.object(dispatcher.time, 'monotonic', side_effect=lambda: self.now), \
             patch.object(dispatcher.time, 'time', side_effect=lambda: self.now + self.anchor):
            return dispatcher.dispatch(self.snapshot, self.backlog, self.packet, transport, **self.options)

    def reject_transport(self, *_):
        self.fail('network transport must never be reached in this case')

    def assert_state(self, expected):
        [row] = read_queue_holds(self.database)
        self.assertEqual(row['state'], expected)
        self.assertEqual(row['nodes'], [] if expected == 'RETIRED_PRELAUNCH' else [dispatcher.NODE])

    def expire_at_prepare(self, args, *, timeout):
        raw = self.published_files(args, timeout=timeout)
        if args[-1] == self.fixture.publication+':'+self.last_blob:
            self.now = self.packet['expires_local_monotonic'] + 1
        return raw

    def test_real_frozen_main_arms_before_injected_transport_and_remains_unknown(self):
        observed = []

        def record(body, out, err, timeout):
            self.assert_state('UNKNOWN')
            self.assertGreater(timeout, 0)
            self.assertEqual(json.loads(self.intent.read_bytes())['body_SHA256'], sha256(body).hexdigest())
            observed.append(sha256(body).hexdigest())
            return SimpleNamespace(returncode=0)

        result = self.invoke(record)
        self.assertEqual(len(observed), 1)
        self.assertEqual(result['launcher_returncode'], 0)
        self.assertTrue(result['transport_attempted'])
        self.assertIsNone(result['actual_slurm_submissions'])
        self.assertFalse(result['terminal_verified'])
        self.assertFalse(self.proof.exists())
        self.assert_state('UNKNOWN')
        with self.assertRaisesRegex(ValueError, 'reconcile without replay'):
            self.invoke(self.reject_transport)

    def test_expired_exact_callback_retires_with_durable_evidence_and_admits_replacement(self):
        result = self.invoke(self.reject_transport, publication_response=self.expire_at_prepare)
        self.assertTrue(result['callback_reached'])
        self.assertFalse(result['transport_attempted'])
        self.assertEqual(result['actual_slurm_submissions'], 0)
        self.assert_state('RETIRED_PRELAUNCH')
        raw = self.proof.read_bytes()
        proof = json.loads(raw)
        self.assertEqual(proof['reason'], 'DEVICE_ADMISSION_EXPIRED_AT_ORIGINAL_CALLBACK')
        self.assertEqual(proof['remote_outcome'], 'NOT_ATTEMPTED')
        self.assertEqual(proof['body_SHA256'], json.loads(self.intent.read_bytes())['body_SHA256'])
        self.assertEqual(proof['marker_SHA256']['intent.json'], sha256(self.intent.read_bytes()).hexdigest())
        self.assertEqual(proof['device_receipt'], self.snapshot['local_receipt'])
        self.assertEqual(proof['handoff_SHA256'], self.packet['handoff_SHA256'])
        self.assertGreater(proof['callback_monotonic'], proof['expires_local_monotonic'])
        queue = Queue(self.database)
        with closing(queue.db):
            stored = queue.db.execute('SELECT result FROM runs WHERE id=?', (result['run'],)).fetchone()[0]
        self.assertEqual(json.loads(stored)['receipt_SHA256'], sha256(raw).hexdigest())

        fleet = self.fixture.fleet
        fleet.now = self.now
        fleet.snapshot['request_id'] = 'e' * 32
        board = dict(self.backlog, items=self.backlog['items']+[fleet.trial(2, 'validation', dispatcher.NODE)])
        report = build_handoff(fleet.verified_snapshot(), board, repo=self.repo,
            holds=read_queue_holds(self.database), now=self.now, boot_anchor=self.anchor)
        [replacement] = report['coordinator_handoffs']
        self.assertEqual(replacement['candidate'], 'trial-2')
        self.assertEqual(report['outstanding_training_ready'], 0)

    def test_error_before_callback_keeps_original_claim_without_proof(self):
        def failed(*_, **__):
            raise TimeoutError('publication lookup timed out before the launcher callback')

        result = self.invoke(self.reject_transport, publication_response=failed)
        self.assertEqual(result['launcher_error'], 'TimeoutError')
        self.assertFalse(result['callback_reached'])
        self.assertFalse(self.intent.exists())
        self.assertFalse(self.proof.exists())
        self.assert_state('DISPATCHING')
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.invoke(self.reject_transport)

    def test_expiry_with_mismatched_body_is_not_retired(self):
        original_load = dispatcher.load_scope

        def changed_scope(*args):
            original_main, files, worker = original_load(*args)

            def main(publication, callback):
                return original_main(publication, lambda body, *rest: callback(body + b'\n', *rest))

            return main, files, worker

        with patch.object(dispatcher, 'load_scope', side_effect=changed_scope):
            result = self.invoke(self.reject_transport, publication_response=self.expire_at_prepare)
        self.assertTrue(result['callback_reached'])
        self.assertFalse(self.proof.exists())
        self.assert_state('DISPATCHING')

    def test_boot_anchor_change_prevents_retirement(self):
        def clock_changed(args, *, timeout):
            raw = self.expire_at_prepare(args, timeout=timeout)
            if self.now > self.packet['expires_local_monotonic']:
                self.anchor += 3
            return raw

        result = self.invoke(self.reject_transport, publication_response=clock_changed)
        self.assertTrue(result['callback_reached'])
        self.assertFalse(result['transport_attempted'])
        self.assertFalse(self.proof.exists())
        self.assert_state('DISPATCHING')

    def test_expiry_while_committing_unknown_keeps_reservation(self):
        original_submission = Queue.submission

        def late_submission(queue, *args, **kwargs):
            original_submission(queue, *args, **kwargs)
            self.now = self.packet['expires_local_monotonic'] + 1

        with patch.object(Queue, 'submission', late_submission):
            result = self.invoke(self.reject_transport)
        self.assertTrue(result['callback_reached'])
        self.assertFalse(result['transport_attempted'])
        self.assertFalse(self.proof.exists())
        self.assert_state('UNKNOWN')

    def test_guard_evidence_write_failure_cannot_release(self):
        with patch.object(dispatcher, 'write_once', side_effect=OSError('could not save proof')):
            result = self.invoke(self.reject_transport, publication_response=self.expire_at_prepare)
        self.assertTrue(result['callback_reached'])
        self.assertFalse(result['transport_attempted'])
        self.assertFalse(self.proof.exists())
        self.assert_state('DISPATCHING')

    def test_preexisting_claim_is_never_taken_over(self):
        queue = Queue(self.database)
        with closing(queue.db):
            claim_handoff(queue, self.snapshot, self.backlog, self.packet, dispatcher.INTENT,
                          repo=self.repo, now=self.now, boot_anchor=self.anchor)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.invoke(self.reject_transport, publication_response=self.reject_transport)
        self.assert_state('DISPATCHING')
        self.assertFalse(self.intent.exists())


if __name__ == '__main__':
    unittest.main()
