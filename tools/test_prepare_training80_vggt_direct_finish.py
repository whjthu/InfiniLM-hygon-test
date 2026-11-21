from copy import deepcopy
from hashlib import sha256
import json
import unittest

import prepare_training80_vggt_direct_finish as binder
import read_vggt_tum_bidirectional30_terminal as reader
import test_read_vggt_tum_bidirectional30_terminal as fixtures
from prepare_training80_reconnect_worklist import WORKLIST, prepare
from tuning_queue import Queue, canonical, digest


class DirectFinishTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TerminalReaderTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.arms()
        self.target = (self.fixture.root / 'intent.json').read_bytes()
        local = json.loads(self.target)
        local.pop('node')
        local.pop('parent')
        local.update(body_SHA256='e' * 64, unknown_replay_allowed=False)
        self.local = json.dumps(local).encode()
        placement = {binder.PARENT: {binder.NODE: 1}}
        self.packet = dict(request=binder.REQUEST, case=binder.CASE, kind='training',
                           requested_DCU=1, placement_by_parent=placement, attempt_receipts=[binder.INTENT],
                           publication_commit=self.fixture.publication, manifest={'SHA256': self.fixture.manifest},
                           claim={'SHA256': self.fixture.claim})
        self.packet['handoff_SHA256'] = digest(self.packet)
        self.handle = dict(request_id=binder.REQUEST, direct_kind='training', placement_by_parent=placement,
                           parent=binder.PARENT, nodes=[binder.NODE], intent=binder.INTENT,
                           handoff_SHA256=self.packet['handoff_SHA256'])

    def originals(self):
        raw = '\n'.join('|'.join(row) for row in self.fixture.rows.values()).encode()
        audit = reader.verify(self.fixture.root, raw, self.fixture.manifest, self.fixture.claim)
        return raw, json.dumps(audit).encode()

    def test_real_reader_fixture_binds_a_success_and_ledger_inputs(self):
        raw, audit = self.originals()
        result = binder.prepare(('UNKNOWN', self.packet, self.handle), self.local, self.target, raw, audit)
        self.assertEqual((result['planned'], result['completed'], result['missing']), (30, 30, 0))
        self.assertTrue(result['passed'])
        self.assertTrue(result['post_clean'])
        self.assertFalse(result['promote'])
        self.assertEqual(result['actual_gpu_step'], self.fixture.rows['gpu'][0])
        self.assertEqual(result['actual_post_step'], self.fixture.rows['post'][0])
        self.assertEqual(result['evidence_SHA256'], sha256(audit).hexdigest())

    def test_gpu_failure_stays_failure_and_keeps_full_update_denominator(self):
        self.fixture.rows['gpu'][2:4] = ['TIMEOUT', '0:0']
        (self.fixture.root / 'arms/resume20/receipt.json').unlink()
        raw, audit = self.originals()
        result = binder.prepare(('SUBMITTED', self.packet, self.handle), self.local, self.target, raw, audit)
        self.assertEqual((result['planned'], result['completed'], result['missing']), (30, 25, 5))
        self.assertEqual(result['actual_gpu_state'], 'TIMEOUT')
        self.assertTrue(result['post_clean'])
        self.assertFalse(result['passed'])

    def test_rejects_running_accounting_and_unbound_cleanup(self):
        self.fixture.rows['gpu'][2] = 'RUNNING'
        raw, audit = self.originals()
        args = (('SUBMITTED', self.packet, self.handle), self.local, self.target)
        with self.assertRaisesRegex(ValueError, 'terminal and completed pre/post'):
            binder.prepare(*args, raw, audit)
        self.fixture.rows['gpu'][2] = 'COMPLETED'
        value = json.loads((self.fixture.root / 'after.json').read_bytes())
        self.fixture.put('after.json', dict(value, kfd_holder_count=1))
        raw, audit = self.originals()
        with self.assertRaisesRegex(ValueError, 'independent audit does not bind'):
            binder.prepare(*args, raw, audit)

    def test_rejects_wrong_target_epoch_hash_or_recorded_original_step(self):
        raw, audit = self.originals()
        claim = ('UNKNOWN', self.packet, self.handle)
        intent = json.loads(self.target)
        with self.assertRaisesRegex(ValueError, 'original epoch'):
            wrong = dict(intent, epoch=intent['epoch'] + 1, deadline=intent['deadline'] + 1)
            binder.prepare(claim, self.local, json.dumps(wrong).encode(), raw, audit)
        with self.assertRaisesRegex(ValueError, 'does not bind the actual accounting'):
            binder.prepare(claim, self.local, self.target, raw + b'\n', audit)
        with self.assertRaisesRegex(ValueError, 'does not bind the actual accounting'):
            binder.prepare(claim, self.local, self.target + b'\n', raw, audit)
        with self.assertRaisesRegex(ValueError, 'durable handle'):
            other = dict(self.handle, gpu_step=binder.PARENT + '.999')
            binder.prepare(('SUBMITTED', self.packet, other), self.local, self.target, raw, audit)
        with self.assertRaisesRegex(ValueError, 'claimed handoff bytes differ'):
            edited = deepcopy(self.packet)
            edited['manifest']['SHA256'] = '1' * 64
            binder.prepare(('UNKNOWN', edited, self.handle), self.local, self.target, raw, audit)

    def test_read_only_claim_to_existing_queue_finish_integration(self):
        database = self.fixture.root / 'queue.sqlite3'
        queue = Queue(database)
        self.addCleanup(queue.db.close)
        queue.db.execute('INSERT INTO candidates values(?,?)', ('fixture', '{}'))
        queue.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
                         (binder.DIRECT_RUN, 'fixture', binder.DIRECT_STAGE, 'UNKNOWN',
                          canonical(self.packet).decode(), json.dumps([binder.NODE]),
                          canonical(self.handle).decode(), None, self.fixture.epoch))
        before = sha256(database.read_bytes()).hexdigest()
        claim = binder.claimed_run(database)
        self.assertEqual(claim, ('UNKNOWN', self.packet, self.handle))
        self.assertEqual(sha256(database.read_bytes()).hexdigest(), before)
        raw, audit = self.originals()
        finish = binder.prepare(claim, self.local, self.target, raw, audit)
        self.assertEqual(queue.dispatch_state(binder.DIRECT_RUN)['state'], 'UNKNOWN')
        queue.finish(binder.DIRECT_RUN, finish)
        self.assertEqual(queue.dispatch_state(binder.DIRECT_RUN)['state'], 'TERMINAL')
        with self.assertRaisesRegex(ValueError, 'not active'):
            binder.claimed_run(database)

    def test_reconnect_worklist_exposes_the_pinned_independent_terminal_reader(self):
        from report_training80_backlog import TRAINING80_LEDGER

        plan = prepare(binder.REPO, json.loads(WORKLIST.read_bytes()), TRAINING80_LEDGER.read_bytes())
        row, = [candidate for candidate in plan['candidates'] if candidate['request'] == binder.REQUEST]
        probe = row['target_completion_reader']
        self.assertEqual(probe['entry']['path'], 'tools/read_vggt_tum_bidirectional30_terminal.py')
        launch_source = binder.REPO / probe['entry']['path']
        self.assertEqual(probe['entry']['SHA256'], sha256(launch_source.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
