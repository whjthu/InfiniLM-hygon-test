from copy import deepcopy
from hashlib import sha256
import json
import unittest

import prepare_training80_vggt_direct_finish as binder
import read_vggt_tum_bidirectional30_terminal as reader
import test_read_vggt_tum_bidirectional30_terminal as fixtures
from prepare_training80_dispatch_probe_args import probe_arguments
from prepare_training80_qwen_admission import CANDIDATE as QWEN_CASE, REQUEST as QWEN_REQUEST
from prepare_training80_reconnect_worklist import WORKLIST, prepare
from prepare_training80_vggt_admission import CANDIDATE as VGGT_CASE
from report_training80_backlog import BACKLOG, TRAINING80_LEDGER, pin_file
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

    def test_published_closed_runs_do_not_protect_nodes_in_the_active_worklist(self):
        backlog = json.loads(BACKLOG.read_bytes())
        candidates = {candidate['id']: candidate for candidate in backlog['items']}
        ledger = TRAINING80_LEDGER.read_bytes()
        worklist = prepare(binder.REPO, json.loads(WORKLIST.read_bytes()), ledger)['candidates']
        closed = []
        for candidate, label, request, gpu, post, state, updates in (
            (VGGT_CASE, 'VGGT', binder.REQUEST, '1103606.498', '1103606.507', 'COMPLETED', 30),
            (QWEN_CASE, 'Qwen', QWEN_REQUEST, '1103606.501', '1103606.508', 'FAILED', 25),
        ):
            row = candidates[candidate]
            self.assertEqual(row['state'], 'terminal')
            raw = pin_file(binder.REPO, row['evidence'], ('results/bw1000/',))
            published = json.loads(raw)[label]
            self.assertEqual(published['request'], request)
            self.assertEqual(published['steps']['gpu'][0], gpu)
            self.assertEqual(published['steps']['gpu'][2], state)
            self.assertEqual(published['steps']['post'][0], post)
            self.assertTrue(published['post_clean'])
            self.assertEqual(published['independently_verified_updates'], updates)
            self.assertEqual(published['planned_updates'], 30)
            self.assertNotIn(request, {candidate['request'] for candidate in worklist})
            closed.append(dict(run=published['queue_run'], state='TERMINAL', nodes=[], direct_request=request))

        report = probe_arguments(binder.REPO, backlog, json.loads(ledger), closed, worklist)
        self.assertNotIn('f13r1n10', report['held_nodes'])
        self.assertNotIn('f13r4n09', report['held_nodes'])
        self.assertIn('f11r4n09', report['held_nodes'])
        self.assertIn('f11r4n19', report['held_nodes'])


if __name__ == '__main__':
    unittest.main()
