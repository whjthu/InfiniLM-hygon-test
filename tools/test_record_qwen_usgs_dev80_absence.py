from copy import deepcopy
import json
from shutil import copyfile
import unittest

from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from prepare_training80_dispatch_probe_args import probe_arguments
import reconcile_qwen_usgs_dev80_original as original
import record_qwen_usgs_dev80_absence as recorder
from report_training80_backlog import BACKLOG, TRAINING80_LEDGER
import test_report_training80_backlog as fixtures
from tuning_queue import Queue, digest


class Dev80AbsenceTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixtures.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.repo = self.fleet.repo.resolve()
        self.ledger = json.loads(TRAINING80_LEDGER.read_bytes())
        [self.dev] = [job for job in json.loads(BACKLOG.read_bytes())['items'] if job['id'] == recorder.JOB]
        for name in (original.AUTH_RECEIPT, original.CLAIM):
            path = self.repo/name
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(original.REPO/name, path)
        self.base = self.repo/'cluster-fixture'
        self.base.mkdir()
        self.node = original.NODES['base']
        self.training = self.fleet.trial(node=self.node)
        self.backlog = dict(minimum_ready_training=5, items=[self.dev, self.training])
        self.preflight = [dict(case='T01.domain-usgs-exposure25-resume20', node=self.node, requested_DCU=8,
                               target_preflight_state='WAITING_AUTHENTICATED_TARGET', frozen_rendezvous_port=29918)]

    def target(self, epoch=12, last=None):
        parent = b'1103606|original-parent|RUNNING|0:0|10|node-set|cpu=2560,node=40,gres/dcu=320'
        accounts = iter((parent, parent if last is None else last))
        return original.reconcile(self.repo, self.base, account=lambda: next(accounts),
                                  membership=lambda _: dict(observed_epoch=epoch,
                                      node_membership=dict.fromkeys(original.NODES.values(), 'NO_NUMBERED_STEP')))

    def emit(self, report):
        return json.dumps(report, allow_nan=False).encode()

    def handoff(self, snapshot=None, holds=()):
        return build_handoff(self.fleet.verified_snapshot() if snapshot is None else snapshot, self.backlog,
                             repo=self.repo, holds=holds, now=self.fleet.now, boot_anchor=self.fleet.anchor)

    def probe(self):
        return probe_arguments(self.repo, self.backlog, self.ledger, (), self.preflight, priority_only=True)

    def test_original_reader_receipt_unlocks_probing_only_for_new_work(self):
        before = self.probe()
        self.assertEqual(before['held_nodes'], sorted(original.NODES.values()))
        self.assertNotIn(29918, before['required_ports'])
        self.assertEqual(self.handoff()['coordinator_handoffs'], [])
        raw = self.emit(self.target())
        once = recorder.record_once(self.repo, raw)
        self.assertEqual(recorder.record_once(self.repo, raw), once)
        self.assertEqual((self.repo/once['path']).read_bytes(), raw)
        after = self.probe()
        self.assertEqual(after['held_nodes'], [])
        self.assertIn(29918, after['required_ports'])
        self.assertEqual(after['actual_submissions'], 0)
        self.assertEqual(self.handoff(snapshot=self.fleet.membership)['coordinator_handoffs'], [])
        self.assertEqual(self.handoff(snapshot=self.fleet.verified_snapshot(age=70))['coordinator_handoffs'], [])
        occupied = [dict(run='unrelated', state='UNKNOWN', nodes=[self.node])]
        self.assertEqual(self.handoff(holds=occupied)['coordinator_handoffs'], [])
        result = self.handoff()
        [packet] = result['coordinator_handoffs']
        self.assertEqual(packet['request'], self.training['launch']['request'])
        self.assertNotEqual(packet['request'], original.REQUEST)
        self.assertEqual(packet['verified_original_non_submissions_SHA256'],
                         digest(result['verified_original_non_submissions']))

    def test_inconsistent_receipt_fails_closed_and_holds_both_nodes(self):
        report = self.target()
        invalid = [dict(report, original_replay_allowed=True), dict(report, parent='1087656')]
        late = deepcopy(report)
        late['last']['accounting']['rows'] = {'old-phase': ['1103606.512']}
        invalid.append(late)
        for value in invalid:
            with self.subTest(state=value):
                with self.assertRaises(ValueError):
                    recorder.record_once(self.repo, self.emit(value))
        self.fleet.put(recorder.RECORD, late)
        result = self.probe()
        self.assertEqual(result['held_nodes'], sorted(original.NODES.values()))
        self.assertNotIn(29918, result['required_ports'])
        self.assertIn(recorder.JOB, result['invalid_or_manual_launches'])

    def test_changed_or_lost_reconciliation_bytes_invalidate_existing_handoff(self):
        recorder.record_once(self.repo, self.emit(self.target()))
        [packet] = self.handoff()['coordinator_handoffs']
        queue = Queue(self.repo/'queue-fixture.sqlite')
        self.addCleanup(queue.db.close)
        changed = self.target(epoch=13)
        with self.assertRaisesRegex(ValueError, 'preserve its original bytes'):
            recorder.record_once(self.repo, self.emit(changed))
        self.fleet.put(recorder.RECORD, changed)
        intent = self.training['launch']['attempt_receipts'][0]
        for absence, reason in ((False, 'handoff drift'), (True, 'no current device-admitted')):
            if absence:
                (self.repo/recorder.RECORD).unlink()
            with self.assertRaisesRegex(ValueError, reason):
                claim_handoff(queue, self.fleet.verified_snapshot(), self.backlog, packet, intent,
                              repo=self.repo, now=self.fleet.now, boot_anchor=self.fleet.anchor)
        self.assertFalse(queue.summary()['runs'])

    def test_combined_training_preflight_preserves_its_raw_bytes_and_checks_late_dev80(self):
        observed = dict(predecessor_namespace_absent=True, predecessor_rows={})
        combined = dict(state='TRAINING_ASSETS_VERIFIED_DEVICE_PENDING', read_only=True,
                        request=recorder.TRAINING_REQUEST, predecessor=self.target(), first=observed, last=deepcopy(observed))
        with self.assertRaisesRegex(ValueError, 'found a later Dev80 request'):
            recorder.record_once(self.repo, self.emit(dict(combined, last=dict(observed, predecessor_rows={'late': []}))))
        raw = self.emit(combined)
        result = recorder.record_once(self.repo, raw)
        self.assertEqual(result['request'], original.REQUEST)
        self.assertEqual((self.repo/result['path']).read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()
