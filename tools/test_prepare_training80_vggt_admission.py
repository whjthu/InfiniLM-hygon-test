from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
import unittest

from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
import prepare_training80_vggt_admission as admission
import test_report_training80_backlog as fixture
from tuning_queue import Queue


class VggtAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixture.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.repo = self.fleet.repo
        self.output = self.repo / 'local-only/fixture-vggt-admission'
        self.output.parent.mkdir()
        self.publication = 'a' * 40
        candidate = dict(id=admission.CANDIDATE, case='T12.domain-tum-bidirectional30', kind='training',
                         state='blocked', priority=16, placement={admission.NODE: 1}, min_updates=30,
                         attempt_receipts=[admission.INTENT, admission.ATTEMPT], next_action='fixture pending admission')
        self.backlog = dict(minimum_ready_training=5, items=[candidate])
        package = admission.PACKAGE
        manifest_path = package/'SOURCE_FILES.json'
        manifest = json.loads((admission.REPO/manifest_path).read_bytes())['files']
        claim = json.loads((admission.REPO/package/'CLAIM.json').read_bytes())
        self.members = {manifest_path.as_posix(), *((package/name).as_posix() for name in manifest),
                        *(entry['path'] for entry in claim['sources'].values())}
        for path in self.members:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(admission.REPO/path, target)
        upstream = json.loads((self.repo/claim['sources']['UPSTREAM_FILES.json']['path']).read_bytes())['files']
        self.target = dict(state='TARGET_ASSETS_VERIFIED_DEVICE_PENDING', request=admission.REQUEST,
                           source_manifest_SHA256=admission.SOURCE_MANIFEST, claim_SHA256=admission.CLAIM_SHA,
                           left_unmodified=True, parent=admission.PARENT, node=admission.NODE,
                           proposed_updates=30, verified_source_members=15, PCI_KFD_PID_verified=False,
                           training_READY=False, first_namespace_accounting_SHA256='b'*64,
                           original=dict(namespace_absent=True, accounting_label_matches=0, discovered_rows=[],
                                         original_read_only_guard_passed=True, raw_accounting_SHA256='c'*64),
                           membership_observed_epoch=1234.0, returned_epoch=1235.0,
                           assets=dict(model_stat_verified=True, original_model_pin=admission.WEIGHT_SHA,
                                       model_content_bytes_read=0, verified_framework_files=393,
                                       verified_training_media_files=17, verified_upstream_files=len(upstream),
                                       runtime_path_present=True, runtime_imported=False))

    def publication_check(self, repo, commit, members):
        self.assertEqual(repo, self.repo.resolve())
        self.assertEqual(commit, self.publication)
        self.assertEqual(set(members), self.members)

    def prepare(self, target=None, *, publish=None):
        raw = json.dumps(self.target if target is None else target).encode()
        return admission.prepare(self.repo, self.backlog, raw, self.output, self.publication,
                                 check_publication=publish or self.publication_check)

    def report(self, snapshot, backlog):
        return build_handoff(snapshot, backlog, repo=self.repo,
                             now=self.fleet.now, boot_anchor=self.fleet.anchor)

    def test_exact_target_overlay_enters_original_claim_only_with_fresh_devices(self):
        ref = self.prepare()
        backlog = json.loads((self.repo/ref['path']).read_bytes())
        self.assertEqual(self.backlog['items'][0]['state'], 'blocked')
        self.assertEqual(self.report(self.fleet.membership, backlog)['coordinator_handoffs'], [])
        self.assertEqual(self.report(self.fleet.verified_snapshot(age=60), backlog)['coordinator_handoffs'], [])
        snapshot = self.fleet.verified_snapshot()
        [packet] = self.report(snapshot, backlog)['coordinator_handoffs']
        self.assertEqual(packet['request'], admission.REQUEST)
        self.assertEqual(packet['placement_by_parent'], {admission.PARENT: {admission.NODE: 1}})
        self.assertEqual(packet['invocation']['arguments'], [self.publication])
        queue = Queue(self.repo/'test-claim.sqlite')
        self.addCleanup(queue.db.close)
        result = claim_handoff(queue, snapshot, backlog, packet, admission.INTENT, repo=self.repo,
                               now=self.fleet.now, boot_anchor=self.fleet.anchor)
        self.assertEqual(result['state'], 'DISPATCHING')
        self.assertEqual(result['actual_slurm_submissions'], 0)

    def test_rejects_different_claim_incomplete_assets_and_late_original_namespace(self):
        for section, key, value in ((None, 'claim_SHA256', '0'*64), ('assets', 'verified_training_media_files', 16),
                                    ('original', 'namespace_absent', False), (None, 'returned_epoch', 1325.0)):
            with self.subTest(key=key):
                target = deepcopy(self.target)
                (target if section is None else target[section])[key] = value
                with self.assertRaises(ValueError):
                    self.prepare(target)
                self.assertFalse(self.output.exists())

    def test_late_local_attempt_or_failed_publication_does_not_create_ready_overlay(self):
        def rejected(*_):
            raise ValueError('remote branch moved')
        with self.assertRaisesRegex(ValueError, 'branch moved'):
            self.prepare(publish=rejected)
        self.assertFalse(self.output.exists())
        def late(*_):
            self.fleet.put(admission.ATTEMPT, {'state': 'UNKNOWN'})
        with self.assertRaisesRegex(ValueError, 'intent appeared'):
            self.prepare(publish=late)
        with self.assertRaisesRegex(ValueError, 'prior original intent'):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
