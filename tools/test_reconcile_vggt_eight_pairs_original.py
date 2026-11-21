import json
from pathlib import Path
import tempfile
import unittest

import reconcile_vggt_eight_pairs_original as reader


class VGGTOriginalHoldTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve(strict=True) / reader.LABEL
        self.rows = {phase: [f'{reader.PARENT}.{idx}', reader.LABEL + '-' + phase, 'COMPLETED', '0:0', '10',
                             reader.NODE, 'cpu=4,mem=64G,node=1,gres/dcu=1' if phase == 'gpu' else 'cpu=2,mem=2G,node=1']
                     for idx, phase in enumerate(reader.PHASES, 710)}
        self.epoch = 1794878720.0

    def put(self, name, value):
        self.root.mkdir(exist_ok=True)
        (self.root / name).write_text(json.dumps(value), encoding='utf8')

    def fixture(self, *, gpu='COMPLETED', clean=True, assignment=True):
        self.rows['gpu'][2:4] = [gpu, '0:0' if gpu == 'COMPLETED' else '1:0']
        self.put('intent.json', dict(request=reader.REQUEST, publication=reader.PUBLICATION,
                                     parent=reader.PARENT, node=reader.NODE, epoch=self.epoch, deadline=self.epoch+1800))
        records = {}
        for phase, row in self.rows.items():
            passed = phase != 'gpu' or gpu == 'COMPLETED'
            record = dict(row=row, passed=passed, rc=0 if passed else None, terminal_visible=True,
                          original_deadline=self.epoch + 1800)
            self.put(phase + '-accounting.json', record)
            records[phase] = record
        pci = [f'0000:{i:02x}:00.0' for i in range(8)]
        for index, (phase, name) in enumerate((('pre', 'before.json'), ('post', 'after.json')), 1):
            value = dict(step=self.rows[phase][0], node=reader.NODE, pci=pci, pci_to_kfd=None,
                         kind='training_qwen_device_snapshot', mode='host_complete', full_fd_inventory_performed=True,
                         holder_inventory_basis='kernel_KFD_plus_sameUID_FD_complete',
                         kfd_holder_count=int(not clean and phase == 'post'), boot_id='fixture-boot', monotonic_ns=index)
            self.put(name, value)
            self.put(phase + '.log', value)
        if assignment:
            self.put('assignment.json', dict(step=self.rows['gpu'][0], node=reader.NODE, pci=pci[0],
                                            kfd=dict(pci=pci[0]), pid=17, startticks=1039))
        self.put('outcome.json', dict(request=reader.REQUEST, publication=reader.PUBLICATION, stages=records,
                                      passed=gpu == 'COMPLETED'))

    def reconcile(self, rows=None):
        if rows is None:
            rows = self.rows.values()
        raw = '\n'.join('|'.join(row) for row in rows).encode()
        return reader.reconcile(self.root, raw)

    def test_empty_namespace_and_accounting_stay_unknown(self):
        result = self.reconcile([])
        self.assertEqual(result['state'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'original_namespace_missing')
        self.assertFalse(result['hold_retirement_evidence'])
        self.assertFalse(result['replay_allowed'])

    def test_bound_actual_rows_and_separate_cleanup_provide_release_evidence_only(self):
        self.fixture()
        result = self.reconcile()
        self.assertEqual(result['state'], 'ORIGINAL_TERMINAL_POST_CLEAN')
        self.assertTrue(result['independent_post_clean'])
        self.assertTrue(result['hold_retirement_evidence'])
        self.assertTrue(result['assignment_observed'])
        self.assertEqual(result['phase_rows']['gpu'][0], f'{reader.PARENT}.711')
        self.assertIn('post.log', result['original_receipt_SHA256'])
        self.assertFalse(result['training_result_audited'])

    def test_gpu_failure_stays_failure_with_independent_post(self):
        self.fixture(gpu='TIMEOUT', assignment=False)
        result = self.reconcile()
        self.assertEqual(result['phase_rows']['gpu'][2], 'TIMEOUT')
        self.assertTrue(result['hold_retirement_evidence'])
        self.assertFalse(result['original_collector_passed'])
        self.assertFalse(result['assignment_observed'])

    def test_missing_post_or_running_gpu_cannot_supply_hold_retirement(self):
        self.fixture()
        result = self.reconcile([self.rows['pre'], self.rows['gpu']])
        self.assertTrue(result['gpu_terminal'])
        self.assertFalse(result['hold_retirement_evidence'])
        self.rows['gpu'][2] = 'RUNNING'
        result = self.reconcile()
        self.assertEqual(result['state'], 'UNKNOWN')
        self.assertFalse(result['gpu_terminal'])

    def test_dirty_inventory_and_changed_independent_log_cannot_supply_retirement(self):
        self.fixture(clean=False)
        self.assertFalse(self.reconcile()['hold_retirement_evidence'])
        self.fixture()
        altered = json.loads((self.root / 'post.log').read_bytes())
        self.put('post.log', dict(altered, boot_id='another-boot'))
        self.assertFalse(self.reconcile()['independent_post_clean'])

    def test_duplicate_label_and_foreign_request_cannot_supply_retirement(self):
        self.fixture()
        result = self.reconcile([*self.rows.values(), self.rows['gpu']])
        self.assertFalse(result['hold_retirement_evidence'])
        original = json.loads((self.root / 'intent.json').read_bytes())
        self.put('intent.json', dict(original, request='f' * 32))
        self.assertFalse(self.reconcile()['hold_retirement_evidence'])

    def test_gres_and_producer_row_mismatch_keep_the_hold(self):
        self.fixture()
        self.rows['gpu'][6] = 'cpu=4,mem=64G,node=1,gres/dcu=2'
        self.assertFalse(self.reconcile()['hold_retirement_evidence'])
        self.rows['gpu'][6] = 'cpu=4,mem=64G,node=1,gres/dcu=1'
        self.rows['post'][0] = reader.PARENT + '.999'
        self.assertFalse(self.reconcile()['hold_retirement_evidence'])


if __name__ == '__main__':
    unittest.main()
