from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

import prepare_minicpm_usgs_pair_target as stage


class MiniCPMTargetGateTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(stage.CORE))
        self.addCleanup(sys.path.remove, str(stage.CORE))
        self.handle = stage.PARENT + '.8888'
        self.worker = json.dumps(dict(source_dataset_SHA256=
            '0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62',
            passed=True, completed_rows=88)).encode()
        self.receipt = dict(state='MINICPM_USGS88_NATIVE_CPU_INDEPENDENT_METADATA_PASS', passed=True,
            planned_rows=88, completed_rows=88, missing_rows=0, failed_rows=0, actual_encode_calls=176,
            actual_collator_calls=88, suffix_EOS_and_real_separator_verified=True, dev_or_blind_read=False,
            actual_CPU_step=self.handle, actual_step_elapsed_seconds=11, request=stage.CPU_REQUEST,
            publication='f'*40, actual_GPU_allocation=0, actual_optimizer_updates=0, model_loaded=False,
            EOS_supervised=88, full_untruncated=88, metadata_pins={'worker-result.json':
                dict(bytes=len(self.worker), SHA256=sha256(self.worker).hexdigest())})
        self.accounting = ('|'.join([self.handle, stage.CPU_LABEL, 'COMPLETED', '0:0', '11', stage.NODE,
                                    'cpu=8,node=1,mem=12G'])+'\n').encode()

    def verify(self, *, accounting=None, worker=None):
        raw = json.dumps(self.receipt).encode()
        return stage.verify_native_cpu(raw, sha256(raw).hexdigest(),
                    self.accounting if accounting is None else accounting, self.worker if worker is None else worker)

    def test_authentic_fields_are_preserved_from_exact_step(self):
        binding = self.verify()
        self.assertEqual(binding['step'], self.handle)
        self.assertEqual(binding['worker_SHA256'], sha256(self.worker).hexdigest())

    def test_current_accounting_must_be_unique_and_terminal(self):
        for corrupted in (self.accounting * 2, self.accounting.replace(b'COMPLETED', b'RUNNING'),
                          self.accounting.replace(b'12G', b'12G,gres/dcu=1')):
            with self.subTest(accounting=corrupted), self.assertRaises(ValueError):
                self.verify(accounting=corrupted)

    def test_wrong_original_request_and_unpinned_worker_are_denied(self):
        with self.assertRaisesRegex(ValueError, 'intended train-only inputs'):
            self.verify(worker=b'{}\n')
        self.receipt['request'] = 'a'*32
        with self.assertRaisesRegex(ValueError, 'receipt identity'):
            self.verify()

    def test_file_identity_rejects_changed_original_and_oversize(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name).resolve() / 'asset'
            p.write_bytes(b'xyz')
            expected = sha256(b'xyz').hexdigest()
            self.assertEqual(stage.stable_file(p, size=3, digest=expected, collect=True), b'xyz')
            with self.assertRaises(ValueError):
                stage.stable_file(p, ceiling=2)
            p.write_bytes(b'zxy')
            with self.assertRaises(ValueError):
                stage.stable_file(p, size=3, digest=expected)


if __name__ == '__main__':
    unittest.main()
