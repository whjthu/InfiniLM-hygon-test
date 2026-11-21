from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import preflight_vggt_tum_bidirectional30 as preflight


PARENT = b'1103606|original-parent|RUNNING|0:0|10|node-set|cpu=2560,node=40,gres/dcu=320'


class VggtPreflightTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()

    def put(self, path, raw):
        path = self.base / path
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def assets(self):
        model = self.base / preflight.WEIGHT_PATH
        self.put(preflight.WEIGHT_PATH, b'stat-only-fixture')
        st = model.stat()
        model_stat = (st.st_size, st.st_ino, st.st_mtime_ns)
        framework = {}
        for i in range(393):
            name = f'module/{i}.py'
            raw = f'content_{i}'.encode()
            digest = self.put(preflight.FRAMEWORK_BASE / 'framework' / name, raw)
            framework[name] = dict(SHA256=digest, bytes=len(raw))
        manifest = self.put(preflight.FRAMEWORK_BASE / 'source/FRAMEWORK_FILES.json',
                            json.dumps({'files': framework}).encode())
        data = {}
        for i in range(17):
            name = f'pair/{i}.raw'
            raw = f'fixture_{i}'.encode()
            digest = self.put(preflight.INPUT_BASE / 'data' / name, raw)
            data[name] = dict(sha=digest, bytes=len(raw))
        digest = self.put(preflight.UPSTREAM / 'source.py', b'provider')
        self.put('training-t06-sdxl-runtime-v1/site/package', b'present')
        return (json.dumps({'data_files': data}).encode(),
                json.dumps({'files': [dict(name='source.py', SHA256=digest, bytes=8)]}).encode(),
                model_stat, manifest)

    def test_real_source_only_package_identity(self):
        sources, count = preflight.check_source(preflight.REPO)
        self.assertEqual(count, 15)
        self.assertIn('original_claim.json', sources)
        self.assertIn('UPSTREAM_FILES.json', sources)

    def test_all_assets_checked_without_opening_model(self):
        data, upstream, model_stat, manifest = self.assets()
        path_open = Path.open
        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path, self.base / preflight.WEIGHT_PATH)
            return path_open(path, *args, **kwargs)
        with patch.object(preflight, 'FRAMEWORK_SHA', manifest), patch.object(Path, 'open', guarded_open):
            check = preflight.verify_assets(self.base, data, upstream, model_stat=model_stat)
        self.assertEqual(check['verified_framework_files'], 393)
        self.assertEqual(check['verified_training_media_files'], 17)
        self.assertEqual(check['model_content_bytes_read'], 0)
        self.assertFalse(check['runtime_imported'])

    def test_changed_media_or_model_stat_fail(self):
        data, upstream, model_stat, manifest = self.assets()
        self.put(preflight.INPUT_BASE / 'data/pair/2.raw', b'same_size')
        with patch.object(preflight, 'FRAMEWORK_SHA', manifest):
            with self.assertRaisesRegex(ValueError, 'original asset bytes differ'):
                preflight.verify_assets(self.base, data, upstream, model_stat=model_stat)
            with self.assertRaisesRegex(ValueError, 'original model file identity differs'):
                preflight.verify_assets(self.base, data, upstream, model_stat=(0,0,0))

    def test_old_namespace_and_missing_parent_fail_closed(self):
        self.assertTrue(preflight.check_namespace(PARENT, self.base)['original_read_only_guard_passed'])
        (self.base / preflight.LABEL).mkdir()
        self.assertFalse(preflight.check_namespace(PARENT, self.base)['original_read_only_guard_passed'])
        with self.assertRaisesRegex(ValueError, 'did not include the original parent'):
            preflight.check_namespace(PARENT.replace(b'1103606', b'1087656'), self.base)

    def test_late_original_label_does_not_produce_admission(self):
        row = f'1103606.801|{preflight.LABEL}-gpu|RUNNING|0:0|2|{preflight.NODE}|cpu=4,node=1,gres/dcu=1'.encode()
        with patch.object(preflight, 'check_source', return_value=(
                {'original_claim.json': b'data', 'UPSTREAM_FILES.json': b'provider'}, 15)), patch.object(
                    preflight, 'verify_assets', return_value={'verified_training_media_files':17}):
            for final, expected in ((PARENT, 'TARGET_ASSETS_VERIFIED_DEVICE_PENDING'),
                                    (PARENT+b'\n'+row, 'ORIGINAL_REQUIRES_RECONCILIATION')):
                raw = iter((PARENT, final))
                outcome = preflight.preflight(preflight.REPO,self.base,get_accounting=lambda: next(raw),
                                               placement=lambda repo: 123)
                self.assertEqual(outcome['state'], expected)
                self.assertFalse(outcome['training_READY'])
                self.assertFalse(outcome['PCI_KFD_PID_verified'])
        self.assertFalse((self.base / preflight.LABEL).exists())


if __name__ == '__main__':
    unittest.main()
