"""Stdlib CPU checks; synthetic owner fixtures are never target evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gencuda.frozen_bootstrap import bind_frozen_framework, OriginalDeadline


class FrozenBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        tmp = Path(self.temporary.name).resolve()
        self.root = tmp / 'frozen'
        self.root.mkdir()
        (self.root / 'a.py').write_bytes(b'value = 1\n')
        (self.root / 'b.json').write_bytes(b'{"fixed":true}\n')
        files = {p.name: dict(bytes=p.stat().st_size, SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              mode=p.stat().st_mode & 0o777) for p in self.root.iterdir()}
        raw = json.dumps(dict(files=files)).encode()
        self.work = tmp / 'request'
        self.args = dict(request_id='a' * 32, claim_commit='b' * 40, source_manifest_SHA256='c' * 64,
                         framework_root=self.root, framework_manifest=raw,
                         framework_manifest_SHA256=hashlib.sha256(raw).hexdigest(), epoch=1000., deadline=1100.,
                         wall=lambda: 1001., monotonic=lambda: 10.,
                         owner_reader=lambda: dict(PID=101, PGID=101, startticks=123, host='CPU-fixture', boot_id_SHA256='d' * 64))

    def test_readonly_reference_records_owner_without_copying_source(self):
        result = bind_frozen_framework(self.work, **self.args)
        receipt = json.loads((self.work / 'bootstrap.json').read_bytes())
        self.assertEqual(receipt['owner']['startticks'], 123)
        self.assertEqual(receipt['execution_result'], 'UNKNOWN')
        self.assertEqual(result['files_verified'], 2)
        self.assertEqual(result['framework_files_copied'], 0)
        self.assertEqual(result['required_bind_mode'], 'read-only')
        self.assertFalse((self.work / 'framework').exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a.py', 'b.json'])

    def test_existing_partial_request_is_never_replayed_or_overwritten(self):
        self.work.mkdir()
        (self.work / 'partial').write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            bind_frozen_framework(self.work, **self.args)
        self.assertEqual((self.work / 'partial').read_bytes(), b'original')
        self.assertFalse((self.work / 'bootstrap.json').exists())

    def assert_failed_with_owner(self):
        with self.assertRaises(ValueError):
            bind_frozen_framework(self.work, **self.args)
        self.assertEqual(json.loads((self.work / 'bootstrap.json').read_bytes())['owner']['PID'], 101)
        failure = json.loads((self.work / 'bootstrap-failure.json').read_bytes())
        self.assertEqual(failure['execution_result'], 'UNKNOWN')
        self.assertFalse((self.work / 'framework-reference.json').exists())

    def test_changed_bytes_leave_original_owner_and_unknown(self):
        (self.root / 'a.py').write_bytes(b'value = 9\n')
        self.assert_failed_with_owner()

    def test_missing_file_leaves_original_owner_and_unknown(self):
        (self.root / 'a.py').unlink()
        self.assert_failed_with_owner()

    def test_wrong_manifest_leaves_original_owner_and_unknown(self):
        self.args['framework_manifest_SHA256'] = '0' * 64
        self.assert_failed_with_owner()

    def test_manifest_path_escape_is_rejected_after_owner_record(self):
        m = json.loads(self.args['framework_manifest'])
        m['files']['../escape'] = m['files'].pop('a.py')
        raw = json.dumps(m).encode()
        self.args.update(framework_manifest=raw, framework_manifest_SHA256=hashlib.sha256(raw).hexdigest())
        self.assert_failed_with_owner()
        self.assertFalse((self.work.parent / 'escape').exists())

    def test_monotonic_deadline_cannot_reset_when_wallclock_stalls(self):
        wall, mono = [1001.], [10.]
        clock = OriginalDeadline(1000., 1010., wall=lambda: wall[0], monotonic=lambda: mono[0])
        mono[0] = 20.
        with self.assertRaises(TimeoutError):
            clock.check()

    def test_expired_epoch_is_rejected_before_creating_workspace(self):
        self.args['wall'] = lambda: 1100.
        with self.assertRaises(TimeoutError):
            bind_frozen_framework(self.work, **self.args)
        self.assertFalse(self.work.exists())


if __name__ == '__main__':
    unittest.main()
