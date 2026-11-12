"""Saved-pair identity and original CPU epoch are checked before Torch import."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ENTRY = Path(__file__).resolve().parents[1] / 'tools/hygon_sdxl_operand_attribution.py'
SPEC = importlib.util.spec_from_file_location('saved_sdxl_operands', ENTRY)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SavedOperandEntryTests(unittest.TestCase):
    def invoke(self, directory, epoch, end):
        root = Path(directory)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')
        args = [sys.executable, '-B', str(ENTRY), '--a100', str(root/'a.pt'), '--hip', str(root/'h.pt'),
                '--output', str(root/'result.json'), '--epoch', str(epoch), '--deadline', str(end)]
        return subprocess.run(args, env=env, capture_output=True, timeout=10)

    def test_expired_run_retains_failure(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.invoke(d, 1, 2).returncode, 1)
            value = json.loads((Path(d)/'result.json').read_bytes())
            self.assertEqual(value['error_type'], 'TimeoutError')
            self.assertFalse(value['Model_GO'])

    def test_unpinned_tensor_archive_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'a.pt').write_bytes(b'untrusted')
            now = time.time()
            self.assertEqual(self.invoke(d, now-1, now+30).returncode, 1)
            self.assertEqual(json.loads((Path(d)/'result.json').read_bytes())['error_type'], 'ValueError')

    def test_original_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'
            p.write_bytes(b'{"original":true}')
            now = time.time()
            self.assertEqual(self.invoke(d, now-1, now+30).returncode, 1)
            self.assertEqual(p.read_bytes(), b'{"original":true}')

    def test_nonfinite_or_rebased_budget_rejected(self):
        for a, b in [(float('nan'), 2), (1, float('inf')), (1, 222), (3, 2)]:
            with self.subTest(epoch=a, deadline=b), self.assertRaises(ValueError):
                module.check_deadline(a, b)


if __name__ == '__main__':
    unittest.main()
