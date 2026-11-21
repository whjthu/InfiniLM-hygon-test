from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest

import control_deep_inverse_rope_compile as control


class CompileControllerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.intent = Path(directory.name) / 'intent'
        self.output = Path(directory.name) / 'candidate'
        helper = runpy.run_path(str(control.REPO / control.ACCOUNTING))['wait_terminal']
        self.wait = lambda read, deadline, **kwargs: helper(read, deadline, interval=.001, **kwargs)
        self.row = ['1103606.600', control.LABEL, 'COMPLETED', '0:0', '8', control.NODE,
                    'cpu=4,mem=8G,node=1']
        self.calls = []

    def controller(self, rows, worker):
        rows = iter(rows)
        return control.controller(self.intent, self.output, 'a'*64, self.wait,
                                  query=lambda timeout: next(rows), observe_slot=lambda: None, run=worker)

    def good_worker(self, command, **kwargs):
        self.calls.append(command)
        epoch = json.loads((self.intent / 'intent.json').read_bytes())['epoch']
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        files = {'object_SHA256': 'inverse_output_rope.o', 'fresh_M_SHA256': 'inverse_output_rope.before.d',
                 'actual_MD_SHA256': 'inverse_output_rope.d'}
        receipt = {}
        for field, name in files.items():
            raw = name.encode()
            (self.output / name).write_bytes(raw)
            receipt[field] = sha256(raw).hexdigest()
        receipt.update(label=control.LABEL, original_step=self.row[0], node=control.NODE,
                       manifest_SHA256=control.MANIFEST, epoch=epoch, deadline=epoch+300,
                       GRES='none', planned_device_TUs=1, actual_device_TUs=1, actual_GPU_steps=0,
                       explicit_Runtime_calls=0, exports=control.EXPORTS, passed=True, completed=True,
                       device_qualified=False)
        (self.output / 'CPU-result.json').write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 0)

    def test_terminal_result_checks_actual_artifact_and_no_gpu(self):
        running = self.row[:]
        running[2] = 'RUNNING'
        result = self.controller([None, None, running, self.row], self.good_worker)
        self.assertEqual(result['state'], 'COMPILE_VERIFIED')
        self.assertEqual(result['observations'], [None, running, self.row])
        self.assertEqual(len(self.calls), 1)
        command = self.calls[0]
        self.assertIn('--cpus-per-task=4', command)
        self.assertIn('--mem=8192M', command)
        self.assertIn('--gres=none', command)
        self.assertNotIn('--rocm', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=-1', command[-1])

    def test_timeout_with_running_row_persists_unknown_and_refuses_replay(self):
        row = self.row[:]
        row[2] = 'RUNNING'
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 1)
        result = self.controller([None, *([row]*6)], timeout)
        self.assertEqual(result['state'], 'UNKNOWN')
        self.assertTrue(result['client_timeout'])
        self.assertFalse(result['terminal_visible'])
        with self.assertRaisesRegex(ValueError, 'namespace already exists'):
            self.controller([], timeout)

    def test_terminal_zero_requires_original_object_bytes(self):
        def corrupt(command, **kwargs):
            rc = self.good_worker(command, **kwargs)
            (self.output / 'inverse_output_rope.o').write_bytes(b'drift')
            return rc
        result = self.controller([None, self.row], corrupt)
        self.assertEqual(result['state'], 'TERMINAL_UNVERIFIED')
        self.assertIn('frozen bytes differ', result['error'])

    def test_accounting_foreign_or_duplicate_label_refused(self):
        line = '|'.join(self.row).encode()
        self.assertEqual(control.accounting_row(line), self.row)
        for raw in (line + b'\n' + line, line.replace(control.NODE.encode(), b'other-node')):
            with self.assertRaises(ValueError):
                control.accounting_row(raw)
        with self.assertRaisesRegex(ValueError, 'already submitted'):
            self.controller([self.row], self.good_worker)
        self.assertFalse(self.intent.exists())


if __name__ == '__main__':
    unittest.main()
