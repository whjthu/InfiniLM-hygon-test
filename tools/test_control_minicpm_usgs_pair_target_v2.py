from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import control_minicpm_usgs_pair_target as original
import control_minicpm_usgs_pair_target_v2 as driver


class MiniCPMControllerV2Tests(unittest.TestCase):
    def manifest_sha(self, module):
        return sha256((module.REPO / module.SOURCE_MANIFEST).read_bytes()).hexdigest()

    def test_source_manifest_binds_new_entry_and_the_same_workers(self):
        prior_sha, current_sha = map(self.manifest_sha, (original, driver))
        prior = json.loads((original.REPO / original.SOURCE_MANIFEST).read_bytes())
        current = json.loads((driver.REPO / driver.SOURCE_MANIFEST).read_bytes())
        old_entry = 'tools/control_minicpm_usgs_pair_target.py'
        new_entry = 'tools/control_minicpm_usgs_pair_target_v2.py'
        self.assertEqual(len(prior), 14)
        self.assertEqual(len(current), 14)
        self.assertEqual({k: v for k, v in prior.items() if k != old_entry},
                         {k: v for k, v in current.items() if k != new_entry})
        self.assertEqual(driver.stage_worker(current_sha), original.stage_worker(prior_sha))
        with self.assertRaises(ValueError):
            driver.stage_worker(prior_sha)

    def test_controller_uses_own_path_for_both_cpu_boundaries(self):
        snapshots = []
        before = dict(step='1103606.001', node=driver.NODE, boot_id='fixture', pci=['fixture'],
                      pci_to_kfd={}, master_port=driver.PORT, owned_processes_remaining=[],
                      owned_registry_complete=True)

        def simulate(phase, command, elapsed, epoch):
            if phase == 'gpu':
                return dict(passed=False, terminal_visible=True, row=['1103606.002'])
            snapshots.append(command)
            step = before['step'] if phase == 'pre' else '1103606.003'
            driver.write(driver.RUN / (phase + '.log'), dict(before, step=step))
            return dict(passed=True, terminal_visible=True, row=[step])

        with TemporaryDirectory() as root, patch.object(driver, 'RUN', Path(root).resolve()), \
                patch.object(driver, 'accepted_stage', return_value={}), \
                patch.object(driver, 'stage_worker', return_value={}), patch.object(driver, 'original_membership'), \
                patch.object(driver, 'accounting', return_value=None), \
                patch.object(driver, 'execute_step', side_effect=simulate):
            result = driver.controller('a'*64, 'b'*64, 'c'*40)
            intent = json.loads((Path(root) / 'training-intent.json').read_bytes())
        self.assertTrue(result['clean_post'])
        self.assertFalse(result['completed'])
        self.assertEqual(intent['request'], original.REQUEST)
        self.assertEqual([call[2:] for call in snapshots],
                         [[str(Path(driver.__file__).resolve()), '--snapshot', phase] for phase in ('pre', 'post')])

    def test_late_gpu_passes_bounded_argv_and_preserves_unknown_accounting(self):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])

        terminal = SimpleNamespace(wait_terminal=lambda *_a, **_kw: dict(terminal_visible=False, row=None))
        with TemporaryDirectory() as root, patch.object(driver, 'RUN', Path(root)), \
                patch.object(driver, 'accounting', return_value=None), patch.dict(sys.modules, slurm_accounting=terminal), \
                patch.object(driver.subprocess, 'run', side_effect=timeout) as run:
            result = driver.execute_step('gpu', ['fixture'], lambda: 1910, 1000.0)
            argv = run.call_args.args[0]
            stored = json.loads((Path(root) / 'gpu-accounting.json').read_bytes())
        self.assertEqual(argv[argv.index('-t')+1], '3')
        self.assertEqual(run.call_args.kwargs['timeout'], 190)
        self.assertEqual(stored['budget']['original_whole_deadline'], 3400.0)
        self.assertTrue(result['client_timeout'])
        self.assertFalse(result['terminal_visible'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
