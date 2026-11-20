import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import control_minicpm_usgs_pair_target as driver


class MiniCPMControllerTests(unittest.TestCase):
    def run_controller(self, root, gpu_terminal):
        seen = []
        def simulate(phase, command, elapsed, epoch):
            seen.append(phase)
            step = driver.PARENT + '.' + str(len(seen))
            if phase in ('pre', 'post'):
                driver.write(root / (phase + '.log'), dict(step=step, node=driver.NODE, boot_id='fixture',
                    pci=['fixture'], pci_to_kfd=None, master_port=driver.PORT, owned_processes_remaining=[],
                    owned_registry_complete=True))
            elif not gpu_terminal:
                return dict(passed=False, terminal_visible=False, row=None)
            else:
                cfg = json.loads((root / 'worker-config.json').read_bytes())
                self.assertEqual(cfg['epoch'], epoch)
                self.assertEqual((cfg['deadline']-epoch, cfg['native_deadline']-epoch), (2400, 1980))
                driver.write(root / 'worker-assigned.json', dict(actual_step=step))
                driver.write(root / 'worker-outcome.json', dict(request=driver.REQUEST, epoch=epoch, passed=True))
                driver.write(root / 'paired-recovery.json', dict(exact_paired_recovery=True))
            return dict(passed=True, terminal_visible=True, row=[step])
        with patch.object(driver, 'RUN', root), patch.object(driver, 'accepted_stage', return_value={}), patch.object(
                driver, 'stage_worker', return_value={'worker.py': b'fixture'}), patch.object(driver, 'original_membership'), patch.object(
                driver, 'accounting', return_value=None), patch.object(driver, 'execute_step', side_effect=simulate) as execute:
            result = driver.controller('a'*64, 'b'*64, 'f'*40)
            with self.assertRaises(FileExistsError):
                driver.controller('a'*64, 'b'*64, 'f'*40)
            self.assertEqual(execute.call_count, len(seen))
        self.assertEqual(result, json.loads((root / 'training-outcome.json').read_bytes()))
        return result, seen

    def test_completed_fixture_runs_once_and_proves_independent_clean_step(self):
        with tempfile.TemporaryDirectory() as root:
            result, seen = self.run_controller(Path(root).resolve(), gpu_terminal=True)
        self.assertEqual(seen, ['pre', 'gpu', 'post'])
        self.assertTrue(result['completed'])
        self.assertTrue(result['clean_post'])
        self.assertFalse(result['independently_audited_training'])

    def test_unknown_gpu_is_preserved_without_cleanup_step_or_replay(self):
        with tempfile.TemporaryDirectory() as root:
            result, seen = self.run_controller(Path(root).resolve(), gpu_terminal=False)
        self.assertEqual(seen, ['pre', 'gpu'])
        self.assertFalse(result['phases']['gpu']['terminal_visible'])
        self.assertFalse(result['completed'])

    def test_exact_native_and_cpu_requests(self):
        gpu = driver.step_command('gpu', ['worker'])
        self.assertIn('--gres=dcu:8', gpu)
        self.assertIn('-c64', gpu)
        self.assertIn('--mem=224000M', gpu)
        self.assertIn(driver.NODE, gpu)
        self.assertEqual(gpu[gpu.index('-t')+1], '35')
        for phase in ('pre', 'post'):
            cpu = driver.step_command(phase, ['inventory'])
            self.assertIn('--gres=none', cpu)

    def test_runtime_mounts_preserve_original_model_inputs(self):
        command = driver.gpu_command(dict(path='worker config.json', SHA256='c'*64), 1000.0)
        mounts = [command[i+1] for i, entry in enumerate(command) if entry == '--bind']
        self.assertEqual([item for item in mounts if item.endswith(':rw')], [f'{driver.RUN}:{driver.RUN}:rw'])
        self.assertTrue(any('training-minicpm-original-model-v1' in item and item.endswith(':ro') for item in mounts))
        self.assertIn("--config 'worker config.json'", command[-1])
        self.assertIn('HF_HUB_OFFLINE=1', command[-1])
        self.assertIn('3080.0 - time.time()', command[-1])
        self.assertIn('/usr/bin/timeout --kill-after=10s', command[-1])

    def test_config_uses_frozen_internal_schema_and_original_epoch(self):
        value = driver.worker_config(1000.0, dict(ledger={'SHA256': 'd'*64}), {'SHA256': 'b'*64})
        self.assertEqual(value['schema'], 'MINICPM_USGS_TARGET_WORKER_CONFIG_V1')
        self.assertEqual((value['deadline'], value['native_deadline']), (3400.0, 2980.0))
        self.assertEqual(value['request'], driver.REQUEST)
        with self.assertRaises(ValueError):
            driver.worker_config(float('nan'), {}, {})


if __name__ == '__main__':
    unittest.main()
