import unittest
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import control_vggt_tum_cross_camera_dev as control


class VGGTControllerTests(unittest.TestCase):
    def run_original_step(self, phase, accounting, *, returncode=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        control.import_helpers()
        with patch.object(control, 'slurm_row', side_effect=accounting), patch.object(
                control.subprocess, 'run', return_value=control.subprocess.CompletedProcess([], returncode)) as run:
            result = control.run_step(root, phase, ['original-worker'], time.monotonic() + 1500)
        self.assertEqual(result, control.read_object(root / (phase + '-accounting.json')))
        return result, run

    def test_gpu_step_uses_original_node_and_exact_one_card(self):
        gpu = [control.PARENT + '.382', control.LABEL + '-gpu', 'COMPLETED', '0:0', '123', control.NODE,
               'cpu=4,gres/dcu=1,mem=64G,node=1']
        result, run = self.run_original_step('gpu', [None, gpu])
        self.assertTrue(result['passed'])
        run.assert_called_once()
        cmd = run.call_args.args[0]
        self.assertIn('--gres=dcu:1', cmd)
        self.assertIn(control.NODE, cmd)
        self.assertIn('--exact', cmd)

    def test_terminal_wrong_accelerator_count_denies_pass(self):
        row = [control.PARENT + '.383', control.LABEL + '-gpu', 'COMPLETED', '0:0', '123', control.NODE,
               'cpu=4,gres/dcu=2,mem=64G,node=1']
        result, _ = self.run_original_step('gpu', [None, row])
        self.assertTrue(result['terminal_visible'])
        self.assertFalse(result['passed'])

    def test_existing_label_and_corrupt_accounting_do_not_launch(self):
        for accounting in ('1087656.1|' + control.LABEL + '-gpu|RUNNING|0:0|1|f11r4n19|cpu=4\n', 'bad|layout'):
            with patch.object(control.subprocess, 'check_output', return_value=accounting), patch.object(
                    control.subprocess, 'run') as run, self.assertRaises(ValueError):
                control.import_helpers()
                control.run_step(Path('.'), 'gpu', [], time.monotonic() + 1500)
            run.assert_not_called()

    def test_only_original_score_directory_has_explicit_write_mount(self):
        path = Path('/opt/proof')
        cmd = control.gpu_command(path)
        mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == '--bind']
        self.assertEqual([mount for mount in mounts if mount.endswith(':rw')], [f'{path}:/pilot:rw'])
        self.assertIn(':'.join((str(control.RESULTS / control.SOURCES['alternate25'][0]), '/alternate', 'ro')), mounts)
        self.assertTrue(all(mount.endswith((':ro', ':rw')) for mount in mounts))

    def test_accounting_error_after_launch_is_recorded_as_unknown(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        control.import_helpers()
        with patch.object(control, 'slurm_row', side_effect=[None, ValueError('accounting error')]), patch.object(
                control.subprocess, 'run', return_value=control.subprocess.CompletedProcess([], 0)) as run:
            with self.assertRaisesRegex(ValueError, 'accounting error'):
                control.run_step(root, 'gpu', [], time.monotonic() + 1500)
        run.assert_called_once()
        result = control.read_object(root / 'gpu-accounting.json')
        self.assertFalse(result['terminal_visible'])
        self.assertIsNone(result['row'])


if __name__ == '__main__':
    unittest.main()
