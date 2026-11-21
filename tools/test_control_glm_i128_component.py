import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import control_glm_i128_component as controller


class GLMControllerTests(unittest.TestCase):
    def run_fixture(self, root, terminal, original_link='COMPLETED'):
        node = controller.NODE
        link = controller.link
        folder = root / 'candidates' / link.LABEL
        folder.mkdir(parents=True)
        result = folder / 'CPU-link-result.json'
        pin = link.write(result, dict(state='ISOLATED_HOST_LINK_PASS', actual_CPU_step=controller.PARENT+'.510'))
        (root/'results').mkdir()
        raw = f'{controller.PARENT}.510|{link.LABEL}|{original_link}|0:0|19|{node}|cpu=4,mem=8G,node=1\n'.encode()
        run = root / 'results' / controller.LABEL
        args = argparse.Namespace(publication='a'*40, link_sha256=pin)
        old_shape = ['0000:04:00.0']
        phases = []

        def step(out, phase, command, deadline):
            self.assertEqual(out, run)
            phases.append(phase)
            handle = controller.PARENT + '.' + str(520+len(phases))
            if phase in ('pre','post'):
                link.write(run/(phase+'.log'), dict(step=handle, pci=old_shape, boot_id='original', owned_registry_present=True))
            elif not terminal:
                return dict(passed=False, terminal_visible=False, row=None)
            else:
                providers, hip = {'fixture':'providers'}, {'fixture':'runtime'}
                binding = link.write(run/'binding.json', dict(expected_pci=old_shape[0], providers=providers, hip=hip))
                link.write(run/'component-private.json', dict(passed=True, completed=True, fixture_binding_sha256=controller.INPUT_MF,
                    normal_stream_and_event_destroy=True, binding_sha256=binding, providers=providers, hip=hip,
                    device=dict(pci=old_shape[0],visible_devices=1), passed_timed_launches=3840,
                    numeric_counts={'PASS':24}, pair_counts={'PASS':48}))
                link.write(run/'worker-owner.json', dict(step=handle, request=controller.GPU_REQUEST, pci=old_shape[0]))
            return dict(passed=True, terminal_visible=True, row=[handle])

        with patch.object(controller,'ROOT',root), patch.object(controller,'RESULTS',root/'results'), patch.object(
                controller,'accounting',return_value=raw), patch.object(controller,'node_is_clear'), patch.object(
                controller,'run_step',side_effect=step) as invoke:
            if original_link != 'COMPLETED':
                with self.assertRaises(ValueError):
                    controller.run_component(args)
                self.assertFalse(run.exists())
                invoke.assert_not_called()
                return None, phases
            output = controller.run_component(args)
            with self.assertRaises(FileExistsError):
                controller.run_component(args)
            self.assertEqual(invoke.call_count, len(phases))
        return output, phases

    def test_exact_one_device_and_zero_card_link_commands(self):
        link = controller.srun('link',['tool'])
        gpu = controller.srun('gpu',['worker'])
        self.assertIn('--gres=none', link)
        self.assertIn('--gres=dcu:1', gpu)
        self.assertIn('-c4', gpu)
        self.assertIn('--mem=16384M', gpu)
        self.assertIn(controller.NODE, gpu)
        self.assertNotEqual(controller.LINK_REQUEST, controller.GPU_REQUEST)

    def test_only_link_mode_can_write_candidate_directory(self):
        for_link = controller.mount(Path('/scope-test'), ['tool'], True)
        actual_gpu = controller.mount(Path('/scope-test'), ['worker'])
        self.assertFalse(any(x.endswith('/refactor/candidates:rw') for x in actual_gpu))
        self.assertTrue(any(x.endswith('/refactor/candidates:rw') for x in for_link))
        self.assertIn('CUDA_VISIBLE_DEVICES=-1', for_link[-1])
        self.assertIn(controller.GLM_ENV, actual_gpu[-1])

    def test_only_terminal_link_allows_original_gpu_and_clean_post(self):
        with tempfile.TemporaryDirectory() as loc:
            root = Path(loc).resolve()
            report, phases = self.run_fixture(root, terminal=True)
            run = root / 'results' / controller.LABEL
            data = controller.small_json(run/'component-private.json')
            data['passed_timed_launches'] = 3839
            (run/'component-private.json').unlink()
            controller.link.write(run/'component-private.json', data)
            self.assertFalse(controller.component_result_pass(run, report['phases']['gpu']['row']))
        self.assertEqual(phases, ['pre','gpu','post'])
        self.assertTrue(report['original_post_clean'])
        self.assertTrue(report['passed'])
        self.assertFalse(report['full_model_qualified'])

    def test_unseen_gpu_terminal_and_inflight_link_do_not_replay_or_release(self):
        with tempfile.TemporaryDirectory() as loc:
            report, phases = self.run_fixture(Path(loc).resolve(), terminal=False)
        self.assertEqual(phases, ['pre','gpu'])
        self.assertFalse(report['original_post_clean'])
        self.assertFalse(report['passed'])
        with tempfile.TemporaryDirectory() as loc:
            report, phases = self.run_fixture(Path(loc).resolve(), terminal=False, original_link='RUNNING')
        self.assertEqual(phases, [])


if __name__ == '__main__':
    unittest.main()
