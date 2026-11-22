"""Exercise the actual entry renderer and owned workflow with local fixtures."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bracket_contract as scope
import host_once as host
import worker


class Test(unittest.TestCase):
    def test_exact_closure_and_three_argvs(self):
        first, bundle = host.verified(host.REPO)
        old = first.controller(host.REPO)
        host.bind(first, old)
        old.RUN, old.RESULTS = scope.RUN, scope.RESULTS
        apptainer = first.gpu_command(old, {'path': str(scope.RUN / 'worker-config.json'), 'SHA256': 'f'*64}, 1000)
        self.assertEqual(len(bundle), 13)
        self.assertIn(f'{scope.ORIGINAL}:{scope.ORIGINAL}:ro', apptainer)
        self.assertIn(f'{scope.RUN}:{scope.RUN}:rw', apptainer)
        self.assertIn(str(1000 + scope.CLEANUP), apptainer[-1])
        steps = [old.step_command(phase, apptainer) for phase in scope.PHASES]
        self.assertEqual([step[step.index('-t')+1] for step in steps], ['1', '24', '1'])
        self.assertIn('--gres=dcu:8', steps[1])
        for step in (steps[0], steps[2]):
            self.assertIn('--gres=none', step)

        source = host.REPO / 'code/bw1000/training-minicpm-usgs-exposure25-resume20-hip-v1'
        translator = host.REPO / 'code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/framework/tools/hygon_vlm_recipe.py'
        recipe = scope.import_file('test_original_recipe', source / 'minicpm_domain_recipe.py')
        actual_old = scope.import_file('test_original_worker', host.REPO / 'code/bw1000' / scope.ORIGINAL.name / 'worker.py')
        old_plan = recipe.command_plan(translator, (source / 'original-launch.json').read_bytes(),
                                       str(scope.ORIGINAL), 'a'*64, 'resumed', 29929)
        actual_old.normalized_expected(old_plan, scope.ORIGINAL)
        from contract import render
        for arm in scope.AB:
            plan = scope.render(SimpleNamespace(render=render), old_plan, arm)
            changed = [(old, new) for old, new in zip(old_plan['command'], plan['command'], strict=True) if old != new]
            self.assertEqual(len(changed), 2)
            self.assertIn(str(scope.RUN / arm / 'train'), plan['command'])
            self.assertIn(str(scope.RUN / 'target-worker/native_rng_observer.py'), plan['command'])
            self.assertEqual(plan['environment_overrides']['MINICPM_RNG_TREATMENT'], arm)
            self.assertEqual(plan['environment_overrides']['MASTER_PORT'], str(scope.PORT))
            self.assertEqual(plan['command'][plan['command'].index('--max_steps')+1], '25')

    def test_single_epoch_and_arm_expiry(self):
        previous, _ = host.verified(host.REPO)
        archived = scope.import_file('bracket_test_clock', host.REPO / 'code/bw1000' / scope.ORIGINAL.name / 'worker.py')
        archived.WHOLE, archived.NATIVE = scope.WHOLE, scope.NATIVE
        clock = {'wall': 1010, 'mono': 3}
        epoch = 1000
        budget = archived.Budget(epoch, epoch+scope.WHOLE, epoch+scope.NATIVE,
                                 wall=lambda:clock['wall'], monotonic=lambda:clock['mono'])
        self.assertEqual(budget.left(scope.AB['a1'], reserve=scope.ARM_FLOOR), 270)
        clock['mono'] += 300
        with self.assertRaises(ValueError):
            budget.left(scope.AB['a1'], reserve=scope.ARM_FLOOR)
        self.assertEqual(budget.left(scope.AB['b'], reserve=scope.ARM_FLOOR), 350)
        clock['wall'] = epoch+scope.AB['a2']-179
        with self.assertRaises(ValueError):
            budget.left(scope.AB['a2'], reserve=scope.ARM_FLOOR)
        self.assertGreater(scope.WHOLE - scope.GPU_CLIENT, 90 + 26)

    def test_actual_arm_writes_intent_and_requires_strong_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(strict=True)
            recorded = []

            def save(path, doc):
                path.write_text(json.dumps(doc), encoding='utf-8')

            def enough(offset, reserve=0):
                recorded.append((offset, reserve))
                return 20

            def popen(*_args, **kwargs):
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(kwargs['env']['MINICPM_RNG_TREATMENT'], 'b')
                self.assertEqual(kwargs['env']['MINICPM_USGS_CONFIG_SHA256'], 'd'*64)
                return SimpleNamespace(pid=42, poll=lambda:0, returncode=0, wait=lambda **_:0)

            owned = SimpleNamespace(snapshot=lambda:{42:dict(PID=42, PPID=41, PGID=42, startticks=21, state='S')},
                                    refresh=lambda *_:{}, cleanup=lambda *_args, **_kw:
                                    dict(owned_processes_remaining=0, termination_needed=False))
            fake_port = SimpleNamespace(bind=lambda addr:self.assertEqual(addr, ('127.0.0.1', scope.PORT)))
            with (patch.object(worker.subprocess, 'Popen', side_effect=popen), patch.object(worker.signal, 'signal'),
                  patch.object(worker.socket, 'socket') as sock,
                  patch.object(worker, 'observed_ranks', return_value=list(range(8)))):
                sock.return_value.__enter__.return_value = fake_port
                plan = dict(command=['frozen-swift.py'], environment_overrides=dict(MASTER_PORT=str(scope.PORT), MINICPM_RNG_TREATMENT='b'))
                result = worker.run_arm(root, plan, dict(epoch=100, native_deadline=1360, boot_id='boot'),
                                        dict(path='configs.json', SHA256='d'*64), 'b', SimpleNamespace(left=enough), owned, save)
            self.assertTrue(result['passed'])
            self.assertEqual(result['owners'][42]['startticks'], 21)
            self.assertEqual(recorded[0], (scope.AB['b'], scope.ARM_FLOOR))
            intent = json.loads((root / 'b/native-intent.json').read_bytes())
            self.assertEqual(intent['arm_deadline'], 940)
            self.assertEqual(intent['planned_optimizer_updates'], 0)

    def workflow(self, kind):
        previous, bundle = host.verified(host.REPO)
        old = host.bind(previous, previous.controller(host.REPO))
        bundle.update({'bracket_contract.py': b'fixture', 'worker.py': b'fixture'})
        with tempfile.TemporaryDirectory() as folder:
            old.RUN = Path(folder).resolve(strict=True) / 'new'
            old.original_membership = lambda:None
            old.accounting = lambda _:None
            before = dict(boot_id='boot', pci=list(range(8)), pci_to_kfd=None, master_port=scope.PORT)
            old.read_cpu_inventory = lambda phase, _: (before if phase == 'pre' else
                dict(before, owned_processes_remaining=[], owned_registry_complete=True))
            calls = []

            def execute(phase, command, elapsed, epoch):
                self.assertGreaterEqual(elapsed(), 0)
                self.assertTrue(command)
                calls.append(phase)
                step = scope.PARENT + '.1234'
                if phase == 'gpu' and kind != 'unknown':
                    names = list(scope.AB) if kind == 'pass' else ['a1']
                    arms = []
                    for name in names:
                        (old.RUN / name).mkdir()
                        old.write(old.RUN / name / 'native-intent.json', {})
                        arms.append(dict(arm=name, cleanup_known=True, cleanup=dict(owned_processes_remaining=0)))
                    if kind == 'cleanup_missing':
                        arms.clear()
                    old.write(old.RUN / 'worker-outcome.json', dict(request=scope.REQUEST, epoch=epoch, passed=kind=='pass',
                              arms=arms, planned_forwards=24, planned_optimizer_updates=0))
                    if kind == 'pass':
                        old.write(old.RUN / 'worker-assigned.json', dict(request=scope.REQUEST, epoch=epoch,
                                   actual_step=step, boot_id='boot', visible_PCI_by_index=before['pci']))
                        old.write(old.RUN / 'rng-bracket-comparison.json', dict(actual_step=step, complete=True,
                                   planned_forwards=24, verified_individual_forwards=24, comparable_ranks=8))
                return dict(passed=phase!='gpu' or kind=='pass', row=[step], terminal_visible=kind!='unknown' or phase!='gpu')

            old.execute_step = execute
            with patch.object(previous, 'gpu_command', return_value=['fake-apptainer']):
                result = host.workflow(old, previous, dict(publication='a'*40, stage={}), bundle, 'b'*64)
                with self.assertRaises(ValueError):
                    host.workflow(old, previous, dict(publication='a'*40, stage={}), bundle, 'b'*64)
            return result, calls

    def test_terminal_and_unknown_paths_preserve_denominator(self):
        for kind in ('pass', 'failed', 'cleanup_missing', 'unknown'):
            with self.subTest(kind=kind):
                result, calls = self.workflow(kind)
                self.assertEqual(calls, ['pre','gpu'] if kind=='unknown' else ['pre','gpu','post'])
                self.assertIs(result['complete'], kind=='pass')
                self.assertIs(result['clean_post'], kind in ('pass','failed'))
                self.assertIs(result.get('terminal_outcome_unknown', False), kind=='unknown')
                self.assertEqual(result['planned_forwards'], 24)
                self.assertEqual(result['planned_optimizer_updates'], 0)


if __name__ == '__main__':
    unittest.main()
