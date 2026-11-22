"""Read-only source and renderer checks plus local fake Slurm receipts."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contract as scope
import host_once as host


class Check(unittest.TestCase):
    def test_pinned_original_renderer_changed_two_values_and_one_env_value(self):
        repo = host.REPO
        source = repo / 'code/bw1000/training-minicpm-usgs-exposure25-resume20-hip-v1'
        path = repo / 'code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/framework/tools/hygon_vlm_recipe.py'
        recipe = scope.import_file('original_recipe_for_test', source / 'minicpm_domain_recipe.py')
        old = scope.import_file('original_worker_for_test', repo / 'code/bw1000' / scope.ORIGINAL_LABEL / 'worker.py')
        archived = recipe.command_plan(path, (source / 'original-launch.json').read_bytes(),
                                       str(scope.ORIGINAL), '0' * 64, 'resumed', 29929)
        old.normalized_expected(archived, scope.ORIGINAL)
        new = scope.render(archived)
        changes = [(a, b) for a, b in zip(archived['command'], new['command'], strict=True) if a != b]
        self.assertCountEqual(changes, [
            (str(scope.ORIGINAL / 'resumed/train'), str(scope.RUN / 'resumed/train')),
            (str(scope.ORIGINAL / 'src/minicpm_usgs_witness.py'), str(scope.RUN / 'target-worker/native_observer.py')),
        ])
        changed_env = {key for key in archived['environment_overrides']
                       if archived['environment_overrides'][key] != new['environment_overrides'][key]}
        self.assertEqual(changed_env, {'MASTER_PORT'})
        self.assertIn(str(scope.ORIGINAL / scope.CHECKPOINT), new['command'])
        new_plan = dict(archived, command=[*archived['command'], '--max_steps', '1'])
        with self.assertRaises(ValueError):
            scope.render(new_plan)

    def test_exact_sources_and_bounded_argv(self):
        sources = host.verified_sources(host.REPO)
        self.assertEqual(len(sources), 7)
        old = host.bind_new(host.controller(host.REPO))
        old.RUN, old.RESULTS = scope.RUN, scope.ORIGINAL.parent
        epoch = 1_795_000_000
        argv = host.gpu_command(old, dict(path=str(scope.RUN / 'worker-config.json'), SHA256='0' * 64), epoch)
        self.assertIn(f'{scope.ORIGINAL}:{scope.ORIGINAL}:ro', argv)
        self.assertIn(f'{scope.RUN}:{scope.RUN}:rw', argv)
        self.assertIn(str(scope.ORIGINAL / 'framework'), argv[-1])
        self.assertIn(str(scope.RUN / 'target-worker/worker.py'), argv[-1])
        self.assertIn(str(epoch + scope.CLEANUP), argv[-1])
        gpu = old.step_command('gpu', argv)
        self.assertIn('--gres=dcu:8', gpu)
        self.assertIn('--jobid=' + scope.PARENT, gpu)
        self.assertEqual(gpu[gpu.index('-t') + 1], '15')
        for phase in ('pre', 'post'):
            query = old.step_command(phase, ['/bin/true'])
            self.assertIn('--gres=none', query)

    def test_checkpoint_inventory_opens_no_tensor_members(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(strict=True)
            files = root / scope.CHECKPOINT
            files.mkdir(parents=True)
            for name in scope.MEMBERS:
                (files / name).write_bytes(b'x')
            value = dict(arm='continuous', global_step=20, member_count=14,
                         actual_tensor_deserialization_here=False,
                         accumulation_boundary='native witness complete-four-micro-update',
                         files={name: dict(bytes=1, SHA256='0' * 64) for name in scope.MEMBERS})
            raw = json.dumps(value).encode()
            inventory = root / scope.INVENTORY
            inventory.write_bytes(raw)
            pin = dict(path=str(inventory), bytes=len(raw), SHA256=scope.digest(raw))
            real_open = Path.open
            seen = []

            def reject_member_read(path, *args, **kwargs):
                seen.append(path)
                if path != inventory:
                    raise AssertionError('checkpoint member body was read')
                return real_open(path, *args, **kwargs)

            with patch.object(Path, 'open', reject_member_read):
                self.assertEqual(scope.checkpoint(root, pin), value['files'])
            self.assertEqual(seen, [inventory])
            (files / 'scheduler.pt').write_bytes(b'xx')
            with self.assertRaises(ValueError):
                scope.checkpoint(root, pin)

    def workflow(self, state):
        with tempfile.TemporaryDirectory() as td:
            old = host.bind_new(host.controller(host.REPO))
            old.RUN = Path(td).resolve(strict=True) / 'results'
            old.original_membership = lambda: None
            old.accounting = lambda _label: None
            before = dict(boot_id='boot', pci=list(range(8)), pci_to_kfd=None, master_port=scope.PORT)
            old.read_cpu_inventory = lambda phase, _record: (before if phase == 'pre' else
                dict(before, owned_processes_remaining=[], owned_registry_complete=True))
            calls = []

            def fake_execute(phase, argv, elapsed, epoch):
                assert not argv[0] == 'srun'
                calls.append(phase)
                assert elapsed() >= 0
                code = dict(pre='1', gpu='2', post='3')[phase]
                step = scope.PARENT + '.' + code
                row = [step, scope.LABEL + '-' + phase, 'COMPLETED', '0:0', '1', scope.NODE, '']
                if phase == 'gpu' and state == 'pass':
                    old.write(old.RUN / 'worker-assigned.json', dict(request=scope.REQUEST, actual_step=step, epoch=epoch,
                               boot_id=before['boot_id'], visible_PCI_by_index=before['pci']))
                    compared = dict(ranks_planned=8, ranks_compared=8, rows=list(range(8)), micro_forwards_per_rank=1,
                                    backward_calls=0, optimizer_updates=0)
                    old.write(old.RUN / 'worker-outcome.json', dict(request=scope.REQUEST, epoch=epoch, passed=True,
                               arms=[dict(passed=True, comparison=compared)]))
                if phase == 'gpu' and state == 'failed':
                    row[2:4] = ['FAILED', '1:0']
                return dict(passed=(phase != 'gpu' or state == 'pass'), row=row, rc=0,
                            terminal_visible=(phase != 'gpu' or state != 'unknown'))

            old.execute_step = fake_execute
            adoption = dict(publication='a' * 40, stage={}, checkpoint20_inventory={})
            with patch.object(host, 'gpu_command', return_value=['apptainer', 'exec']):
                result = host.workflow(old, adoption, {}, '0' * 64)
                with self.assertRaises(ValueError):
                    host.workflow(old, adoption, {}, '0' * 64)
            return result, calls

    def test_host_positive_and_terminal_failure_run_post(self):
        for state in ('pass', 'failed'):
            with self.subTest(state=state):
                result, calls = self.workflow(state)
                self.assertEqual(calls, ['pre', 'gpu', 'post'])
                self.assertIs(result['complete'], state == 'pass', result['error'])
                self.assertIs(result['clean_post'], True)
                self.assertEqual(result['planned_optimizer_updates'], 0)

    def test_unknown_stops_before_post_and_never_replays(self):
        result, calls = self.workflow('unknown')
        self.assertEqual(calls, ['pre', 'gpu'])
        self.assertIs(result['complete'], False)
        self.assertIs(result['clean_post'], False)
        self.assertEqual(result['planned_ranks'], 8)

    def test_canonical_request_and_actual_adoption_schema(self):
        doc = dict(schema='MINICPM_CHECKPOINT20_FIRST_FORWARD_ADMISSION_V1', request=scope.REQUEST,
                   original_step='1103606.1171', parent=scope.PARENT, node=scope.NODE, port=scope.PORT,
                   root_GO=True, GPU_GO=True, publication='a' * 40, stage_SHA256=scope.STAGE_SHA,
                   planned_optimizer_updates=0, checkpoint20_inventory=dict(
                       path=str(scope.ORIGINAL / scope.INVENTORY), bytes=scope.INVENTORY_BYTES,
                       SHA256=scope.INVENTORY_SHA))
        self.assertEqual(scope.admission(doc), doc['checkpoint20_inventory'])
        for bad in (dict(doc, request=scope.ORIGINAL_REQUEST), dict(doc, GPU_GO=False),
                    dict(doc, planned_optimizer_updates=1), dict(doc, checkpoint20_inventory={})):
            with self.assertRaises((ValueError, KeyError)):
                scope.admission(bad)

    def test_owned_rank_cohort_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(strict=True)
            identity = dict(config_SHA256='a' * 64, epoch=100, deadline=820, actual_step='1103606.2000')
            for rank in range(8):
                folder = root / 'minicpm-usgs-witness' / f'rank-{rank:02d}'
                folder.mkdir(parents=True)
                (folder / 'first-forward-ready.json').write_text(json.dumps(dict(
                    rank=rank, world_size=8, checkpoint_step=20, run=identity, state='FORWARD_RETURNED')))
            records = dict.fromkeys(range(8))
            with patch.dict(sys.modules, compare_first=SimpleNamespace(probe_rows=lambda _: records)):
                self.assertIs(scope.ready(root, *identity.values()), records)
                other_step = dict(identity, actual_step='1103606.2001')
                with self.assertRaises(ValueError):
                    scope.ready(root, *other_step.values())


if __name__ == '__main__':
    unittest.main()
