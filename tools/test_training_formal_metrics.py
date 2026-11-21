import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from training_formal_metrics import (
    effect, timing, verify_hunyuan_a100_recorded_updates, verify_hunyuan_dp2_a100_windows,
)
import training_fid


class HunyuanOriginalWindowTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads((root / 'coordination/training/training-case-acceptance-matrix-v1.json').read_bytes())
        self.cases = {case['case']: case for case in matrix['cases']}
        self.reference = json.loads((root / self.cases['T09.sage-grpo']['reference_evidence']).read_bytes())
        self.sage = json.loads((root / 'results/bw1000/2026-11-13/'
                                'training-hunyuanvideo-sage-a100-three-original-step-logs.json').read_bytes())

    def verify(self, case):
        return verify_hunyuan_a100_recorded_updates(self.cases[case], self.reference, self.sage)

    def test_verifies_both_original_three_repeat_medians(self):
        self.assertEqual(self.verify('T09.sft'), .2212546978937898)
        self.assertEqual(self.verify('T09.sage-grpo'), .011123785892607565)

    def test_rejects_sft_resume_as_full_window(self):
        metrics = self.reference['cases']['T09.sft']['runs'][0]['summary']['metrics']
        metrics['completed_global_step_at_start'] = 20
        with self.assertRaisesRegex(ValueError, 'training run from step zero'):
            self.verify('T09.sft')

    def test_rejects_sage_step_sum_substituted_for_walltime(self):
        metrics = self.reference['cases']['T09.sage-grpo']['runs'][0]['summary']['metrics']
        seconds = self.sage['audit']['runs'][0]['sum_measured_step_seconds']
        metrics.update(measurement_wall_seconds=seconds, measurement_updates_per_second=20/seconds)
        with self.assertRaisesRegex(ValueError, 'complete wall clock differs'):
            self.verify('T09.sage-grpo')

    def test_rejects_duplicate_sage_step_log_under_another_repeat(self):
        paths = [path for path in self.sage['pins'] if path.endswith('/steps.ndjson')]
        self.sage['pins'][paths[1]]['sha256'] = self.sage['pins'][paths[0]]['sha256']
        with self.assertRaisesRegex(ValueError, 'step log is unbound or reused'):
            self.verify('T09.sage-grpo')

    def test_rejects_foreign_a100_launch(self):
        self.reference['cases']['T09.sft']['runs'][0]['launch']['sha256'] = self.cases['T09.dp2']['launch_SHA256']
        with self.assertRaisesRegex(ValueError, 'different launch'):
            self.verify('T09.sft')


class Dp2OriginalWindowTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads((root / 'coordination/training/training-case-acceptance-matrix-v1.json').read_bytes())
        self.case = next(case for case in matrix['cases'] if case['case'] == 'T09.dp2')
        self.audit = json.loads((root / self.case['reference_evidence']).read_bytes())
        self.original = json.loads((root / self.audit['reference']['path']).read_bytes())
        results = self.audit['audit']['runs']
        self.baselines = [run for run in results.values() if run['start'] == 0]

    def verify(self):
        return verify_hunyuan_dp2_a100_windows(self.case, self.audit, lambda ref: self.original)

    def test_real_original_three_repeats(self):
        self.assertAlmostEqual(self.verify(), 0.19871607975371872)
        self.assertLess(self.baselines[1]['windows'][0]['seconds'], self.baselines[1]['windows'][1]['seconds'])

    def test_rejects_fast_rank_aggregate(self):
        first, second = self.baselines[1]['windows']
        self.assertLess(first['seconds'], second['seconds'])
        self.baselines[1]['measurement_updates_per_second'] = 20 / first['seconds']
        with self.assertRaisesRegex(ValueError, 'differs from the slowest rank'):
            self.verify()

    def test_rejects_checkpoints_outside_original_timing_boundary(self):
        self.baselines[0]['windows'][0]['boundary'] = 'excludes checkpoint IO'
        with self.assertRaisesRegex(ValueError, 'original boundary'):
            self.verify()

    def test_rejects_short_resume_as_reference_repeat(self):
        audited = self.audit['audit']['runs']
        del audited[self.baselines[1]['run']]
        resumed = next(run for run in audited.values() if run['start'] == 10)
        resumed['start'] = 0
        with self.assertRaisesRegex(ValueError, 'reference run identity'):
            self.verify()

    def test_rejects_copied_timing_summary(self):
        first, second, _ = self.baselines
        path = 'rank-0/summary.json'
        second['original_artifacts'][path]['sha256'] = first['original_artifacts'][path]['sha256']
        with self.assertRaisesRegex(ValueError, 'duplicated a timed summary'):
            self.verify()

    def test_rejects_unproven_matrix_median(self):
        self.case['a100_median_updates_per_second'] = 0.2
        with self.assertRaisesRegex(ValueError, 'matrix median differs'):
            self.verify()


class HoldoutInventoryTest(unittest.TestCase):
    def setUp(self):
        self.attachments = {}
        self.identity = {'case': 'T10.shape-flow-lora', 'dataset_SHA256': '5' * 64}
        self.checkpoints = {platform: digit * 64 for platform, digit in
                            zip(('A100', 'Hygon', 'Hygon_before'), '123')}

    def pin(self, value):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.attachments[digest] = value
        return {'SHA256': digest}

    def verify(self, ids, arms, expected_failure=1):
        metric = 'Chamfer-L1'
        plan = self.pin(dict(self.identity, metric=metric, sample_ids=ids))
        inventory = self.pin(dict(self.identity, metric=metric,
                                  sample_plan_SHA256=plan['SHA256'],
                                  checkpoint_SHA256=self.checkpoints, arms=arms))
        row = {'A100': 10, 'Hygon': 10.4, 'Hygon_before': 12, 'planned_samples': 2,
               'sample_plan': plan, 'sample_inventory': inventory}
        for platform in self.checkpoints:
            row[platform + '_scored'] = 2 - expected_failure
            row[platform + '_failed'] = expected_failure
        row['raw_scores'] = self.pin(dict(row, **self.identity, metric=metric,
                                         checkpoint_SHA256=self.checkpoints))
        row['paired_checkpoint_binding'] = self.pin(dict(self.identity, **self.checkpoints))
        return effect({metric: row}, {metric: 'lower'}, self.identity,
                      lambda item: self.attachments[item['SHA256']])

    def inventory(self):
        records = [
            {'sample_id': 'a', 'status': 'scored',
             'artifact': {'location': 'private/output', 'SHA256': 'a' * 64}},
            {'sample_id': 'b', 'status': 'failed',
             'artifact': {'location': 'private/failure', 'SHA256': 'b' * 64}},
        ]
        return {arm: copy.deepcopy(records) for arm in self.checkpoints}

    def test_accepts_same_ids_and_preserves_failed_sample(self):
        result = self.verify(['a', 'b'], self.inventory())
        self.assertAlmostEqual(result['Chamfer-L1']['relative_degradation'], .04)

    def test_rejects_same_count_with_foreign_id(self):
        arms = self.inventory()
        arms['Hygon'][1]['sample_id'] = 'c'
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            self.verify(['a', 'b'], arms)

    def test_rejects_duplicate_and_missing_ids(self):
        arms = self.inventory()
        arms['Hygon_before'][1]['sample_id'] = 'a'
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            self.verify(['a', 'b'], arms)

    def test_rejects_missing_failure_artifact(self):
        arms = self.inventory()
        arms['Hygon'][1]['artifact']['SHA256'] = ''
        with self.assertRaisesRegex(ValueError, 'output or failure evidence'):
            self.verify(['a', 'b'], arms)

    def test_rejects_aggregate_with_unreported_failure(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            self.verify(['a', 'b'], self.inventory(), expected_failure=0)


class FormalFidTest(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.json = {}
        self.binary = {}
        self.read = []
        self.checkpoints = dict(zip(training_fid.ARMS, [d * 64 for d in '123']))
        spec = self.pin(dict(policy=training_fid.POLICY, layer='pool_3', dimension=2048,
                             implementation_SHA256='a'*64, weights_SHA256='b'*64, preprocess_SHA256='c'*64))
        self.identity = dict(case='T06.sft', dataset_SHA256='d'*64, scorer_SHA256=spec['SHA256'])
        self.ids = ['sample1', 'sample2', 'sample3']
        self.reference_ids = ['real1', 'real2', 'real3', 'real4']
        self.real_images = {name: f'{index+1:064x}' for index, name in enumerate(self.reference_ids)}
        self.planned = self.pin(dict(self.identity, metric='FID', sample_ids=self.ids,
                                     fid_reference_image_SHA256=self.real_images))
        reference = np.zeros((4, 2048), np.float32)
        self.base = np.zeros((3, 2048), np.float32)
        reference[:, 0] = [-1, -1, 1, 1]
        self.base[:, 0] = [-1, 0, 1]
        self.inventory = {}
        arm_features = {}
        self.values = {}
        for index, arm in enumerate(training_fid.ARMS, 1):
            images = {name: f'{index*10+i:064x}' for i, name in enumerate(self.ids)}
            self.inventory[arm] = [dict(sample_id=name, status='scored',
                                        artifact=dict(location=name+'.png', SHA256=images[name])) for name in self.ids]
            data = self.base.copy()
            data[:, 0] += index
            arm_features[arm] = dict(sample_ids=self.ids.copy(), image_SHA256=images,
                                     features=self.pin_array(data))
            self.values[arm] = index**2 + (1 - (4/3)**.5)**2
        self.inputs = dict(self.identity, metric='FID', policy=training_fid.POLICY,
                           sample_plan_SHA256=self.planned['SHA256'], checkpoint_SHA256=self.checkpoints,
                           feature_extractor=spec, reference=dict(sample_ids=self.reference_ids,
                               image_SHA256=self.real_images, features=self.pin_array(reference)),
                           arms=arm_features)

    def pin(self, value):
        sha = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.json[sha] = copy.deepcopy(value)
        return {'SHA256': sha}

    def pin_array(self, value):
        import numpy as np

        out = io.BytesIO()
        np.save(out, value, allow_pickle=False)
        raw = out.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        self.binary[digest] = raw
        return {'SHA256': digest}

    def read_binary(self, item, maximum):
        data = self.binary[item['SHA256']]
        self.assertLessEqual(len(data), maximum)
        self.read.append(item['SHA256'])
        return data

    def verify(self, *, attach=True, failed=False):
        inventory = dict(self.identity, metric='FID', arms=self.inventory,
                         sample_plan_SHA256=self.planned['SHA256'], checkpoint_SHA256=self.checkpoints)
        if attach:
            inventory['fid_inputs'] = self.pin(self.inputs)
        row = dict(self.values, planned_samples=len(self.ids), sample_plan=self.planned,
                   sample_inventory=self.pin(inventory))
        for arm in self.checkpoints:
            failures = int(failed and arm == 'Hygon')
            row.update({arm+'_scored': len(self.ids)-failures, arm+'_failed': failures})
        row['raw_scores'] = self.pin(dict(row, **self.identity, metric='FID', checkpoint_SHA256=self.checkpoints))
        row['paired_checkpoint_binding'] = self.pin(dict(self.identity, **self.checkpoints))
        return effect({'FID': row}, {'FID': 'lower'}, self.identity,
                      lambda item: self.json[item['SHA256']], read_binary=self.read_binary)

    def test_recomputes_compared_fid_with_distinct_real_denominator(self):
        result = self.verify()
        self.assertAlmostEqual(result['FID']['absolute_degradation'], 3.)
        self.assertEqual(len(self.read), 4)

    def test_rejects_digest_only_scores_and_coherent_but_wrong_summary(self):
        with self.assertRaisesRegex(ValueError, 'missing independently readable'):
            self.verify(attach=False)
        self.values['Hygon'] = 1.01
        with self.assertRaisesRegex(ValueError, 'independently recomputed'):
            self.verify()

    def test_rejects_missing_generation_or_changed_feature_order(self):
        self.inventory['Hygon'][1]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'complete pinned generated outputs'):
            self.verify(failed=True)
        self.inventory['Hygon'][1]['status'] = 'scored'
        self.inputs['arms']['Hygon']['sample_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'plan order'):
            self.verify()

    def test_rejects_nonfinite_and_tampered_features(self):
        bad = self.base.copy()
        bad[0, 1] = float('nan')
        self.inputs['arms']['Hygon']['features'] = ref = self.pin_array(bad)
        with self.assertRaisesRegex(ValueError, 'nonfinite FID'):
            self.verify()
        self.binary[ref['SHA256']] = b'tampered'
        with self.assertRaisesRegex(ValueError, 'feature body differs'):
            self.verify()

    def test_factor_reduction_preserves_covariance_for_tall_full_rank_arrays(self):
        import numpy as np

        values = np.random.default_rng(75).normal(size=(9, 3))
        ref = self.pin_array(values)
        with patch.object(training_fid, 'DIMENSION', 3):
            mean, factor = training_fid.fit(self.binary[ref['SHA256']], 9)
        np.testing.assert_allclose(mean, values.mean(axis=0), atol=1e-15)
        np.testing.assert_allclose(factor.T @ factor, np.cov(values, rowvar=False), atol=1e-15)

    def test_fid_matches_independent_dense_covariance_square_root(self):
        import numpy as np

        rng = np.random.default_rng(177)
        left, right = rng.normal(size=(12, 5)), rng.uniform(-2, 2, (9, 5))
        ca, cb = np.cov(left, rowvar=False), np.cov(right, rowvar=False)
        values, vectors = np.linalg.eigh(ca)
        root = (vectors * np.sqrt(values)) @ vectors.T
        middle = root @ cb @ root
        spectral_trace = np.sqrt(np.linalg.eigvalsh(middle)).sum()
        independent = np.square(left.mean(0)-right.mean(0)).sum() + np.trace(ca+cb) - 2*spectral_trace
        with patch.object(training_fid, 'DIMENSION', 5):
            a, b = (training_fid.fit(self.binary[self.pin_array(rows)['SHA256']], len(rows)) for rows in (left, right))
        self.assertAlmostEqual(training_fid.distance(a, b), independent, 12)
        self.assertAlmostEqual(training_fid.distance(b, a), independent, 12)
        self.assertAlmostEqual(training_fid.distance(a, a), 0., 12)


class TimingExecutionTest(unittest.TestCase):
    recipe = 'a' * 64
    boundary = 'b' * 64
    provenance = {'case': 'T09.dp2', 'platform': 'Hygon', 'framework_commit': 'c' * 40}

    def pin(self, row):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        self.attachments[digest] = copy.deepcopy(row)
        return {'SHA256': digest}

    def setUp(self):
        self.attachments = {}
        self.runs = []
        for i in range(3):
            windows = [dict(rank=rank, update_ordinals=list(range(6, 26)), measured_updates=20,
                            seconds=80 + i*10 + rank*10) for rank in range(2)]
            raw = dict(self.provenance, run_id=f'run-{i}', recipe_SHA256=self.recipe, boundary_SHA256=self.boundary,
                       warmup_updates=5, measured_updates=20, profiler=False, completed=True, ranks=windows,
                       admission_seconds=4, load_seconds=20, cache_preparation_seconds=10)
            raw_pin = self.pin(raw)
            terminal = dict(method='slurm-step', handle=f'1103606.{200+i}', state='COMPLETED',
                            exit_code='0:0', elapsed_seconds=150)
            devices = [dict(rank=rank, node='n1', pci=f'0000:1{rank}:00.0', model='test DCU') for rank in range(2)]
            runtime = dict(driver='test-drv', framework='test-runtime', operator_library='test-ops', interconnect='test-link')
            progress = dict(configured_resume=None, start_global_step=0,
                            rank_steps=[dict(rank=rank, global_update_ordinals=list(range(1, 26)))
                                        for rank in range(2)])
            evidence = self.pin(dict(self.provenance, raw_timing_SHA256=raw_pin['SHA256'], terminal=terminal,
                                     devices=devices, runtime=runtime, training_progress=progress))
            self.runs.append(dict(raw, raw_receipt=raw_pin, execution_evidence=evidence))

    def audit(self, *, shared_device_clock=False, return_scope=False):
        return timing(self.runs, 2, self.recipe, self.boundary, lambda item: self.attachments[item['SHA256']],
                      provenance=self.provenance, shared_device_clock=shared_device_clock, return_scope=return_scope)

    def replace_execution(self, index, change):
        run = self.runs[index]
        evidence = copy.deepcopy(self.attachments[run['execution_evidence']['SHA256']])
        change(evidence)
        run['execution_evidence'] = self.pin(evidence)

    def test_successful_terminal_and_physical_rank_bindings(self):
        self.assertAlmostEqual(self.audit(), 20/100)

    def test_rejects_distinct_labels_for_one_execution(self):
        self.replace_execution(1, lambda evidence: evidence['terminal'].update(handle='1103606.200'))
        with self.assertRaisesRegex(ValueError, 'reuse one actual execution'):
            self.audit()

    def test_rejects_incompatible_receipt_terminal_and_device(self):
        for change, reason in (
            (lambda e: e.update(raw_timing_SHA256='1'*64), 'original timing receipt'),
            (lambda e: e['terminal'].update(state='RUNNING'), 'successful terminal execution'),
            (lambda e: e['terminal'].update(method='launcher-process'), 'original Slurm step'),
            (lambda e: e['terminal'].update(elapsed_seconds=30), 'exceeds the actual execution'),
            (lambda e: e['devices'][1].update(node='N1.', pci='00000000:10:00.0'), 'same physical device'),
        ):
            with self.subTest(reason=reason):
                old = self.runs[0]['execution_evidence']
                self.replace_execution(0, change)
                with self.assertRaisesRegex(ValueError, reason):
                    self.audit()
                self.runs[0]['execution_evidence'] = old

    def test_gpu_model_rank_layout_and_runtime_remain_comparable_across_repeats(self):
        original = self.runs[1]['execution_evidence']
        changes = (
            lambda e: e['devices'][1].update(model='other DCU'),
            lambda e: e['devices'][1].update(node='n2'),
            lambda e: e['runtime'].update(interconnect='different link'),
            lambda e: e['runtime'].update(driver='different driver'),
            lambda e: e['runtime'].update(framework='different framework'),
            lambda e: e['runtime'].update(operator_library='different library'),
        )
        for change in changes:
            self.replace_execution(1, change)
            with self.assertRaisesRegex(ValueError, 'repeats differ in hardware, rank placement or runtime'):
                self.audit()
            self.runs[1]['execution_evidence'] = original

    def test_same_hardware_scope_allows_equivalent_nodes_and_pci_addresses(self):
        def move(evidence):
            for rank, device in enumerate(evidence['devices']):
                device.update(node='ReplacementNode.', pci=f'0000:9{rank}:00.0', model=' Test  DCU ')
        self.replace_execution(1, move)
        median, scope = self.audit(return_scope=True)
        self.assertAlmostEqual(median, 20/100)
        self.assertEqual(scope['device_models_by_rank'], ['test dcu', 'test dcu'])
        self.assertEqual(scope['rank_groups_by_node'], [[0, 1]])
        self.assertEqual(scope['runtime']['interconnect'], 'test-link')

    def test_a100_label_requires_a100_model_identity(self):
        archive = (Path(__file__).resolve().parents[1] /
                   'analysis/bw1000/training-vlm-a100-original-120-rank-windows-v1.json')
        model = json.loads(archive.read_bytes())['cases']['T01.sft']['original_hardware']
        self.provenance = dict(self.provenance, platform='A100')
        for i, run in enumerate(self.runs):
            run.update(self.provenance)
            raw = {k: v for k, v in run.items() if k not in ('raw_receipt', 'execution_evidence')}
            run['raw_receipt'] = self.pin(raw)
            def use_archive_label(evidence):
                evidence.update(self.provenance, raw_timing_SHA256=run['raw_receipt']['SHA256'])
                for device in evidence['devices']:
                    device['model'] = model
            self.replace_execution(i, use_archive_label)
        self.assertEqual(self.audit(return_scope=True)[1]['device_models_by_rank'], [model.lower()] * 2)
        self.replace_execution(0, lambda e: e['devices'][0].update(model='NVIDIA H100'))
        with self.assertRaisesRegex(ValueError, 'A100 timing used a different accelerator model'):
            self.audit()

    def test_rejects_a_missing_execution_pin(self):
        del self.runs[0]['execution_evidence']
        with self.assertRaisesRegex(ValueError, 'missing pinned execution evidence'):
            self.audit()

    def test_rejects_original_resume_suffix_despite_relabelled_window(self):
        path = (Path(__file__).resolve().parents[1] / 'results/bw1000/2026-11-13/'
                'training-hunyuanvideo-sft-A100-fresh-process-resume20-terminal.json')
        resumed = json.loads(path.read_bytes())
        self.assertIs(resumed['formal_performance_eligible'], False)
        self.replace_execution(0, lambda evidence: evidence['training_progress'].update(
            configured_resume=resumed['resume_checkpoint_pin']['SHA256'],
            start_global_step=resumed['result']['completed_global_step_at_start'],
            rank_steps=[dict(rank=0, global_update_ordinals=resumed['result']['completed_steps'])]))
        with self.assertRaisesRegex(ValueError, 'fresh run from global step zero'):
            self.audit()

    def test_rejects_nonzero_start_even_when_no_resume_path_is_reported(self):
        self.replace_execution(0, lambda e: e['training_progress'].update(start_global_step=5))
        with self.assertRaisesRegex(ValueError, 'fresh run from global step zero'):
            self.audit()

    def test_rejects_missing_configured_resume_identity(self):
        self.replace_execution(0, lambda e: e['training_progress'].pop('configured_resume'))
        with self.assertRaisesRegex(ValueError, 'fresh run from global step zero'):
            self.audit()

    def test_requires_full_global_update_history_for_every_training_rank(self):
        for wrong_steps in (list(range(6, 26)), [1, *range(3, 26)], [True, *range(2, 26)]):
            with self.subTest(steps=wrong_steps):
                self.replace_execution(0, lambda e: e['training_progress']['rank_steps'][1].update(
                    global_update_ordinals=wrong_steps))
                with self.assertRaisesRegex(ValueError, 'global optimizer updates 1 through 25'):
                    self.audit()

    def test_sage_uses_both_clock_devices_and_only_actor_training_progress(self):
        self.provenance = dict(self.provenance, case='T09.sage-grpo')
        for i, run in enumerate(self.runs):
            ranks = run.pop('ranks')
            run.update(self.provenance)
            run['shared_device_clock'] = dict(seconds=ranks[-1]['seconds'], measured_updates=20,
                                              actor_device=0, vae_device=1, synchronized_before=[0, 1],
                                              synchronized_after=[0, 1], update_ordinals=list(range(6, 26)))
            raw = {key: value for key, value in run.items() if key not in ('raw_receipt', 'execution_evidence')}
            run['raw_receipt'] = self.pin(raw)
            def actor_only(evidence):
                evidence.update(self.provenance, raw_timing_SHA256=run['raw_receipt']['SHA256'])
                evidence['training_progress']['rank_steps'].pop()
            self.replace_execution(i, actor_only)
        self.assertAlmostEqual(self.audit(shared_device_clock=True), 20/100)
        self.replace_execution(1, lambda e: e['training_progress']['rank_steps'].append(
            dict(rank=1, global_update_ordinals=list(range(1, 26)))))
        with self.assertRaisesRegex(ValueError, 'execution progress for training ranks'):
            self.audit(shared_device_clock=True)


if __name__ == '__main__':
    unittest.main()
