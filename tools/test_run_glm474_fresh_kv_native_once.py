import json
import unittest

import run_glm474_fresh_kv_native_once as runner


class NativeOnceTest(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(planned=dict(runner.PLANNED), host={name: dict(preprocess=['make-deps', name],
                                                                         compile=['compile', name])
                                                               for name in ('u15', 'u17')},
                           link=['link'], reuse={str(i): {} for i in range(30)})
        self.ledger = dict(attempted=dict(fresh_M=0, HOST_compile=0, links=0),
                           actual=dict.fromkeys(runner.PLANNED, 0), passed=False)
        self.calls = []

    def command(self, name, argv, timeout):
        self.calls.append(name)
        self.assertGreater(timeout, 0)
        self.assertTrue(argv)

    def execute(self, error=None):
        def verify_dependencies(phase):
            self.calls.append('verified-'+phase)
            if phase == error:
                raise ValueError('dependency drift')
            return {'phase': phase}

        def verify_artifacts(recipe):
            self.calls.append('checked-31-objects')
            self.assertEqual(len(recipe['reuse']), 30)
            if error == 'reuse':
                raise ValueError('object drift')

        def verify_link():
            self.calls.append('verified-link')
            if error == 'link':
                raise ValueError('unexpected provider')
            return {'providers': 71}

        runner.phase_runner(self.recipe, self.command, verify_dependencies, verify_artifacts, verify_link,
                            self.ledger, lambda: None)

    def test_both_fresh_graphs_precede_compile_and_both_actual_graphs_precede_link(self):
        self.execute()
        self.assertEqual(self.calls, ['fresh-u15', 'fresh-u17', 'verified-fresh', 'compile-u15', 'compile-u17',
                                      'verified-actual', 'checked-31-objects', 'link', 'verified-link'])
        self.assertEqual(self.ledger['actual'], runner.PLANNED)
        self.assertTrue(self.ledger['passed'])

    def test_fresh_graph_failure_never_compiles(self):
        with self.assertRaisesRegex(ValueError, 'dependency drift'):
            self.execute('fresh')
        self.assertEqual(self.ledger['actual']['HOST_compile'], 0)
        self.assertEqual(self.calls, ['fresh-u15', 'fresh-u17', 'verified-fresh'])

    def test_bad_actual_graph_or_reused_object_blocks_the_link(self):
        for error in ('actual', 'reuse'):
            with self.subTest(error=error), self.assertRaises(ValueError):
                self.execute(error)
            self.assertNotIn('link', self.calls)
            self.assertFalse(self.ledger['passed'])

    def test_bad_link_evidence_preserves_original_attempt_counts(self):
        with self.assertRaisesRegex(ValueError, 'unexpected provider'):
            self.execute('link')
        self.assertEqual(self.ledger['actual'], runner.PLANNED)
        self.assertFalse(self.ledger['passed'])

    def test_frozen_claim_and_exact_cpu_isolation(self):
        raw = runner.CLAIM.read_bytes()
        claim = runner.validate_claim(runner.sha256(raw).hexdigest())
        self.assertEqual(claim, json.loads(raw))
        self.assertTrue(callable(runner.owned_cleanup()))
        env = dict(SLURM_JOB_ID='1103606', SLURM_STEP_ID='444', HIP_VISIBLE_DEVICES='-1',
                   CUDA_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')
        runner.validate_placement(env, runner.NODE, 4)
        for current,node,cpus in ((dict(env, SLURM_STEP_GPUS='0'), runner.NODE, 4), (env,'other',4), (env,runner.NODE,8)):
            with self.subTest(node=node,cpus=cpus), self.assertRaisesRegex(ValueError, 'zero-GRES'):
                runner.validate_placement(current, node, cpus)


if __name__ == '__main__':
    unittest.main()
