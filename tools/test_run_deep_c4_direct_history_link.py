import ast
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from run_deep_c4_direct_history_link import NODE, bind_step, deadline_clock, run_once
from verify_deep_c4_direct_history_link import REPO


class RunDirectHistoryLinkTest(unittest.TestCase):
    def setUp(self):
        self.env = dict(SLURM_JOB_ID='1103606', SLURM_STEP_ID='901', SLURM_CPUS_PER_TASK='4',
                        SLURM_GPUS_ON_NODE='0', CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1',
                        ROCR_VISIBLE_DEVICES='-1', TUNING_CPU_REQUEST='a' * 32, TUNING_CPU_PUBLICATION='b' * 40,
                        TUNING_CPU_MANIFEST_SHA='c' * 64, TUNING_CPU_EPOCH='1000', TUNING_CPU_DEADLINE='1300')
        self.binding = bind_step(self.env, NODE, 1001)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'case'
        self.plan = dict(input_SHA256={'baseline-object': '0' * 64}, expected_DT_NEEDED=('candidate',),
                         linker_argv=['linker-fixture', 'candidate'], candidate_receipt_SHA256='1' * 64)

    def test_placement_and_expired_or_gpu_step_rejected(self):
        for patch in (dict(SLURM_JOB_ID='1087656'), dict(SLURM_STEP_ID='487'), dict(SLURM_STEP_GPUS='0'),
                      dict(HIP_VISIBLE_DEVICES='0'), dict(TUNING_CPU_DEADLINE='nan')):
            with self.assertRaises(ValueError):
                bind_step(dict(self.env, **patch), NODE, 1001)
        with self.assertRaisesRegex(ValueError, 'expired deadline'):
            bind_step(self.env, NODE, 1293)

    def test_budget_is_single_epoch_and_monotonic_when_wall_moves_back(self):
        wall, mono = [1220], [500]
        remaining = deadline_clock(self.binding, lambda: wall[0], lambda: mono[0])
        self.assertEqual(remaining(), 73)
        mono[0] += 10
        wall[0] -= 20
        self.assertEqual(remaining(), 63)
        mono[0] += 63
        with self.assertRaises(TimeoutError):
            remaining()

    def test_pass_is_published_only_after_audited_relink_and_cannot_replay(self):
        invocations = []
        def link(argv, seconds):
            self.assertTrue((self.output / 'link-attempt.json').is_file())
            invocations.append((argv, seconds))
            return 0
        audited = dict(self.plan, relinked_probe_passive_verified=True, relinked_probe_SHA256='d' * 64)
        kwargs = (self.output, self.binding, lambda: self.plan, link, lambda: audited, lambda: 100)
        result = run_once(*kwargs)
        self.assertTrue(result['passed'])
        self.assertEqual(invocations, [(self.plan['linker_argv'], 65)])
        self.assertEqual(result['relinked_probe_SHA256'], 'd' * 64)
        with self.assertRaises(FileExistsError):
            run_once(*kwargs)
        self.assertEqual(len(invocations), 1)

    def test_preflight_failure_preserves_zero_attempts(self):
        def missing():
            raise FileNotFoundError('exact original receipt missing')
        result = run_once(self.output, self.binding, missing, None, None, lambda: 100)
        self.assertFalse(result['passed'])
        self.assertEqual(result['linker_attempts'], 0)
        self.assertFalse((self.output / 'link-attempt.json').exists())
        self.assertEqual(result, json.loads((self.output / 'CPU-result.json').read_bytes()))

    def test_timeout_and_post_link_pin_drift_keep_failed_denominators(self):
        for index, (returncode, failed_check) in enumerate(((124, 'linker failed'), (0, 'pins drifted'))):
            output = self.output.with_name(f'failure-{index}')
            audited = dict(self.plan, relinked_probe_passive_verified=True, input_SHA256={})
            result = run_once(output, self.binding, lambda: self.plan, lambda *_: returncode,
                              lambda: audited, lambda: 100)
            self.assertEqual(result['linker_attempts'], 1)
            self.assertIn(failed_check, result['failed_check'])
            self.assertTrue(result['completed'])
            self.assertFalse(result['passed'])
            self.assertFalse(result['GPU_GO'])


class DirectHistoryLinkHostPacketTest(unittest.TestCase):
    package = REPO / 'code/bw1000/Deep-C4-causal-consumer-r483/direct-history-T1-guarded-link-CPU-v1'
    publication = 'b' * 40

    def test_published_packet_and_host_hash_checks_agree_on_all_eight_members(self):
        launch = runpy.run_path(str(self.package / 'launch.py'))
        receiver = runpy.run_path(str(self.package / 'receive.py'))

        def query(*args):
            if args[0] == 'ls-remote':
                return f'{self.publication}\trefs/heads/bw1000-experiments\n'.encode()
            self.assertEqual(args[:2], ('cat-file', 'blob'))
            commit, path = args[2].split(':', 1)
            self.assertEqual(commit, self.publication)
            return (REPO / path).read_bytes()

        body, mf = launch['load_packet'](self.publication, query)
        assigned = ast.parse(body.split(b'\n', 1)[0]).body[0]
        packet = ast.literal_eval(assigned.value)
        files = receiver['decode'](packet)
        self.assertEqual(len(files), 9)
        self.assertEqual(mf, packet['manifest_SHA256'])
        self.assertTrue(files['repo/tools/run_deep_c4_direct_history_link.py'])
        with self.assertRaisesRegex(ValueError, 'wrong frozen original request'):
            receiver['decode'](dict(packet, request='c' * 32))
        bad = dict(packet['files'], **{'CPU_inside.sh': 'MA=='})
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            receiver['decode'](dict(packet, files=bad))

    def test_host_requires_both_actual_upstream_terminal_rows_and_clear_node(self):
        receiver = runpy.run_path(str(self.package / 'receive.py'))
        states = {'1103606.156': 'COMPLETED', '1103606.487': 'COMPLETED'}
        current_steps = ['StepId=1103606.extern']

        def query(*args):
            if args[:3] == ('scontrol', 'show', 'job'):
                return ['JobState=RUNNING UserId=owner(502) NumNodes=40 NodeList=nodes AllocTRES=cpu=2560,gres/dcu=320']
            if args[:3] == ('scontrol', 'show', 'step'):
                return current_steps
            if args[:3] == ('scontrol', 'show', 'hostnames'):
                return [NODE]
            if args[3] in states:
                return [f'{args[3]}|{states[args[3]]}|0:0']
            return []

        receiver['check_slurm'](query, 502)
        states['1103606.156'] = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'upstream target build terminal'):
            receiver['check_slurm'](query, 502)
        current_steps.append('StepId=1103606.902 NodeList=nodes')
        with self.assertRaisesRegex(ValueError, 'active or unknown target step'):
            receiver['check_slurm'](query, 502)


if __name__ == '__main__':
    unittest.main()
