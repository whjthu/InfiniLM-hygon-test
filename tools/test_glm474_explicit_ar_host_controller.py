import json
from pathlib import Path, PurePosixPath
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

import test_read_glm474_fresh_kv_explicit_ar_native_terminal as terminal_fixture


REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / 'code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-explicit-ar-native-host-v1/host_once.py'


class ExplicitARHostTests(unittest.TestCase):
    def setUp(self):
        self.host = runpy.run_path(str(ENTRY))['controller']()
        self.kernel_row = ['1103606.81', self.host['KERNEL_LABEL'], 'COMPLETED', '0:0', '8',
                           self.host['NODE'], 'cpu=4,mem=8192M,node=1']

    def membership(self, active=None):
        slot = dict(state='CPU_STEP_ONLY' if active else 'NO_NUMBERED_STEP',
                    numbered_step_assigned_DCU_bounds=[0, 0])
        return dict(membership_qualified=True, observed_epoch=time.time(), parents={
            '1087656': dict(owned_by_observer=True, active_steps={}),
            '1103606': dict(owned_by_observer=True, active_steps={active: {}} if active else {},
                            coverage_by_group_and_node={'O01': {self.host['NODE']: slot}}),
        })

    def proof(self, claim, *, reconcile):
        self.assertTrue(reconcile)
        self.assertEqual(claim, self.host['KERNEL_HOST_CLAIM'])
        row = json.dumps(self.kernel_row, separators=(',', ':')).encode()
        return dict(request=self.host['KERNEL_REQUEST'], state='COMPILE_VERIFIED', independent_observation=True,
                    actual_GPU_steps=0, row=self.kernel_row, terminal_row_SHA256=self.host['hash_bytes'](row),
                    object_SHA256='a'*64, receipt_SHA256='b'*64, fresh_M_SHA256='c'*64, actual_MD_SHA256='c'*64)

    def test_actual_pinned_bootstrap_enters_three_unit_worker_cli(self):
        self.host['frozen']()
        root = PurePosixPath('/public/home/fixture/project')
        epoch = 1000.0
        command = self.host['command'](root, root/'checkout', root/'run', epoch, 'a'*64, 1740)
        self.assertIn('--gres=none', command)
        self.assertIn('--mem=16384M', command)
        self.assertIn('--time=00:29:00', command)
        self.assertIn('--nodelist=f12r2n13', command)
        self.assertEqual(self.host['PLANNED']['HOST_compile'], 3)
        words = shlex.split(command[-1])
        self.assertTrue(words[words.index('-c')+2].endswith('/run_glm474_fresh_kv_explicit_ar_once.py'))
        self.assertIn(self.host['CLAIM_PIN'], words)
        self.assertIn('2680.0', words)
        bootstrap = 'import os; os.sched_getaffinity=lambda _: {1,2,3,4}\n' + words[words.index('-c')+1]
        output = subprocess.check_output([sys.executable, '-I', '-S', '-B', '-c', bootstrap,
                                          str(REPO/self.host['WORKER']), '--help'], timeout=10)
        self.assertEqual(json.loads(output.splitlines()[0])['cpus'], [1, 2, 3, 4])
        self.assertIn(b'--kernel-reconciliation-sha256', output)

    def test_kernel_still_occupying_node_blocks_before_original_intent(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'explicit'
            with self.assertRaisesRegex(ValueError, 'conflicting work remains active'):
                self.host['launch'](path, Path(root)/'object', Path(root)/'source', reconcile_kernel=self.proof,
                    observe=lambda: self.membership(self.kernel_row[0]), expires=None, accounting=None,
                    runner=None, wait_terminal=None)
            self.assertFalse(path.exists())

    def test_timeout_reconciles_the_same_step_and_refuses_replay(self):
        commands = []
        command = self.host['command']
        remote = PurePosixPath('/public/home/fixture/project')

        def target_command(_root, _checkout, _run, *arguments):
            return command(remote, remote/'source', remote/'run', *arguments)

        self.host['command'] = target_command

        def time_out(argv, **_):
            commands.append(argv)
            raise subprocess.TimeoutExpired(argv, 1)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path, output = root/'explicit', root/'object'
            result = self.host['launch'](path, output, root/'source', reconcile_kernel=self.proof,
                observe=self.membership, expires=lambda: time.time()+2000, accounting=lambda _: None,
                runner=time_out, wait_terminal=lambda *_1, **_2: dict(row=None, terminal_visible=False))
            self.assertTrue(result['client_timeout'])
            self.assertEqual(result['state'], 'UNKNOWN')
            row = ['1103606.82', self.host['LABEL'], 'FAILED', '1:0', '20', self.host['NODE'],
                   'cpu=4,mem=16G,node=1']
            later = self.host['inspect_original'](path, output, lambda _: row, lambda value: value == row)
            self.assertEqual(later['state'], 'TERMINAL_UNVERIFIED')
            self.assertEqual(later['row'], row)
            self.assertEqual(later['intent_SHA256'], result['intent_SHA256'])
            with self.assertRaisesRegex(ValueError, 'namespace exists'):
                self.host['launch'](path, output, root/'source', reconcile_kernel=self.proof,
                    observe=self.membership, expires=None, accounting=None, runner=time_out, wait_terminal=None)
        self.assertEqual(len(commands), 1)

    def test_controller_terminal_requires_the_third_dependency_pair(self):
        fixture = terminal_fixture.ExplicitTerminalReaderTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertEqual(fixture.row[1], self.host['LABEL'])
        fixture.worker['recipe_SHA256'] = fixture.recipe_pin
        fixture.write(fixture.out/'CPU-result.json', fixture.worker)
        log = fixture.directory/'srun.log'
        fixture.write(log, dict(kind=self.host['AFFINITY_KIND'], step=fixture.row[0], cpus=[2, 4, 6, 8],
                                cpus_per_task='4'))
        proof = self.host['verify_terminal'](fixture.out, fixture.row, fixture.epoch, fixture.proof, log)
        self.assertEqual(set(proof['dependency_file_SHA256']), {'u00', 'u15', 'u17'})
        broken = fixture.out/'u00.actual.d'
        broken.write_bytes(broken.read_bytes() + b' /foreign.h')
        with self.assertRaisesRegex(ValueError, 'M/MD paths differ'):
            self.host['verify_terminal'](fixture.out, fixture.row, fixture.epoch, fixture.proof, log)


if __name__ == '__main__':
    unittest.main()
