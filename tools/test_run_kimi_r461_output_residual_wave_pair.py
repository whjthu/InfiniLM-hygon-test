from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import run_kimi_r461_output_residual_wave_pair as wave


class ResidualPairTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.libraries = {}
        providers = ['qualified provider']
        for name in ('full-tree', 'wave32'):
            path = self.root/(name+'.so')
            path.write_bytes(name.encode())
            self.libraries[name] = dict(path=str(path), SHA256=wave.digest(path), providers=providers,
                                        exports=list(wave.VISIBLE))

    def test_terminal_link_receipt_binds_both_libraries_providers_and_card_count(self):
        receipt = dict(completed=True, passed=True, generated_manifest_SHA256=wave.MANIFEST, step='1103606.900',
                       node='f13r1n01', device_calls=0, weights_read=0, arms=self.libraries)
        accounting = dict(columns=['JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES'],
                          row=['1103606.900', 'COMPLETED', '0:0', '60', 'f13r1n01', 'cpu=4,node=1,mem=8G'])
        self.assertEqual(wave.linked_libraries(receipt, accounting, self.root), self.libraries)
        for index, value in ((0, '1103606.901'), (1, 'RUNNING'), (5, 'cpu=4,node=1,mem=8G,gres/dcu=1')):
            changed = deepcopy(accounting)
            changed['row'][index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                wave.linked_libraries(receipt, changed, self.root)
        self.libraries['wave32']['providers'] = ['different provider']
        with self.assertRaisesRegex(ValueError, 'identities, providers'):
            wave.linked_libraries(receipt, accounting, self.root)

    def test_original_operand_and_postload_contract_is_independent_of_new_DSOs(self):
        assets = wave.REPO/wave.BASE
        claim = json.loads(wave.pinned(assets/'CLAIM.json', wave.PINS['CLAIM.json']))
        weights = json.loads(wave.pinned(assets/'CHECKPOINT_WEIGHTS.json', wave.PINS['CHECKPOINT_WEIGHTS.json']))
        binding = {name: weights[name] for name in ('config_SHA256', 'index_SHA256', 'rms_norm_eps')}
        binding.update(pre_transpose=False, component_SHA256=claim['component']['SHA256'],
                       BF16_files=dict(weights['bindings'], residual=claim['residual_fixture']))
        self.assertEqual(set(wave.trusted_inputs(claim, weights, binding)), {'norm', 'projection', 'residual'})
        altered = deepcopy(binding)
        altered['BF16_files']['norm']['SHA256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'checkpoint operand differs'):
            wave.trusted_inputs(claim, weights, altered)

    def test_both_generated_libraries_call_the_same_frozen_mode2_ABI(self):
        before = wave.helper('run_arm.py')
        created = []
        called = []

        class ExpectedABI:
            def __call__(self, mode, transposed, eps, *args):
                called.append((mode, transposed, eps))
                self.assertions(args)
                args[-4][:] = [1., 2., 3.]
                args[-2][:] = [6 << 30, 60 << 30] * 3
                return 0

            @staticmethod
            def assertions(args):
                if args[7] != 512*7168 or args[-3] != 3 or args[-1] != 6:
                    raise ValueError('ABI geometry differs')

        class Loaded:
            kimi_output_residual_component_r461 = ExpectedABI()
            kimi_output_residual_error_r461 = staticmethod(lambda: b'')

        def load(path, *, mode):
            created.append((Path(path).name, mode))
            return Loaded()

        def load_input():
            return {'pre_transpose': False, 'rms_norm_eps': 1e-6}, dict.fromkeys(before['WORDS'], {}), {}

        before['load_exact'] = lambda *_: (ctypes.c_uint16*1)(1)
        with patch.object(wave, 'LIBS', self.root), patch.object(wave, 'helper', return_value=before):
            for name in ('full-tree', 'wave32'):
                directory = self.root/name
                directory.mkdir()
                record = wave.arm(name, self.libraries[name]['SHA256'], directory, inputs=load_input, load=load)
                self.assertEqual(record['mode'], 2)
                self.assertTrue(record['complete'])
        self.assertEqual(called, [(2, 0, 1e-6)] * 2)
        self.assertEqual([path for path, _ in created], ['full-tree.so', 'wave32.so'])

    def test_three_fixed_pairs_compare_all_saved_output_words(self):
        raw = b'\x80\x3f' * (512*7168)
        seen = []

        def report(path, arm, sha, deadline):
            seen.append(arm)
            (path/'output.bf16').write_bytes(raw)
            return dict(complete=True, arm=arm, mode=2, library_SHA256=sha,
                        original_input_receipt_SHA256=wave.load_original_pin(),
                        seconds={'dispatch_through_sync': 4.0 if arm == 'full-tree' else 3.0},
                        output_bytes=len(raw), output_SHA256=wave.digest(path/'output.bf16'))

        result = wave.children(self.root, self.libraries, time.monotonic()+100, started_arm=report)
        self.assertEqual(seen, list(wave.ORDER))
        self.assertTrue(result['complete'] and result['fixed_input_BF16_output_exact'])
        self.assertEqual([x['words'] for x in result['pairs']], [512*7168]*3)
        self.assertEqual((result['pairs'][1]['full_tree_ordinal'], result['pairs'][1]['wave32_ordinal']), (3, 2))
        self.assertEqual(result['median_dispatch_wave32_over_full_tree'], .75)

    def test_partial_workload_retains_six_arm_and_three_pair_denominators(self):
        def invalid(*_):
            raise TimeoutError('fixture timeout')
        result = wave.children(self.root, self.libraries, time.monotonic()+100, started_arm=invalid)
        self.assertFalse(result['complete'])
        self.assertEqual((result['missing_arms'], result['missing_pairs']), (6, 3))
        self.assertEqual(result['failure']['kind'], 'TimeoutError')

    def test_physical_card_uses_the_original_pci_kfd_probe_schema(self):
        address = '0000:01:00.0'
        pre = dict(actual_CPU='1103606.910', node='f13r1n01', boot_id='original-boot',
                   PCI=[f'0000:{n:02x}:00.0' for n in range(1, 9)], KFD_holders=[], matching_owned_PIDs=[])
        env = dict(SLURM_JOB_ID='1103606', SLURM_STEP_ID='911', SLURM_STEP_GPUS='0', HIP_VISIBLE_DEVICES='0')

        class CUDA:
            @staticmethod
            def cudaGetDeviceCount(pointer):
                pointer._obj.value = 1
                return 0

            @staticmethod
            def cudaDeviceGetPCIBusId(buffer, size, index):
                buffer.value = address.encode()
                return 0

        with patch.dict(os.environ, env), patch.object(wave.socket, 'gethostname', return_value=pre['node']), \
                patch.object(Path, 'read_text', return_value='original-boot'), patch.object(wave.C, 'CDLL', return_value=CUDA()):
            self.assertEqual(wave.physical_gpu(pre)['PCI'], address)
            for value in (dict(pre, actual_CPU='1087656.910'), dict(pre, KFD_holders=['99999']),
                          dict(pre, PCI=pre['PCI'][1:]+['0000:09:00.0'])):
                with self.assertRaises(ValueError):
                    wave.physical_gpu(value)


if __name__ == '__main__':
    unittest.main()
