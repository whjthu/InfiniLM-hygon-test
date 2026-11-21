"""Verify the native producer delta and source-only entry without model dependencies."""

from argparse import Namespace
import ast
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


HERE = Path(__file__).resolve().parent
ENTRY = 'generation/scripts/prepare_hy_motion_training_cache.py'
PATCHED = HERE/'framework'/ENTRY
REPO = HERE.parents[2]


def original():
    archive = (REPO/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz').read_bytes()
    assert sha256(archive).hexdigest() == '85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'
    with tarfile.open(fileobj=BytesIO(archive), mode='r:gz') as tar:
        original = tar.extractfile('framework/'+ENTRY).read()
    assert sha256(original).hexdigest() == '7a8fceef0d4ba30c6676881d3c7dc59fed4c73b54e292d14b57bcdba1c40a203'
    return original


spec = spec_from_file_location('hy_cache_producer_delta', PATCHED)
producer = module_from_spec(spec)
spec.loader.exec_module(producer)


class CacheProducerTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for folder in ('joints_6d','new_joints','texts'):
            (self.root/folder).mkdir()

    def put(self, sample, rotations, joints):
        np.save(self.root/'joints_6d'/f'{sample}.npy', rotations)
        np.save(self.root/'new_joints'/f'{sample}.npy', joints)
        (self.root/'texts'/f'{sample}.txt').write_text('walk forward\n', encoding='utf8')

    def train(self, *ids):
        (self.root/'train.txt').write_text('\n'.join(ids)+'\n', encoding='utf8')

    def windows(self, **args):
        return producer.iter_selected_samples(Namespace(dataset_root=self.root, samples=1, frames=128,
                                                        min_frames=40, **args), [])

    def test_legacy_selection_still_accepts_60_and_strict_selection_requires_paired_full_128(self):
        self.train('short','long')
        self.put('short', np.ones((60,135)), np.ones((60,22,3)))
        rot = np.arange(130*135,dtype=np.float32).reshape(130,135)
        self.put('long', rot, np.ones((130,22,3)))
        [legacy] = list(self.windows())
        self.assertEqual((legacy['sample_id'],legacy['length']), ('short',60))
        [strict] = list(self.windows(require_full_frames=True))
        self.assertEqual((strict['sample_id'],strict['length']), ('long',128))
        np.testing.assert_array_equal(strict['motion_135'],rot[:128])

    def test_mismatched_body_lengths_wrong_width_and_nonfinite_source_are_rejected(self):
        self.train('input')
        valid = np.ones((128,135),dtype=np.float32)
        invalid = valid.copy()
        invalid[-1,-1] = np.nan
        cases = ((valid,np.ones((127,22,3)), 'paired native source'),
                 (np.ones((128,134)),np.ones((128,22,3)), 'paired native source'),
                 (invalid,np.ones((128,22,3)), 'non-finite native source'))
        for rotations,joints,error in cases:
            with self.subTest(error=error):
                self.put('input', rotations, joints)
                with self.assertRaisesRegex(RuntimeError,error):
                    list(self.windows(require_full_frames=True))

    def test_128_source_only_cli_uses_no_cache_or_model_directories(self):
        self.train('short','full')
        self.put('short', np.ones((127,135)), np.ones((127,22,3)))
        self.put('full', np.ones((128,135)), np.ones((128,22,3)))
        absent = self.root/'unavailable-output'
        cmd = [sys.executable,'-B',str(PATCHED),'--dataset-root',str(self.root),'--weights',str(absent),
               '--output-dir',str(absent),'--frames','128','--samples','1','--device','cuda',
               '--require-full-frames','--check-source-only']
        response = subprocess.run(cmd, capture_output=True, check=True, timeout=15)
        self.assertEqual(json.loads(response.stdout), dict(source_only=True, eligible_samples=1,
            verified_frames_per_sample=128, skipped_samples=1, cache_written=False, model_invocations=0))
        self.assertFalse(absent.exists())

    def test_mlu_and_torch_cuda_seed_distinct_backends(self):
        for hip,backend in ((None,'CUDA'),('available','ROCm')):
            with self.subTest(backend=backend):
                torch = SimpleNamespace(cuda=Mock(), mlu=Mock(), manual_seed=Mock(), version=SimpleNamespace(hip=hip))
                torch.cuda.is_available.return_value = True
                with patch.dict(sys.modules,{'torch_mlu':None}):
                    self.assertEqual(producer.initialize_device(torch,'cuda',7),
                                     dict(device='cuda',backend=backend,dtype='bfloat16'))
                    torch.cuda.manual_seed_all.assert_called_once_with(7)
                    torch.mlu.manual_seed_all.assert_not_called()
                with patch.dict(sys.modules,{'torch_mlu':object()}):
                    self.assertEqual(producer.initialize_device(torch,'mlu',9)['backend'],'MLU')
                    torch.mlu.manual_seed_all.assert_called_once_with(9)
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError,'available torch.cuda'):
            producer.initialize_device(torch,'cuda',7)

    def test_201_features_and_shard_serialization_keep_original_functions(self):
        names = {'assemble','_write_shard','_bind_dataset_manifest','_publish_staged_output'}
        source = original()
        def selected(raw):
            return {func.name: ast.dump(func) for func in ast.parse(raw).body
                    if isinstance(func,ast.FunctionDef) and func.name in names}
        self.assertEqual(selected(source),selected(PATCHED.read_bytes()))
        rot = np.arange(128*135,dtype=np.float32).reshape(128,135)
        joints = np.arange(128*66,dtype=np.float32).reshape(128,22,3)
        full = producer.assemble(rot,joints)
        self.assertEqual(full.shape,(128,201))
        np.testing.assert_array_equal(full[:,:135],rot)
        np.testing.assert_array_equal(full[:,135:],joints.reshape(128,66))


if __name__ == '__main__':
    unittest.main()
