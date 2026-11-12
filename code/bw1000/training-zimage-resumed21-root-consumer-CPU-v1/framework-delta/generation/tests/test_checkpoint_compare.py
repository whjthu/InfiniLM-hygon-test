"""Reproduce ambiguous NumPy RNG comparisons and real state differences."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gencuda import hy_resume_checkpoint as common
from gencuda.checkpoint_compare import compare_trees


class NativeCheckpointComparison(unittest.TestCase):
    def test_real_rng_ndarray_roundtrip_and_single_element_change(self):
        a=common.capture_rng([]);b=common.cpu_tree(a)
        self.assertTrue(compare_trees(a,b)['tree_equal'])
        self.assertEqual(compare_trees(a,b)['numpy_array_leaves'],1)
        b['numpy'][1][3]^=np.uint32(1)
        r=compare_trees(a,b);self.assertFalse(r['tree_equal']);self.assertEqual(r['mismatched_leaves'],1)

    def test_bf16_tensor_and_array_shape_dtype_differences(self):
        a=dict(p=torch.tensor([1.,-2.],dtype=torch.bfloat16),rng=np.arange(4,dtype=np.uint32))
        b=common.cpu_tree(a);b['p'][0]+=1
        r=compare_trees(a,b);self.assertFalse(r['tree_equal']);self.assertEqual(r['max_abs'],1.)
        b=common.cpu_tree(a);b['rng']=b['rng'].astype(np.uint64)
        self.assertEqual(compare_trees(a,b)['mismatched_leaves'],1)

    def test_mixed_nested_state_and_scalar_are_not_silently_accepted(self):
        a=dict(state=[(None,True,2),dict(rng=np.arange(8).reshape(2,4))],optimizer={})
        b=common.cpu_tree(a);b['state'][0]=(None,False,2)
        self.assertFalse(compare_trees(a,b)['tree_equal'])
        self.assertTrue(compare_trees(a,common.cpu_tree(a))['tree_equal'])


if __name__=='__main__':unittest.main()
