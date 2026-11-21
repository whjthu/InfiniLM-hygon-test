"""Small CPU checks for the proper-rotation and kinematic replay rules."""

import unittest

import numpy as np

from fit_poses import best_rotation, position_fit, replay_fk
from resample_rotations import decimate, decode6, encode6, exp_local
from transport_twist import retwist


class GeometryTest(unittest.TestCase):
    def test_unique_orientation_and_reflection_rejection(self):
        before = np.eye(3)[None]
        half_pi = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        test,rank1,_ = best_rotation(before,before@half_pi.T)
        self.assertEqual(rank1,0)
        np.testing.assert_allclose(test[0],half_pi,atol=1e-12)
        mirror = np.diag([-1.,1,1])
        result,_,_ = best_rotation(before,before@mirror)
        np.testing.assert_allclose(np.linalg.det(result),1,atol=1e-12)
        self.assertFalse(np.allclose(result[0],mirror))

    def test_half_turn_is_a_proper_rotation_and_reaches_the_target_vector(self):
        before = np.array([[[1.,0,0]]])
        turn,rank1,opposites = best_rotation(before,-before)
        self.assertEqual((rank1,opposites),(1,1))
        np.testing.assert_allclose(turn[0]@before[0,0],-before[0,0],atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(turn),1,atol=1e-12)

    def test_missing_intermediate_anchor_does_not_break_fk(self):
        tree = np.array([-1,0,1,0])
        rest = np.array([[0.,0,0],[1,0,0],[2,0,0],[0,0,1]])
        taken = np.array([0,2,3])
        rows = np.array([[[0.,0,0],[0,2,0],[0,0,1]]])
        fit,local,loss,*_ = position_fit(rest,tree,taken,rows)
        self.assertLess(loss[-1], .01*loss[0])
        self.assertTrue(np.all(np.diff(loss)<=0))
        np.testing.assert_allclose(replay_fk(local,rest,tree),fit,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(fit[:,1:]-fit[:,tree[1:]],axis=-1),1,atol=1e-12)
        np.testing.assert_allclose(local[:,2:],np.broadcast_to(np.eye(3),(1,2,3,3)),atol=1e-12)

    def test_unobservable_twist_can_be_removed_without_moving_the_skeleton(self):
        tree = np.array([-1,0,1])
        rest = np.array([[0.,0,0],[1,0,0],[2,0,0]])
        poses = np.broadcast_to(np.eye(3),(3,3,3,3)).copy()
        poses[1,1] = np.diag([1.,-1,-1])
        adjusted,_,_,_ = retwist(poses,rest,tree)
        np.testing.assert_allclose(replay_fk(adjusted,rest,tree),replay_fk(poses,rest,tree),atol=1e-12)
        np.testing.assert_allclose(adjusted,np.broadcast_to(np.eye(3),adjusted.shape),atol=1e-12)

    def test_resampler_cancels_alternating_spin_and_keeps_static_joint(self):
        vector = np.zeros((24,2,3))
        vector[:,0,2] = .25*(-1)**np.arange(len(vector))
        vector[:,1,0] = .4
        matrix = exp_local(vector)
        indexes, smooth, _ = decimate(matrix)
        np.testing.assert_array_equal(indexes,np.arange(0,24,2))
        np.testing.assert_allclose(smooth[:,0],np.broadcast_to(np.eye(3),(12,3,3)),atol=1e-12)
        np.testing.assert_allclose(smooth[:,1],matrix[indexes,1],atol=1e-12)
        np.testing.assert_allclose(decode6(encode6(smooth)),smooth,atol=1e-7)


if __name__ == '__main__':
    unittest.main()
