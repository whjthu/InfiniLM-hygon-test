import unittest

import numpy as np

from assemble_motion import pack_201, skin, transform
from check_neck_twist_prior import constant_twist


class GeometryTest(unittest.TestCase):
    def test_pose_contains_rig_rest_origin_and_root_translation_is_separate(self):
        rest = np.array([[1,2,3],[1,3,3]],dtype=float)
        parents = np.array([-1,0])
        pose = np.broadcast_to(np.eye(3),(1,2,3,3))
        world, posed = transform(pose,rest,parents)
        np.testing.assert_array_equal(posed,rest[None])
        vertices = np.array([[1.,1,3],[1,2,3]])
        bones = np.array([[0,1],[1,0]])
        weights = np.full((2,2),.5)
        np.testing.assert_array_equal(skin(world,posed,rest,vertices,weights,bones),vertices[None])
        rotations = np.zeros((1,22,6),dtype=np.float32)
        joints = np.arange(66,dtype=np.float32).reshape(1,22,3)
        offered = np.array([[2,3,4]],dtype=np.float32)
        motion,combined = pack_201(offered,rotations,joints)
        np.testing.assert_array_equal(combined[:,:3],offered)
        np.testing.assert_array_equal(combined[:,:135],motion)
        np.testing.assert_array_equal(combined[:,135:],joints.reshape(1,66))

    def test_constant_twist_leaves_the_only_child_anchor_in_place(self):
        axis = np.array([1.,1,1])/np.sqrt(3)
        rotations = constant_twist(axis,np.array([-180,0,64]))
        np.testing.assert_allclose(rotations@axis,np.broadcast_to(axis,(3,3)),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotations),1,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
