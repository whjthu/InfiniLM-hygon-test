import unittest

import numpy as np

from surface_distance import distances, metrics


class TriangleDistanceTest(unittest.TestCase):
    vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    faces = [[0, 1, 2]]

    def test_interior_edge_and_vertex_regions(self):
        points = [[0.25, 0.25, 3], [1.5, 1.5, 0], [-0.5, 0.5, 1], [-3, -4, 0]]
        expected = [3, np.sqrt(0.5), np.sqrt(1.25), 5]
        np.testing.assert_allclose(distances(points, self.vertices, self.faces), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(distances(points, self.vertices, [[0, 2, 1]]), expected, atol=1e-12, rtol=0)

    def test_nearest_face_search_crosses_block_boundaries(self):
        near = np.asarray(self.vertices)
        far = near + [100, 0, 0]
        vertices = np.concatenate((far, near))
        faces = [[0, 1, 2]] * 512 + [[3, 4, 5]]
        answer = distances([[0.25, 0.25, 1]], vertices, faces)
        np.testing.assert_array_equal(answer, [1])

    def test_full_directional_denominators_and_inclusive_threshold(self):
        points = np.tile([0.25, 0.25, 0], (2048, 1))
        fixed = (self.vertices, self.faces, points)
        for shift, expected in ((0., 1.), (0.01, 1.), (0.015, 0.)):
            with self.subTest(shift=shift):
                offset = [0, 0, shift]
                other = (np.asarray(self.vertices) + offset, self.faces, points + offset)
                report = metrics(fixed, other)
                self.assertEqual(report["sampled_surface_F_at_1percent"], expected)
                self.assertAlmostEqual(report["sampled_surface_mean_L1"], shift)
                self.assertEqual((report["predicted_points"], report["reference_points"]), (2048, 2048))
        with self.assertRaisesRegex(ValueError, "denominators"):
            metrics(fixed, (self.vertices, self.faces, points[:-1]))

    def test_invalid_and_degenerate_meshes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            distances([[1, 0, 0]], self.vertices, [[0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "indices"):
            distances([[1, 0, 0]], self.vertices, [[0, 1, 3]])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            distances([[1, 0, np.inf]], self.vertices, self.faces)
