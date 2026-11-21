"""Check the fixed-view rasterizer's visibility and texture convention."""

from unittest import TestCase

import numpy as np

from render_sf3d_boombox_material_views import project, sample_texture, score, visible_uv


class MaterialViewTest(TestCase):
    def test_nearer_triangle_owns_pixels_and_edges_remain_visible(self):
        triangle = np.array([[1., 1., 0.], [6., 1., 0.], [1., 6., 0.]])
        screen = np.vstack((triangle, triangle + [0., 0., 5.]))
        uv = np.array([[0.1, 0.1]] * 3 + [[0.8, 0.7]] * 3)
        faces = np.array([[3, 4, 5], [2, 1, 0]])
        mask, result = visible_uv(screen, faces, uv, 8)
        self.assertEqual(mask.sum(), 15)
        np.testing.assert_allclose(result[1, 1], uv[3])

    def test_uv_sampling_repeats_and_uses_top_left_origin(self):
        texture = np.array([[10, 20], [30, 40]])
        coords = np.array([[0.25, 0.25], [1.75, 0.25], [0.25, -0.25], [0.75, 0.75]])
        np.testing.assert_array_equal(sample_texture(texture, coords), [10, 20, 30, 40])

    def test_camera_is_fixed_and_rgb_reduction_uses_all_channels(self):
        result = project(np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]), (0., 0., 1.), 12, 1.5)
        np.testing.assert_allclose(result, [[6., 6., 0.], [6., -2., 0.], [14., 6., 0.]])
        values = {"albedo": np.array([[0.0, 0.1, 0.2], [0.6, 0.4, 0.2]]), "metallic": np.array([0.2, 0.8])}
        result = score(values, {"albedo": [0.3, 0.4, 0.2], "metallic": [0.5]})
        self.assertAlmostEqual(result["albedo"], 0.15)
        self.assertAlmostEqual(result["metallic"], 0.3)
