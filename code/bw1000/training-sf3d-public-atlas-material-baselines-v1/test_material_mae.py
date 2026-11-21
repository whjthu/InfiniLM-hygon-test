from unittest import TestCase

from material_mae import score


class MaterialMAETest(TestCase):
    def test_rgb_channels_then_equal_object_mean(self):
        targets = {
            "first": dict(albedo=[0, 0, 0], roughness=[0], metallic=[0]),
            "second": dict(albedo=[0, 0, 0], roughness=[0], metallic=[0]),
        }
        predictions = {
            "first": dict(albedo=[1, 0, 0], roughness=[1], metallic=[0]),
            "second": dict(albedo=[0, 0, 0], roughness=[0], metallic=[0.5]),
        }
        result = score(tuple(targets), targets, predictions)
        self.assertTrue(result["complete"])
        self.assertEqual(result["MAE"], dict(albedo=1/6, roughness=0.5, metallic=0.25))

    def test_missing_and_invalid_object_keep_full_denominator(self):
        material = dict(albedo=[0, 0.5, 1], roughness=[0], metallic=[1])
        targets = dict(first=material, second=material, third=material)
        predictions = dict(first=material, third=dict(material, roughness=[float("nan")]))
        result = score(tuple(targets), targets, predictions)
        self.assertEqual((result["planned"], result["scored"], result["failed"]), (3, 1, 2))
        self.assertEqual(result["failures"], dict(second="missing_prediction", third="invalid_material_vector"))
        self.assertIsNone(result["MAE"])

    def test_extra_object_and_corrupt_reference_are_rejected(self):
        material = dict(albedo=[0, 0.5, 1], roughness=[0], metallic=[1])
        with self.assertRaisesRegex(ValueError, "planned object IDs"):
            score(["first"], dict(first=material), dict(second=material))
        with self.assertRaisesRegex(ValueError, "normalized material"):
            score(["first"], dict(first=dict(material, metallic=[1.5])), {})
        with self.assertRaisesRegex(ValueError, "channel count"):
            score(["first"], dict(first=dict(material, albedo=[1])), {})
