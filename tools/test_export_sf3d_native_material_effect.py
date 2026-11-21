"""Check the material exporter against the shared artifact verifier using fixture predictions."""

from hashlib import sha256
import io
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from export_sf3d_native_material_effect import emit, position_data, predictions_for_sample, validate_plan
from training_material_mae import vector
from training_sample_artifacts import verifier


class NativeMaterialExportTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.out = self.root / "output"
        self.out.mkdir()
        pixels = io.BytesIO()
        Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(pixels, format="PNG")
        self.image = pixels.getvalue()
        self.points = struct.pack("<6f", 0, 1, 1, 0, 0, 1)
        self.sample = dict(sample_id="sample-0", RGBA=self.input_file("RGBA.png", self.image),
                           queries=self.input_file("points.f32", self.points))
        self.plan = dict(case="T11.material-head", arm="A100", query_points_per_sample=2,
                         dataset_SHA256="d" * 64, provider_sources_SHA256="e" * 64,
                         checkpoint=dict(kind="full-native-checkpoint", SHA256="a" * 64),
                         samples=[self.sample])
        self.calls = 0

    def input_file(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), SHA256=sha256(raw).hexdigest())

    def predict(self, rgba, queries):
        self.assertEqual(rgba, self.image)
        self.assertEqual(queries, self.points)
        self.calls += 1
        return dict(albedo=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], roughness=[0.25], metallic=[0.75])

    def test_exported_native_shape_is_accepted_by_shared_verifier(self):
        report = emit(self.plan, self.out, self.predict, "1" * 64)
        self.assertEqual((report["scored"], report["failed"], self.calls), (1, 0, 1))
        self.assertEqual(report["plan_SHA256"], "1" * 64)
        original = verifier(self.root)(report["records"], "albedo MAE")["sample-0"]
        fields = vector(original, "sample-0", self.sample["RGBA"]["SHA256"],
                        self.sample["queries"]["SHA256"], self.plan["checkpoint"]["SHA256"], 2)
        self.assertEqual(fields["albedo"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def test_changed_private_image_keeps_one_failure_without_calling_predictor(self):
        Path(self.sample["RGBA"]["path"]).write_bytes(bytes(len(self.image)))
        report = emit(self.plan, self.out, self.predict, "1" * 64)
        self.assertEqual((report["planned"], report["scored"], report["failed"], self.calls), (1, 0, 1, 0))
        row = json.loads((self.out / "sample-00000.json").read_bytes())
        self.assertEqual(row["status"], "failed")

    def test_query_offsets_and_finite_control(self):
        with self.assertRaisesRegex(ValueError, "invalid frozen query"):
            position_data(self.input_file("bad.f32", struct.pack("<6f", 0, 1, 1, 0, 0, float("nan"))), 2)
        with self.assertRaisesRegex(ValueError, "artifact outside"):
            position_data(self.sample["queries"], 1)

    def test_incorrect_albedo_shape_cannot_replace_native_query_values(self):
        def wrong_shape(*_):
            return {"albedo": [[0., 0., 0.]], "roughness": [0.], "metallic": [0.]}

        with self.assertRaisesRegex(ValueError, "shape or range"):
            predictions_for_sample(wrong_shape, self.sample, 2, "a" * 64)

    def test_base_checkpoint_role_and_duplicate_sample_gate(self):
        with self.assertRaisesRegex(ValueError, "denominator"):
            validate_plan(self.plan | {"checkpoint": {"SHA256": "a" * 64, "kind": "native-base"}})
        with self.assertRaisesRegex(ValueError, "denominator"):
            validate_plan(self.plan | {"samples": [self.sample, self.sample]})
