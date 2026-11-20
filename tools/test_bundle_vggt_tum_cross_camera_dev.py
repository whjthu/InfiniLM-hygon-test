from pathlib import Path
import tempfile
import unittest
import zipfile

from bind_vggt_tum_dev_rgb import CALIBRATION, sha
from bundle_vggt_tum_cross_camera_dev import MANIFEST, pack, unpack
from join_vggt_tum_cross_camera_dev import join
from score_vggt_tum_cross_camera_dev import canonical_scorer, read_cases
from test_join_vggt_tum_cross_camera_dev import scene


class CanonicalCohortBundleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        groups = {sequence: scene(self.root / sequence, sequence) for sequence in CALIBRATION}
        self.combined = self.root / "combined"
        self.inputs_sha = join(groups, self.combined)["cohort_manifest_SHA256"]
        self.bundle = self.root / "inputs.zip"
        self.out = self.root / "out"

    def test_deterministic_bundle_round_trip_to_original_scorer(self):
        packed = pack(self.combined / MANIFEST, self.inputs_sha, self.bundle)
        again = pack(self.combined / MANIFEST, self.inputs_sha, self.root / "again.zip")
        self.assertEqual(packed["bundle_SHA256"], again["bundle_SHA256"])
        report = unpack(self.bundle, packed["bundle_SHA256"], self.inputs_sha, self.out)
        scorer = canonical_scorer()
        cases, rows = read_cases(self.out / MANIFEST, self.inputs_sha, scorer["relative_pose"])
        self.assertEqual((report["verified_members"], len(cases), len(rows)), (3, 2, 24))

    def test_rejects_extra_unsafe_name_before_writing_any_files(self):
        pack(self.combined / MANIFEST, self.inputs_sha, self.bundle)
        with zipfile.ZipFile(self.bundle, "a") as malicious:
            malicious.writestr("../unbound.txt", b"unexpected member")
        with self.assertRaisesRegex(ValueError, "bundle membership"):
            unpack(self.bundle, sha(self.bundle.read_bytes()), self.inputs_sha, self.out)
        self.assertFalse(self.out.exists())
        self.assertFalse((self.root / "unbound.txt").exists())

    def test_rejects_a_changed_original_payload_before_packing(self):
        image = next(self.combined.glob("*.npy"))
        image.write_bytes(image.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "does not match"):
            pack(self.combined / MANIFEST, self.inputs_sha, self.bundle)
        self.assertFalse(self.bundle.exists())


if __name__ == "__main__":
    unittest.main()
