import copy
import hashlib
import json
import unittest
from pathlib import Path

from training_framework_identity import manifest_tree, verify_registry_source


class FrameworkIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads(
            (root / "coordination/training/training-case-acceptance-matrix-v1.json").read_bytes()
        )
        cls.candidate = matrix["framework_commit"]
        cls.entry = matrix["entry"]
        cls.registry = matrix["live_registry_evidence"]

        def read(pin):
            raw = (root / pin["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == pin["SHA256"]
            return raw

        cls.manifest = json.loads(read(cls.registry["source_manifest"]))
        cls.observed = json.loads(read(cls.registry))["result"]
        cls.entry_bytes = read(cls.registry["source_entry"])

    def test_original_deployment_tree_and_entry(self):
        actual = verify_registry_source(
            self.candidate, self.entry, self.registry, self.observed, self.manifest, self.entry_bytes
        )
        self.assertEqual(actual["framework_tree"], self.observed["source"]["framework_tree"])
        self.assertEqual(actual["framework_files"], 393)

    def test_matching_registration_cannot_replace_deployed_identity(self):
        with self.assertRaisesRegex(ValueError, "acceptance candidate"):
            verify_registry_source(
                self.registry["recipe_registration_framework_commit"], self.entry, self.registry,
                self.observed, self.manifest, self.entry_bytes,
            )

    def test_adulterated_tree_and_entry_are_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"][self.entry]["mode"] = 0o755 if manifest["files"][self.entry]["mode"] == 0o644 else 0o644
        with self.assertRaisesRegex(ValueError, "acceptance candidate"):
            verify_registry_source(
                self.candidate, self.entry, self.registry, self.observed, manifest, self.entry_bytes
            )
        with self.assertRaisesRegex(ValueError, "entry differs"):
            verify_registry_source(
                self.candidate, self.entry, self.registry, self.observed, self.manifest, self.entry_bytes + b"\n"
            )

    def test_manifest_rejects_invalid_paths_and_file_directory_aliases(self):
        record = self.manifest["files"][self.entry]
        for files in ({"a/../b": record}, {"a//b": record}, {"a": record, "a/b": record},
                      {"a/b": record, "a": record}):
            with self.subTest(files=tuple(files)), self.assertRaises(ValueError):
                manifest_tree(files)


if __name__ == "__main__":
    unittest.main()
