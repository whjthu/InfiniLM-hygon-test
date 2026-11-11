import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "repository_policy", Path(__file__).resolve().parents[1] / "tools/check_repository.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RepositoryPolicyTests(unittest.TestCase):
    def test_reviewed_usage_and_license_paths_are_explicit(self):
        for name in ("generation/docs/host-controller.md", "vlm/runtime/MMF-LICENSE.txt",
                     "vlm/runtime/TEXTVQA-NOTICE.md"):
            with self.subTest(name=name):
                self.assertEqual(module.check_file(name, b"usage or license\n"), [])
        for name in ("generation/docs/results.md", "generation/docs/reports/result.md",
                     "vlm/runtime/another-license.txt", "vlm/runtime/score.md"):
            with self.subTest(name=name):
                self.assertTrue(module.check_file(name, b"unreviewed file\n"))

    def test_config_and_source_are_allowed(self):
        self.assertEqual(module.check_file("generation/configs/model.yaml", b"batch_size: 4\n"), [])

    def test_results_and_environment_files_are_rejected(self):
        for name in (".env", ".env.production", "vlm/.env", "performance-results.json",
                     "results.csv", "report.md", "generation/configs/results.json",
                     "vlm/runtime/report.md", "generation/tests/results.csv",
                     "vlm/runtime/.env.py", "vlm/scripts/.env.sh",
                     "generation/configs/.env.production.yaml"):
            with self.subTest(name=name):
                self.assertTrue(module.check_file(name, b"example: value\n"))

    def test_lock_metadata_and_regression_tests_remain_allowed(self):
        for name in ("generation/sources.lock.json", "generation/tests/test_model.py",
                     "generation/patches/model.patch", "vlm/tests/test_contract.sh",
                     "generation/scripts/prepare_image_context.py", "THIRD_PARTY_NOTICES.md"):
            with self.subTest(name=name):
                self.assertEqual(module.check_file(name, b"example\n"), [])

    def test_integrated_a100_backends_and_origin_metadata_are_source(self):
        for name in ("generation/src/gencuda/backends/sdxl_training.py",
                     "generation/src/gencuda/data/models/hunyuanvideo.yaml",
                     "generation/hygon-a100-backend-origin.json",
                     "vlm/runtime/lora_dropout_session.py", "vlm/plugins/native_dropout_alignment.py"):
            with self.subTest(name=name):
                self.assertEqual(module.check_file(name, b"source_or_origin = True\n"), [])
        for name in ("generation/src/gencuda/results.json", "generation/src/gencuda/weights.pt",
                     "generation/hygon-a100-results.json"):
            with self.subTest(name=name):
                self.assertTrue(module.check_file(name, b"not source\n"))

    def test_parent_traversal_and_absolute_paths_are_rejected(self):
        for name in ("generation/src/genmlu/../../report.py", "/tools/check.py"):
            with self.subTest(name=name):
                self.assertTrue(module.check_file(name, b"pass\n"))

    def test_nested_report_is_rejected(self):
        self.assertTrue(module.check_file("generation/reports/result.json", b"{}"))

    def test_weights_cannot_be_disguised_as_source(self):
        self.assertTrue(module.check_file("src/data.py", b"weights\0binary"))

    def test_personal_path_is_rejected(self):
        self.assertTrue(module.check_file("train.py", b"path = '/mnt/afs/user_home/whj/x'"))

    def test_private_key_is_rejected(self):
        self.assertTrue(module.check_file("config.txt", b"-----BEGIN OPENSSH PRIVATE KEY-----"))

    def test_review_trailer_is_rejected(self):
        self.assertTrue(module.check_file("README.md", b"Co-authored-by: automation"))


if __name__ == "__main__":
    unittest.main()
