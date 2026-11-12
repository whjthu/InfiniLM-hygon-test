"""An old diagnostic namespace must fail before any runtime or model subprocess."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'tools/hygon_vlm_recipe.py'
spec = importlib.util.spec_from_file_location('diagnostic_recipe', SOURCE)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class DiagnosticOutputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'mask-config.json'
        self.launch = dict(environment_overrides={'OUTPUT_DIR': '/experiment/current/train'},
                           diagnostic_observer={'observers': [
                               {'callback': 'native_lora_dropout_alignment'},
                               {'callback': 'original_first_updates'}]})

    def environment(self, output='/experiment/current/masks', capture='/experiment/current/capture'):
        raw = json.dumps({'output': output}).encode()
        self.config.write_bytes(raw)
        return dict(NATIVE_LORA_DROPOUT_CONFIG=str(self.config),
                    NATIVE_LORA_DROPOUT_CONFIG_SHA256=hashlib.sha256(raw).hexdigest(),
                    FIRST_UPDATE_CAPTURE=capture)

    def test_current_scope_and_normal_training(self):
        recipe.validate_diagnostic_outputs(self.launch, self.environment())
        recipe.validate_diagnostic_outputs({'environment_overrides': {'OUTPUT_DIR': '/normal/train'}}, {})

    def test_old_scope_is_rejected_by_real_entry_before_runtime_or_model(self):
        environment = self.environment(output='/experiment/previous/masks')
        receipt = self.root / 'entry.jsonl'
        with patch.dict(recipe.os.environ, environment, clear=True), patch.object(
                recipe.subprocess, 'run', side_effect=AssertionError('runtime/model must not start')) as run:
            with self.assertRaisesRegex(ValueError, 'launch scope'):
                recipe.run_swift(self.launch, receipt, dry_run=False)
            run.assert_not_called()
        rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        self.assertEqual([row['stage'] for row in rows], ['planned', 'failed'])

    def test_changed_config_and_foreign_capture_are_rejected(self):
        environment = self.environment()
        self.config.write_text('{"output":"/experiment/previous/masks"}')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            recipe.validate_diagnostic_outputs(self.launch, environment)
        for capture in ('/experiment/previous/capture', '/experiment/current/../previous/capture', ''):
            with self.subTest(capture=capture), self.assertRaisesRegex(ValueError, 'launch scope'):
                recipe.validate_diagnostic_outputs(self.launch, self.environment(capture=capture))

    def test_native_qwen_and_gemma_capture_use_the_same_scope_guard(self):
        plan = dict(environment_overrides={'OUTPUT_DIR': '/experiment/current/train'},
                    diagnostic_observer={'callback': 'native_first_updates'})
        recipe.validate_diagnostic_outputs(plan, {'FIRST_UPDATE_CAPTURE': '/experiment/current/capture'})
        for capture in ('/experiment/previous/capture', '', '/experiment/current/../other/capture'):
            with self.subTest(capture=capture), self.assertRaisesRegex(ValueError, 'launch scope'):
                recipe.validate_diagnostic_outputs(plan, {'FIRST_UPDATE_CAPTURE': capture})


if __name__ == '__main__':
    unittest.main()
