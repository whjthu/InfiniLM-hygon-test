"""Offline regression for the same guards used by the CPU entry."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reload


class Tensor:
    dtype = "float32"
    device = SimpleNamespace(type="cpu")
    shape = (1,)

    def __init__(self, value):
        self.value = value

    def numel(self):
        return 1

    def detach(self):
        return self


class Model:
    peft_config = {"default": {}}
    active_adapters = ["default"]

    def __init__(self, original, tensors):
        self.original = original
        self.tensors = tensors

    def get_base_model(self):
        return self.original

    def named_parameters(self):
        return self.tensors.items()


class ReloadTests(unittest.TestCase):
    def test_cli_uses_a_real_numeric_slurm_step(self):
        args = ["reload.py", "--request", "b" * 32, "--step", "1103606.4797"]
        with patch.object(sys, "argv", args), patch.object(reload, "run") as run, patch("builtins.print"):
            self.assertEqual(reload.main(), 0)
        self.assertEqual(run.call_args.args[:2], ("b" * 32, "1103606.4797"))

        args[-1] = "1103606x4797"
        with patch.object(sys, "argv", args), patch.object(reload, "run") as run, patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                reload.main()
        run.assert_not_called()

    def test_actual_archive(self):
        repo = Path(__file__).resolve().parents[3]
        artifact = repo / "results/bw1000/2026-11-28/specialized-internvl-4794-native-cp3-to-cp25-22-updates-and-51-checkpoint-files.json"
        data = artifact.read_bytes()
        self.assertEqual(reload.sha256(data).hexdigest(), reload.ARCHIVE_SHA)
        value = json.loads(data)
        reload.check_archive(value)
        value["actual_eight_DCU"]["verified_checkpoints"]["25"]["parameter_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ORIGINAL_CP25_PROVENANCE"):
            reload.check_archive(value)

    def test_inventory_requires_both_members_and_native_step(self):
        items = {str(i): {} for i in range(15)}
        items.update({name: {"bytes": 1, "SHA256": "f" * 64} for name in reload.MEMBERS})
        value = {"step": 25, "native_schedule_steps": 25, "files": items}
        self.assertEqual(set(reload.adapter_pins(value)), set(reload.MEMBERS))
        value["step"] = 20
        with self.assertRaisesRegex(ValueError, "EXACT_CP25_NATIVE_INVENTORY"):
            reload.adapter_pins(value)

    def test_live_values_requires_attached_adapter_and_exact_native_names(self):
        original = object()
        key = "base_model.model.test.lora_A.default.weight"
        live = {key: Tensor(1)}
        model = Model(original, live)
        torch = SimpleNamespace(float32="float32", equal=lambda a, b: a.value == b.value)
        saved = {key.replace(".default", ""): Tensor(1)}

        def fingerprint(values):
            self.assertEqual(set(values), {key})
            return dict(SHA256="fixture", tensors=1, floating_elements=1, finite=True)

        def verify(wrapped, expected="fixture"):
            return reload.check_values(torch, fingerprint, wrapped, original, saved, expected, 1, 1)

        self.assertEqual(verify(model)["SHA256"], "fixture")
        with self.assertRaisesRegex(ValueError, "EXACT_ACTIVE_NATIVE_CP25"):
            verify(original)
        with self.assertRaisesRegex(ValueError, "EXACT_ACTIVE_NATIVE_CP25"):
            verify(Model(object(), live))
        with self.assertRaisesRegex(ValueError, "ORIGINAL_4794_RANK25_NAMED_FINGERPRINT"):
            verify(model, "bad")
        live["extra.lora_B.default.weight"] = Tensor(1)
        with self.assertRaisesRegex(ValueError, "EXACT_NATIVE_LORA_NAMES"):
            verify(model)
        del live["extra.lora_B.default.weight"]
        live[key] = Tensor(2)
        with self.assertRaisesRegex(ValueError, "NATIVE_LOADED_VALUE_MISMATCH"):
            verify(model)


if __name__ == "__main__":
    unittest.main()
