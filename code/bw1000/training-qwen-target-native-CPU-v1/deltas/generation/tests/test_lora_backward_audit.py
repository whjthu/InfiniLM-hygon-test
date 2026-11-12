"""Real Torch/PEFT CPU fixtures for the additive backward observer."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from gencuda.lora_backward_audit import LinearBackwardAudit


class Toy(torch.nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.selected = torch.nn.Linear(4, 8, bias=bias, dtype=torch.bfloat16)
        with torch.no_grad():
            self.selected.weight.fill_(0.25)

    def forward(self, x):
        return self.selected(x)


def operands():
    x = (torch.arange(24).reshape(2, 3, 4) % 7 - 3).to(torch.bfloat16)
    dy = (torch.arange(48).reshape(2, 3, 8) % 5 - 2).to(torch.bfloat16)
    return x, dy


class BackwardAuditTests(unittest.TestCase):
    def test_real_gradient_unchanged_and_CPU_F64_reference(self):
        model, baseline = Toy(), Toy()
        x, dy = operands()
        expected = baseline(x)
        (expected.float() * dy.float()).sum().backward()
        with LinearBackwardAudit(Toy, "selected") as audit:
            found = model(x)
            (found.float() * dy.float()).sum().backward()
            report = audit.report(baseline.selected.weight.grad)
        self.assertTrue(torch.equal(found, expected))
        self.assertTrue(torch.equal(model.selected.weight.grad, baseline.selected.weight.grad))
        self.assertTrue(report["native_vs_local_F64_rounded"]["exact_equal"])
        self.assertTrue(report["native_vs_original_gradient"]["exact_equal"])
        self.assertEqual(report["local_reference_device"], "cpu")

    def test_real_PEFT_zero_B_forward_backward(self):
        from peft import LoraConfig, get_peft_model

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
            def forward(self, x):
                return self.proj(x)

        model = get_peft_model(Base(), LoraConfig(r=4, lora_alpha=4, target_modules=["proj"]))
        model = model.to(dtype=torch.bfloat16)
        name, = [name for name, _ in model.named_modules() if name.endswith("lora_B.default")]
        x, dy = operands()
        with LinearBackwardAudit(type(model), name) as audit:
            (model(x).float() * dy.float()).sum().backward()
            gradient = model.get_submodule(name).weight.grad
            report = audit.report(gradient.detach().cpu())
        self.assertEqual(report["counts"], {"model": 1, "forward": 1, "output_gradient": 1, "weight_gradient": 1})
        self.assertTrue(report["native_vs_original_gradient"]["exact_equal"])

    def test_repeated_model_invocation_rejected_and_hooks_removed(self):
        model = Toy()
        x, _ = operands()
        with self.assertRaisesRegex(ValueError, "repeated model"):
            with LinearBackwardAudit(Toy, "selected"):
                model(x)
                model(x)
        self.assertEqual(list(model(x).shape), [2, 3, 8])

    def test_bias_rejected(self):
        with self.assertRaisesRegex(ValueError, "bias-free"):
            with LinearBackwardAudit(Toy, "selected"):
                Toy(bias=True)(operands()[0])

    def test_input_row_cap(self):
        with self.assertRaisesRegex(ValueError, "topology"):
            with LinearBackwardAudit(Toy, "selected", max_rows=2):
                Toy()(operands()[0])

    def test_capture_byte_cap(self):
        with self.assertRaisesRegex(ValueError, "byte budget"):
            with LinearBackwardAudit(Toy, "selected", max_capture_bytes=1):
                Toy()(operands()[0])

    def test_incomplete_backward_rejected(self):
        with LinearBackwardAudit(Toy, "selected") as audit:
            Toy()(operands()[0])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit.report(torch.zeros(8, 4, dtype=torch.bfloat16))

    def test_reference_schema_rejected(self):
        model = Toy()
        x, dy = operands()
        with LinearBackwardAudit(Toy, "selected") as audit:
            (model(x).float() * dy.float()).sum().backward()
            with self.assertRaisesRegex(ValueError, "schema"):
                audit.report(torch.zeros(8, 4, dtype=torch.float32))

    def test_zero_gradient_has_finite_empty_denominator_result(self):
        model = Toy()
        x, _ = operands()
        with LinearBackwardAudit(Toy, "selected") as audit:
            (model(x).float() * 0).sum().backward()
            report = audit.report(torch.zeros(8, 4, dtype=torch.bfloat16))
        self.assertEqual(report["native_vs_local_F64_rounded"]["relative_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
