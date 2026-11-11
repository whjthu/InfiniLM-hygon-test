"""Real Torch checks for native mask capture and controlled stochastic replay."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from dropout_capture import capture_dropout, replay_dropout


class DropoutCaptureTest(unittest.TestCase):
    def compare_native(self, dtype, transpose=False, zeros=False, gradients=True):
        source = torch.arange(99, dtype=torch.float32).reshape(9, 11).to(dtype)
        if zeros:
            source.zero_()
        if transpose:
            source = source.t()
        baseline_input = source.detach().requires_grad_(gradients)
        captured_input = source.detach().requires_grad_(gradients)
        replay_input = source.detach().requires_grad_(gradients)
        torch.manual_seed(731)
        baseline = functional.dropout(baseline_input, p=.05, training=True)
        baseline_rng = torch.get_rng_state().clone()
        if gradients:
            baseline.sum().backward()
        torch.manual_seed(731)
        captured, record = capture_dropout(captured_input, .05)
        self.assertTrue(torch.equal(captured, baseline))
        self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
        if gradients:
            captured.sum().backward()
            self.assertTrue(torch.equal(captured_input.grad, baseline_input.grad))
        torch.manual_seed(919)
        functional.dropout(source, p=.05, training=True)
        replay_rng = torch.get_rng_state().clone()
        torch.manual_seed(919)
        replay = replay_dropout(replay_input, record)
        self.assertTrue(torch.equal(replay, baseline))
        self.assertTrue(torch.equal(torch.get_rng_state(), replay_rng))
        if gradients:
            replay.sum().backward()
            self.assertTrue(torch.equal(replay_input.grad, baseline_input.grad))

    def test_native_forward_gradient_and_rng(self):
        for dtype in (torch.float32, torch.bfloat16):
            for transpose in (False, True):
                for zeros in (False, True):
                    with self.subTest(dtype=dtype, transpose=transpose, zeros=zeros):
                        self.compare_native(dtype, transpose, zeros)

    def test_no_grad_and_zero_inputs_still_capture_actual_mask(self):
        self.compare_native(torch.float32, zeros=True, gradients=False)
        value = torch.zeros(1, 247, 1024)
        output, record = capture_dropout(value, .05)
        self.assertFalse(output.requires_grad)
        self.assertTrue(record["mask"].any())
        self.assertTrue((~record["mask"]).any())
        self.assertTrue((output == 0).all())

    def test_serialized_record_uses_only_tensors_and_primitive_metadata(self):
        import io
        _, record = capture_dropout(torch.ones(4, 7), .05)
        stream = io.BytesIO()
        torch.save(record, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(replay_dropout(torch.ones(4, 7), restored),
                                    record["mask"].float() / .95))

    def test_rejects_modified_or_incomplete_mask_before_consuming_rng(self):
        value = torch.ones(4, 7)
        _, record = capture_dropout(value, .05)
        broken = []
        row = copy.deepcopy(record); row.pop("source_operator"); broken.append(row)
        row = copy.deepcopy(record); row["mask"][0, 0].logical_not_(); broken.append(row)
        row = copy.deepcopy(record); row["mask"] = row["mask"].float(); broken.append(row)
        row = copy.deepcopy(record); row["mask"] = row["mask"][:, :2]; broken.append(row)
        row = copy.deepcopy(record); row["source_operator"] = "synthetic-mask"; broken.append(row)
        row = copy.deepcopy(record); row["shape"] = [7, 4]; broken.append(row)
        row = copy.deepcopy(record); row["stride"] = [1, 4]; broken.append(row)
        row = copy.deepcopy(record); row["probability"] = float("nan"); broken.append(row)
        for item in broken:
            before = torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                replay_dropout(value, item)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
