"""Verify frozen routes and feed real candidate functions with recording streams."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate
import prepare


class Tensor:
    def __init__(self, tag, shape, stride, calls):
        self.tag = tag
        self.shape = shape
        self.stride = lambda: stride
        self.dtype, self.device = "fixture-BF16", "fixture-DCU"
        self.calls = calls

    def data_ptr(self):
        return id(self)

    def untyped_storage(self):
        return self

    def copy_(self, source, *, non_blocking):
        self.calls.append(("copy", self.tag, source.tag, non_blocking))


class Event:
    def __init__(self, label, clock, calls):
        self.label, self.clock, self.calls = label, clock, calls

    def record(self, stream):
        self.calls.append(("record", self.label, stream))

    def elapsed_time(self, other):
        return other.clock - self.clock

    def synchronize(self):
        self.calls.append(("sync", self.label))


class CandidateContract(unittest.TestCase):
    def test_stage_binds_both_exact_actual_layouts(self):
        rendered, plan = prepare.build()
        stock = plan["stock_kernel_by_arm"]
        self.assertTrue(stock["transposed"].startswith("Cijk_Alik_"))
        self.assertTrue(stock["packed"].startswith("Cijk_Ailk_"))
        self.assertIn(b"candidate.witness_operand(torch, guarded, layout)", rendered["baseline.py"])
        self.assertIn(b"size(cell, live=True)", rendered["baseline.py"])
        with tempfile.TemporaryDirectory() as temp:
            staged = Path(temp) / "candidate"
            subprocess.run([sys.executable, "-I", "-S", "-B", str(prepare.HERE / "prepare.py"),
                            "--out", str(staged)], check=True, capture_output=True)
            help_ = subprocess.run([sys.executable, "-I", "-S", "-B", str(staged / "worker.py"),
                                    "--help"], check=True, capture_output=True, text=True)
            self.assertIn("--scope-sha256", help_.stdout)
            self.assertEqual((staged / "baseline.py").read_bytes(), rendered["baseline.py"])

    def test_real_caller_counts_one_copy_then_sixteen_calls(self):
        for mode in candidate.LAYOUTS:
            with self.subTest(mode=mode):
                log = []
                left = [SimpleNamespace(view=Tensor("a" + str(i), (288, 3840), (3840, 1), log))
                        for i in range(2)]
                right = [SimpleNamespace(view=Tensor(which, (3840, 3840), stride, log))
                         for which, stride in candidate.LAYOUTS.items()]
                ev = [Event("start", 0, log), Event("pack", 0.2, log), Event("end", 2.3, log)]

                def mm(a, b):
                    log.append(("mm", a.tag, b.tag))
                    return a.tag

                burst, values, order = candidate.burst(SimpleNamespace(mm=mm), (left, *right, None),
                                                       mode, ev, "same stream", 1)
                self.assertEqual(len(values), 16)
                self.assertEqual(order, [1, 0] * 8)
                self.assertEqual(burst["copies"], int(mode == "packed"))
                self.assertEqual(log[0], ("record", "start", "same stream"))
                self.assertEqual(log[-2:], [("record", "end", "same stream"), ("sync", "end")])
                if mode == "packed":
                    self.assertEqual(log[1], ("copy", "packed", "transposed", False))
                else:
                    self.assertFalse(any(row[0] == "copy" for row in log))
                self.assertEqual([row[2] for row in log if row[0] == "mm"], [mode] * 16)

    def test_wrong_destination_stride_and_alias_refuse_before_copy(self):
        log = []
        source = SimpleNamespace(view=Tensor("src", (3840, 3840), (1, 3840), log))
        wrong = SimpleNamespace(view=Tensor("dst", (3840, 3840), (1, 3840), log))
        with self.assertRaisesRegex(ValueError, "RHS_LAYOUT_OR_ALIAS"):
            candidate.pack(source, wrong)
        with self.assertRaisesRegex(ValueError, "RHS_LAYOUT_OR_ALIAS"):
            candidate.pack(source, source)
        self.assertEqual(log, [])

    def test_complete_pair_denominator_and_full_cost_promotion_gate(self):
        plan = candidate.pairs()
        self.assertFalse(candidate.summarize(plan)["clears_component_wall_gate"])
        for pair in plan:
            pair["status"] = "COMPLETE"
            for layout in pair["order"]:
                packed = layout == "packed"
                burst = {"total_event_ms": 8 if packed else 7, "GEMM_event_ms": 6 if packed else 7,
                         "wall_seconds": 9 if packed else 8, "copies": int(packed)}
                pair["arms"].append({"layout": layout, "bursts": [burst.copy() for _ in range(4)],
                                     "verified_warm_calls": 32, "verified_measured_calls": 64})
        summary = candidate.summarize(plan)
        self.assertGreater(summary["median"]["GEMM_event_ms"], 1)
        self.assertFalse(summary["clears_component_wall_gate"])
        plan[0]["arms"][1]["bursts"][0]["copies"] = 0
        with self.assertRaisesRegex(ValueError, "CALL_OR_PACK_DENOMINATOR"):
            candidate.summarize(plan)


if __name__ == "__main__":
    unittest.main()
