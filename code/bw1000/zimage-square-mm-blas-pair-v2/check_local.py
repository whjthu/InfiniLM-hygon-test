"""Exercise real source provenance, raw route guards, and failed-pair accounting."""

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract as spec
import worker


def event(kind, *, cid=0, name="aten::mm", device=0):
    return SimpleNamespace(device_type=lambda: kind, name=lambda: name, correlation_id=lambda: cid,
                           linked_correlation_id=lambda: cid, device_index=lambda: device,
                           shapes=lambda: [[spec.M, spec.K], [spec.K, spec.N]])


def profiler(events):
    return SimpleNamespace(profiler=SimpleNamespace(kineto_results=SimpleNamespace(events=lambda: events)))


class Check(unittest.TestCase):
    def test_exact_original_receipt(self):
        claim = json.loads((HERE / "CLAIM.json").read_bytes())
        self.assertEqual(claim["request"], worker.REQUEST)
        self.assertRegex(claim["request"], r"^[0-9a-f]{32}$")
        evidence = claim["evidence"]
        raw = (HERE.parents[2] / evidence["public"]).read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), evidence["public_SHA256"])
        observed = json.loads(raw)["actual"]["operand_capture"]["examples"]
        for layout, stride in spec.LAYOUTS.items():
            rows = [row for row in observed if row["MNK"] == [spec.M, spec.N, spec.K]
                    and row["rhs"]["stride"] == list(stride)]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["lhs"]["shape"], [spec.M, spec.K])
            self.assertEqual(row["lhs"]["stride"], [spec.K, 1])
            self.assertEqual(row["rhs"]["shape"], [spec.K, spec.N])
            self.assertEqual((row["lhs"]["dtype"], row["rhs"]["dtype"]), ("torch.bfloat16", "torch.bfloat16"))
            self.assertEqual(row["linked_kernel_multiplicity"], [[spec.ORIGINAL[layout], 1]])
            self.assertEqual(row["host_mm_calls"], 120)

    def test_real_correlation_gate_and_false_positive_cases(self):
        old = spec.ORIGINAL["contiguous"]
        alt = "Cijk_independent_ISA936_other"
        hosts = [event(spec.CPU, cid=1), event(spec.CPU, cid=2)]
        gpu = [event(spec.CUDA, cid=1, name=old), event(spec.CUDA, cid=2, name=old)]
        self.assertEqual(spec.kernels(profiler(hosts + gpu), 2), (old,))
        for extra in [event(spec.CUDA, cid=0, name=old), event(spec.CUDA, cid=1, name=old, device=1)]:
            with self.assertRaisesRegex(ValueError, "UNBOUND"):
                spec.kernels(profiler(hosts + gpu + [extra]), 2)
        with self.assertRaisesRegex(ValueError, "DENOMINATOR"):
            spec.kernels(profiler(hosts + [hosts[0]] + gpu), 2)
        with self.assertRaisesRegex(ValueError, "INCONSISTENT"):
            spec.kernels(profiler(hosts + gpu[:1]), 2)
        self.assertFalse(spec.compare_route("contiguous", (old,), (old,)))
        self.assertTrue(spec.compare_route("contiguous", (old,), (alt,)))
        with self.assertRaisesRegex(ValueError, "RETAINS"):
            spec.compare_route("contiguous", (old,), (old, alt))
        with self.assertRaisesRegex(ValueError, "NO_UNAMBIGUOUS"):
            spec.compare_route("contiguous", (old,), ("unrelated_elementwise",))
        with self.assertRaisesRegex(ValueError, "STOCK_ROUTE_DIFFERS"):
            spec.compare_route("contiguous", (alt,), (old,))

    def run_recorded_routes(self, fallback=False):
        state = SimpleNamespace(choice=spec.BASELINE)
        requested = []

        def preference(name=None):
            if name is not None:
                requested.append(name)
                if name == "default":
                    state.choice = spec.ALTERNATIVE
                elif fallback and name == spec.ALTERNATIVE:
                    state.choice = spec.BASELINE
                else:
                    state.choice = name
            return state.choice

        cuda = SimpleNamespace(device_count=lambda: 1, set_device=lambda _: None,
                               get_device_properties=lambda _: SimpleNamespace(gcnArchName="gfx936:features"))
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="bound HIP"), __version__="bound Torch",
                                set_num_threads=lambda _: None, no_grad=nullcontext, autocast=lambda **_: nullcontext(),
                                backends=SimpleNamespace(cuda=SimpleNamespace(preferred_blas_library=preference,
                                    _BlasBackends={arm: arm for arm in (spec.BASELINE, spec.ALTERNATIVE)})))

        def observe(_torch, rows, **_):
            layout = rows[0]
            if state.choice == spec.BASELINE or layout == "transposed":
                return (spec.ORIGINAL[layout],)
            return ("Cijk_distinct_ISA936_trial",)

        def measure(_torch, rows, ledger, _check):
            self.assertEqual(rows[0], "contiguous")
            latency = 2 if state.choice == spec.BASELINE else 1
            ledger.update(calls=64, bitwise_reference_calls=64, event_ms=latency, timed_wall_seconds=latency)

        result = {"routes": {}, "preference_readback": {}, "pairs": spec.planned_pairs()}
        with patch.multiple(worker, pci=lambda _: "0000:01:00.0", fixed_inputs=lambda _: None,
                            open_layout=lambda _t, _h, stride: [next(k for k, v in spec.LAYOUTS.items() if v == stride)],
                            audit_inputs=lambda *_: None, guarded=lambda *_: None, check_guard=lambda _: None,
                            observe=observe, measure=measure):
            worker.exercise(torch, {"torch_hip": "bound HIP", "torch_version": "bound Torch"}, result,
                            lambda: None, lambda: None)
        self.assertEqual(state.choice, spec.BASELINE)
        self.assertNotIn("default", requested)
        self.assertEqual(result["inherited_effective_preference"], spec.BASELINE)
        self.assertEqual(result["restored_effective_preference"], spec.BASELINE)
        return result

    def test_worker_layout_pairing_and_restore(self):
        result = self.run_recorded_routes()
        self.assertEqual(len(result["pairs"]), 6)
        summary = spec.summarize(result["pairs"])
        self.assertEqual(summary["contiguous"]["event_throughput_median"], 2)
        self.assertIsNone(summary["transposed"]["wall_throughput_median"])
        self.assertEqual(summary["transposed"]["statuses"], {"SAME_VENDOR_KERNEL": 3})
        result["pairs"][0]["arms"][0]["bitwise_reference_calls"] = 63
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_NUMERICAL"):
            spec.summarize(result["pairs"])
        result["pairs"][0]["status"] = "FAILED"
        self.assertIsNone(spec.summarize(result["pairs"])["contiguous"]["event_throughput_median"])


    def test_explicit_alternative_fallback_consumes_no_timed_calls(self):
        result = self.run_recorded_routes(fallback=True)
        for layout, pair in spec.summarize(result["pairs"]).items():
            self.assertEqual(pair["statuses"], {"ALTERNATIVE_PREFERENCE_FELL_BACK": 3})
            self.assertIsNone(pair["wall_throughput_median"])
            self.assertNotIn(spec.ALTERNATIVE, result["routes"][layout])
            self.assertEqual(result["preference_readback"][layout][spec.ALTERNATIVE],
                             {"effective": spec.BASELINE, "requested_enum_observed": False})
        self.assertTrue(all(not pair["arms"] for pair in result["pairs"]))


if __name__ == "__main__":
    unittest.main()
