"""Run the v2 probe and HOST boundaries using the exact checked-out sources."""

import ast
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive
import host_once as H
import probe as P


REPO = HERE.parents[3]
OLD_SCENARIO = derive.AREA / "fresh-kv-pack-device-host-v1/check_local.py"
raw = (REPO / OLD_SCENARIO).read_bytes()
if derive.digest(raw) != "e246c107fd23c9c38af2a1619479b0d1c5ddb9ef6bb99d60abed79ca61dac7be":
    raise ValueError("original local Slurm fixture bytes differ")
fixture = {"__file__": str(REPO / OLD_SCENARIO), "__name__": "original_slurm_fixture"}
exec(compile(raw, str(OLD_SCENARIO), "exec"), fixture)
fixture["H"] = H


class Scenario(fixture["Scenario"]):
    def records(self, row):
        for record in super().records(row):
            yield_value = record
            if record["state"] == "BOUND":
                if self.variant == "early_failure":
                    yield dict(state="FAIL", type="ValueError", error="unexpected canonical HIP provider path")
                    return
                providers = {**record["providers"], "canonical_HIP": str(P.HIP_CANONICAL),
                             "canonical_HIP_bytes": P.HIP_BYTES}
                if self.variant == "wrong_bound_path":
                    providers["canonical_HIP"] = str(P.HIP_CANONICAL.parent.parent / "other" / P.HIP_CANONICAL.name)
                yield_value = {**record, "providers": providers}
            yield yield_value

    def runner(self, argv, **kwargs):
        result = super().runner(argv, **kwargs)
        if self.variant == "early_failure" and self.calls[-1] == "gpu":
            self.phase_rows["gpu"][2:4] = ["FAILED", "1:0"]
            return SimpleNamespace(returncode=1)
        return result


class RepairedGate(unittest.TestCase):
    def test_frozen_render_and_unchanged_numerical_matrix(self):
        for name, raw in derive.render(REPO).items():
            self.assertEqual((HERE / name).read_bytes(), raw)
        old, new = [(root / "probe.py").read_bytes() for root in (REPO / derive.OLD, HERE)]

        def bodies(raw):
            preserved = {"verify_providers", "fixture", "reference", "run_matrix", "bad_arguments"}
            return {n.name: ast.dump(n) for n in ast.parse(raw).body
                    if isinstance(n, ast.FunctionDef) and n.name in preserved}

        self.assertEqual(bodies(old), bodies(new))
        self.assertEqual({key: len(items) for key, items in P.PLANNED.items()}, {"valid": 4, "rejected": 36})

    def test_real_observed_canonical_path_and_both_byte_pins(self):
        expected = {P.HIP_LINK: P.HIP_SHA}
        pinned = P.HIP_CANONICAL
        alias = Mock()
        actual = Mock()
        alias.resolve.return_value = actual
        actual.__eq__ = lambda _, comparison: comparison == pinned
        actual.is_file.return_value = True
        actual.stat.return_value = SimpleNamespace(st_size=P.HIP_BYTES)
        with patch.object(P, "Path", return_value=alias) as builder:
            self.assertIs(P.hip_identity(expected), actual)
            builder.assert_called_once_with(P.HIP_LINK)
            alias.resolve.assert_called_once_with(strict=True)
        with self.assertRaisesRegex(ValueError, "SHA pin"):
            P.hip_identity({P.HIP_LINK: "b" * 64})
        with patch.object(P, "Path", return_value=alias):
            actual.stat.return_value.st_size += 1
            with self.assertRaisesRegex(ValueError, "provider size"):
                P.hip_identity(expected)
            alias.resolve.return_value = Path(P.HIP_LINK).with_name(pinned.name)
            with self.assertRaisesRegex(ValueError, "provider path"):
                P.hip_identity(expected)

    def run_host(self, arm):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "new-run"
            probe, _, tres = H.verify_stage(REPO)
            scenario = Scenario(run, probe, arm)
            clock = fixture["BASE"]

            def terminal(read, _, **kwargs):
                row = read()
                return dict(terminal_visible=True, row=row)

            outcome = H.launch(REPO, run, "a"*40, "b"*64, probe, SimpleNamespace(wait_terminal=terminal),
                               tres, query=scenario.accounting, expires=lambda: clock+400, runner=scenario.runner,
                               wall=lambda: clock, mono=lambda: 10,
                               mount=lambda *_: H.container(PurePosixPath("/incoming/v2"), PurePosixPath("/results/v2")))
            self.assertEqual(scenario.calls, ["pre", "gpu", "post"])
            self.assertTrue(outcome["cleanup_qualified"])
            return outcome

    def test_host_uses_result_adapter_before_bound(self):
        outcome = self.run_host("early_failure")
        result = outcome["probe"]
        self.assertEqual(result["binding_state"], "FAILED_BEFORE_BOUND")
        self.assertEqual(result["worker_failure"]["error"], "unexpected canonical HIP provider path")
        self.assertEqual(result["passed"], {"valid": 0, "rejected": 0})
        self.assertNotIn("physical_device", result)
        self.assertNotIn("error", result)
        self.assertFalse(outcome["numerical_pass"])

    def test_full_bound_trace_is_accepted_only_for_original_canonical(self):
        outcome = self.run_host("pass")
        self.assertTrue(outcome["numerical_pass"])
        self.assertEqual(outcome["probe"]["passed"], {"valid": 4, "rejected": 36})
        outcome = self.run_host("wrong_bound_path")
        self.assertFalse(outcome["numerical_pass"])
        self.assertIn("assigned GPU receipt differs", outcome["probe"]["error"])


if __name__ == "__main__":
    unittest.main()
