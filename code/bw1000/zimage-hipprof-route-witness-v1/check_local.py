"""Check the bounded trace parser against historical schema and bad ownership."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import contract as spec
import hipprof_trace as trace


def event(name, *, owner=41, device=0, begin=110, end=125):
    return dict(cat="HIPOPS", ph="X", args=dict(pid=owner, devId=device,
                BeginNs=str(begin), EndNs=str(end), name=name))


class WitnessTest(unittest.TestCase):
    def parse(self, rows, owner=None):
        if owner is None:
            owner = dict(pid=41, trace_device_index=0, realtime_ns=[100, 140])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"traceEvents": rows}), encoding="utf8")
            return trace.read_trace(path, owner)

    def test_historical_trace_event_schema(self):
        repo = Path(__file__).resolve().parents[3]
        original = json.loads((repo / "evidence/bw1000/GLM507-spawn-trace-qualified-r8692.json").read_bytes())
        row = copy.deepcopy(next(iter(original["kernel_samples"].values())))
        args = row["args"]
        owner = next(o for o in original["owners"] if o["pid"] == int(args["pid"]))
        self.assertEqual(int(args["devId"]), owner["rank"])
        result = self.parse([row], dict(pid=owner["pid"], trace_device_index=owner["rank"],
                                       realtime_ns=owner["window_realtime_ns"]))
        self.assertEqual(result["counts"]["attached"], 1)
        self.assertIsNone(result["unique_contraction"])

    def test_exact_route_and_distinct_or_same_result(self):
        old = spec.ORIGINAL["contiguous"]
        alternate = "hipblaslt_bf16_gemm"
        result = self.parse([event(old), event("hipblaslt_workspace_reset")])
        self.assertEqual(result["unique_contraction"], old)
        routes = {spec.BASELINE: result, spec.ALTERNATIVE: self.parse([event(alternate)])}
        self.assertEqual(trace.assess(routes, "contiguous", spec.ORIGINAL, spec.BASELINE, spec.ALTERNATIVE),
                         "DIFFERENT_KERNEL")
        routes[spec.ALTERNATIVE] = result
        self.assertEqual(trace.assess(routes, "contiguous", spec.ORIGINAL, spec.BASELINE, spec.ALTERNATIVE),
                         "SAME_KERNEL")

    def test_reject_partial_attribution(self):
        old = spec.ORIGINAL["contiguous"]
        for additional in (event(old, owner=8), event(old, device=1), event(old, end=141), event(old)):
            with self.subTest(additional=additional):
                self.assertIsNone(self.parse([event(old), additional])["unique_contraction"])
        self.assertIsNone(self.parse([])["unique_contraction"])


if __name__ == "__main__":
    unittest.main()
