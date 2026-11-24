"""Check that terminal evidence binds the new numbered step and its independent post check."""

from hashlib import sha256
import sys
from types import ModuleType
import unittest

import render_reader


audit = ModuleType("read_terminal")
exec(compile(render_reader.build(), "<rendered-audit>", "exec"), vars(audit))
sys.modules["read_terminal"] = audit
prior = render_reader.PREVIOUS / "check_local.py"
source = prior.read_bytes()
assert sha256(source).hexdigest() == "003d881c6a266503b1669b05fd31421d808759cc1fe1327cda0d6f103d077bb8"
recording = source.decode().replace('"1103606.2304"', '"1103606.99999"').replace(
    "sdpa-repeatability-comparison.json", "native-mask-repeatability-comparison.json")
fixture = ModuleType("prior_boundaries")
fixture.__file__ = str(prior)
exec(compile(recording, str(prior), "exec"), vars(fixture))


class TerminalTest(fixture.Terminal):
    def test_original_gpu_failure_keeps_verified_cleanup_and_failed_diagnostic(self):
        rows = audit.select_rows(fixture.accounting())
        rows["gpu"][2:4] = ["FAILED", "1:0"]

        def fail(files):
            files["worker-outcome.json"].update(passed=False, error={"type": "RuntimeError"}, not_attempted_arms=["c2", "u2"])

        report = self.run_scope(edit=fail, observed=rows)
        self.assertTrue(report["clean_independent_post"])
        self.assertFalse(report["diagnosis_complete"])
        self.assertEqual(report["not_attempted_arms"], ["c2", "u2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
