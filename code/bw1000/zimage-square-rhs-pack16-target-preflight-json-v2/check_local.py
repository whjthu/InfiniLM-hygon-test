"""Exercise the future v2 renderer and real child stdio boundary locally."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics
import render


REQUEST = "b" * 32
STEP = "1103606.4444"


def successor():
    target = "a" * 32
    return {"schema": "bw1000.o01.pack16-preflight-live-successor.v2", "parent": "1103606", "node": "f12r2n10",
            "standby_step": "1103606.4000", "standby_request": target,
            "standby_label": "training80-dcu-standby-" + target[:12] + "-f12r2n10",
            "precheck_request": REQUEST, "state": "RUNNING", "actual_DCU": 8, "counter_growth_DCU": 8,
            "worker_SHA256": "1" * 64, "original_boot_SHA256": "2" * 64, "live_evidence_SHA256": "3" * 64,
            "observed_epoch": render.OLD_OBSERVED + 100}


def source(data):
    raw = json.dumps(data).encode()
    return render.build(raw, sha256(raw).hexdigest())


class JSONDiagnostic(unittest.TestCase):
    def test_explicit_decode_stages_and_no_raw_contents(self):
        raw = b"unreviewed text\n{}"
        for phase in ("standby_before", "standby_after", "runtime_host", "runtime_inside", "container_stdout"):
            with self.subTest(phase=phase):
                with self.assertRaises(diagnostics.PhaseFailure) as caught:
                    diagnostics.decode(phase, raw)
                m = caught.exception.details
                self.assertEqual(m["phase"], phase)
                self.assertEqual(m["original_failure_type"], "JSONDecodeError")
                self.assertEqual(m["json_location"], {"line": 1, "column": 1, "offset": 0})
                self.assertEqual(m["input_SHA256"], sha256(raw).hexdigest())
                self.assertNotIn("unreviewed text", str(m))
        with self.assertRaisesRegex(diagnostics.PhaseFailure, "PRECHECK_JSON_OBJECT_REQUIRED"):
            diagnostics.decode("runtime_host", b"[]")

    def execute(self, directory, out, err=b"", rc=0):
        if type(out) is dict:
            out = json.dumps(out).encode() + b"\n"
        code = f"import sys; sys.stdout.buffer.write({out!r}); sys.stderr.buffer.write({err!r}); sys.exit({rc})"
        return diagnostics.run_container([sys.executable, "-I", "-S", "-B", "-c", code], os.environ.copy(),
                                         directory, REQUEST, STEP)

    def test_actual_child_mixed_stdout_is_rejected_and_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = b'banner\n{"request":"seldom"}\n'
            with self.assertRaises(diagnostics.PhaseFailure) as caught:
                self.execute(path, raw, b"child diagnostic\n")
            m = caught.exception.details
            self.assertEqual(m["phase"], "container_stdout")
            self.assertEqual(m["container_rc"], 0)
            self.assertEqual(m["stdout"]["SHA256"], sha256(raw).hexdigest())
            self.assertEqual((path / "container.stdout.private").read_bytes(), raw)
            self.assertEqual((path / "container.stderr.private").read_bytes(), b"child diagnostic\n")

    def test_actual_nonzero_inner_json_keeps_the_original_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            record = {"request": REQUEST, "step": STEP, "target_GPU_GO": False, "phase": "runtime_inside",
                      "failure_type": "PhaseFailure", "gate": "PRECHECK_JSON_DECODE",
                      "original_failure_type": "JSONDecodeError",
                      "json_location": {"line": 1, "column": 2, "offset": 1}}
            with self.assertRaises(diagnostics.PhaseFailure) as caught:
                self.execute(path, record, b"env chatter\n", 1)
            m = caught.exception.details
            self.assertEqual((m["phase"], m["container_rc"], m["child_failure"]["phase"]),
                             ("container_exec", 1, "runtime_inside"))
            self.assertEqual(m["child_failure"]["original_failure_type"], "JSONDecodeError")
            self.assertEqual(m["stderr"]["bytes"], len(b"env chatter\n"))

    def test_actual_clean_json_even_with_stderr_chatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            result = self.execute(path, {"request": REQUEST, "step": STEP, "target_GPU_GO": False}, b"env noise\n")
            self.assertEqual(result["step"], STEP)
            self.assertEqual(result["container_logs"]["container_rc"], 0)
            self.assertEqual(result["container_logs"]["stderr"]["bytes"], 10)

    def test_render_does_not_keep_old_scope_and_rejects_incomplete_binding(self):
        proposed, evidence = source(successor())
        self.assertEqual(evidence["request"], REQUEST)
        self.assertTrue(evidence["render_does_not_authorize_launch"])
        self.assertIn(b"source /opt/dtk/env.sh >&2", proposed)
        self.assertIn(b"worker_identity(\"standby_before\")", proposed)
        self.assertIn(b"worker_identity(\"standby_after\")", proposed)
        for old in (render.OLD_STEP, render.OLD_REQUEST, render.OLD_STANDBY):
            self.assertNotIn(old.encode(), proposed)
        for diff in ({"standby_step": render.OLD_STEP}, {"state": "COMPLETED"},
                     {"counter_growth_DCU": 7}, {"precheck_request": render.OLD_REQUEST},
                     {"observed_epoch": render.OLD_OBSERVED}):
            with self.assertRaisesRegex(ValueError, "distinct independently observed"):
                source(successor() | diff)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "new.py"
            file.write_bytes(proposed)
            step_help = subprocess.run([sys.executable, "-I", "-S", "-B", str(file), "--help"],
                                       capture_output=True, text=True, check=True)
            self.assertIn("--source-sha256", step_help.stdout)

    @unittest.skipUnless(Path("C:/Program Files/Git/bin/bash.exe").is_file(), "local Git Bash is unavailable")
    def test_env_source_redirection_with_real_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_env = Path(tmp) / "env.sh"
            fake_env.write_bytes(b"printf 'env stdout\\n'\nprintf 'env stderr\\n' >&2\n")
            expected_result = Path(tmp) / "result.json"
            expected_result.write_bytes(b'{"ok":true}\n')
            ran = subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-c",
                                  'set -euo pipefail; source "$1" >&2; cat "$2"',
                                  "fixture", fake_env.as_posix(), expected_result.as_posix()],
                                 capture_output=True, check=True)
            self.assertEqual(ran.stdout.strip(), b'{"ok":true}')
            self.assertEqual(ran.stderr.splitlines(), [b"env stdout", b"env stderr"])


if __name__ == "__main__":
    unittest.main()
