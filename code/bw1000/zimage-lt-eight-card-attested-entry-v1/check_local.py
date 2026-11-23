from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import node_entry as entry


STEP = "1103606.999"
ROW = f"StepId={STEP} Name={entry.LABEL} NodeList={entry.NODE} State=RUNNING TRES=cpu=12,gres/dcu=8,mem=16G,node=1"
PID = "PID JOBID STEPID LOCALID GLOBALID\n77 1103606 999 0 0\n"


class AttestedEntryTests(unittest.TestCase):
    def test_exact_step_across_multi_line_slurm_response(self):
        other = "StepId=1103606.2104 Name=other NodeList=f13r2n10 State=RUNNING TRES=cpu=8,gres/dcu=8,node=1"
        self.assertEqual(entry.live_step(ROW + "\n" + other, STEP)["StepId"], STEP)
        for text in (other, ROW + "\n" + ROW, ROW.replace("cpu=12", "cpu=4"), ROW.replace("dcu=8", "dcu=1")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                entry.live_step(text, STEP)

    def test_process_evidence_distinguishes_reused_or_neighboring_roles(self):
        self.assertTrue(entry.task_pid(PID, STEP, 77))
        self.assertFalse(entry.task_pid(PID, "1103606.9999", 77))
        self.assertFalse(entry.task_pid(PID, STEP, 7))
        self.assertTrue(entry.step_cgroup("0::/system.slice/slurmstepd.scope/job_1103606/step_999/task_0", STEP))
        self.assertFalse(entry.step_cgroup("0::/job_1103606/step_9999/task_0", STEP))
        self.assertEqual(entry.start_ticks("77 (a ) b) " + " ".join(["1"] * 19 + ["9917"])), 9917)

    def test_host_and_container_handoff_persists_bound_receipts_before_exec(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp).resolve()
            scope = b"{}"
            (p / "scope.json").write_bytes(scope)
            runtime_manifest = b'{"fixture":true}'
            (p / "RUNTIME_FILES.json").write_bytes(runtime_manifest)
            (p / "libzimage_lt_probe.so").write_bytes(b"adopted fixture")
            args = SimpleNamespace(run=p, runtime=p, scope_sha256=sha256(scope).hexdigest(),
                                   entry_sha256=sha256(Path(entry.__file__).read_bytes()).hexdigest())
            identity = {"step": STEP, "node": entry.NODE, "boot_SHA256": "a" * 64, "uid": 1000,
                        "affinity": list(range(12)), "cgroup_SHA256": "b" * 64,
                        "cgroup_names_step": True, "pid": 77, "start_ticks": 9917, "epoch": 1.0}
            with patch.object(entry, "local_identity", return_value=identity), \
                 patch.object(entry, "RUNTIME_SHA", sha256(runtime_manifest).hexdigest()), \
                 patch.object(entry, "BUILD", p), patch.object(entry, "NATIVE_SHA", sha256(b"adopted fixture").hexdigest()), \
                 patch.object(Path, "samefile", return_value=True), \
                 patch.object(entry.subprocess, "check_output", side_effect=(ROW, PID)), \
                 patch.object(entry.os, "execv") as invoke:
                entry.host(args)
                self.assertIn("inside", invoke.call_args.args[1])
                self.assertIn("/usr/local/hyhal:/opt/hyhal:ro", invoke.call_args.args[1])
                receipt = (p / "attested-host-owner.json").read_bytes()
                self.assertEqual(json.loads(receipt)["live_slurm"]["StepId"], STEP)
                args.owner_sha256 = sha256(receipt).hexdigest()
                with patch.object(Path, "read_text", return_value="77 (owner) " + " ".join(["1"] * 19 + ["9917"])):
                    with patch.object(entry, "local_identity", return_value=identity | {"cgroup_SHA256": "c" * 64}):
                        with self.assertRaisesRegex(ValueError, "changed the attested"):
                            entry.inside(args)
                    self.assertFalse((p / "attested-container-owner.json").exists())
                    entry.inside(args)
                    self.assertEqual(invoke.call_args.args[1][2], str(p / "supervisor.py"))
                    self.assertEqual(json.loads((p / "attested-container-owner.json").read_bytes())["host_owner_SHA256"],
                                     args.owner_sha256)


if __name__ == "__main__":
    unittest.main()
