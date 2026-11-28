"""Run the owner control sequence locally with a deferred Slurm fixture."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import owner


REQUEST = "a" * 32
AUTH = dict(request=REQUEST, label="training-llava-clevr-first2-" + REQUEST[:12], auth_sha="b" * 64,
            port=29717, vision_root=str(owner.VISION), epoch=1000, deadline=2400)


def entry(role, result="COMPLETED", exit_code="0:0"):
    allocation = "cpu=64,gres/dcu=8,mem=224000M,node=1" if role == "gpu" else "cpu=2,mem=2G,node=1"
    return ["1103606." + str({"pre": 10, "gpu": 11, "post": 12}[role]), owner.role_label(AUTH, role),
            result, exit_code, "10", owner.NODE, allocation, "2026-11-28T06:00:00", "2026-11-28T06:00:10"]


class OwnerTests(unittest.TestCase):
    def test_real_tres_shapes_and_rejections(self):
        for role in ("pre", "gpu", "post"):
            owner.verified_role(AUTH, role, entry(role))
            argv = owner.step_spec(role, AUTH, Path("/run/path"))
            self.assertTrue({"--threads=1", "--immediate=5", "/usr/local/python3.12/bin/python3"} <= set(argv))
        guest = owner.runtime_command(AUTH, Path("/run/path"), Path("/tmp/a"))
        self.assertTrue(any("exec /usr/bin/python3 " in word for word in guest))
        self.assertFalse(any("/usr/local/python3.12/bin/python3" in word for word in guest))
        self.assertIn("/usr/local/hyhal:/opt/hyhal:ro", guest)
        self.assertNotIn("/opt/hyhal:/opt/hyhal:ro", guest)
        self.assertTrue(owner.tres("cpu=8,gres/dcu=8,mem=16G,node=1", 8, 8, 16384))
        self.assertFalse(owner.tres("cpu=8,gres/dcu=8,mem=16G,node=1,node=1", 8, 8, 16384))
        self.assertFalse(owner.tres("cpu=2,gres/dcu=8,mem=2G,node=1", 0, 2, 2048))
        self.assertFalse(owner.tres("cpu=8,gres/gpu=8,mem=16G,node=1", 8, 8, 16384))
        self.assertEqual(owner.fields("StepId=1103606.10 Name=original TRES=cpu=8,node=1")["TRES"], "cpu=8,node=1")

    def test_exact_positive_and_rejected_release(self):
        holder = ["1103606.4869", "training80-dcu-standby-3896796a9d43-" + owner.NODE, "COMPLETED", "0:0",
                  "1166", owner.NODE, "cpu=8,gres/dcu=8,mem=16G,node=1",
                  "2026-11-28T06:00:00", "2026-11-28T06:19:26"]
        doc = dict(mode="NATURAL", parent=owner.PARENT, node=owner.NODE, accounting=holder, worker_returned=True,
                   independent_physical_clean=True, holder_step=holder[0], holder_label=holder[1],
                   worker_SHA256="b"*64, plan_SHA256="c"*64, independent_post_SHA256="d"*64)
        owner.check_release(doc, holder)
        owner.check_release(dict(doc, mode="ORDERLY_ROOT_HANDOFF", exact_root_handoff_confirmed=True), holder)
        with self.assertRaises(ValueError):
            owner.check_release(doc, [*holder[:2], "RUNNING", *holder[3:]])
        with self.assertRaises(ValueError):
            owner.check_release(dict(doc, worker_returned=False), holder)

    def test_real_launch_gating_and_failed_optimizer_still_post(self):
        for pre_fail in (False, True):
            with self.subTest(pre_fail=pre_fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                phases = []

                def short(role, auth, run):
                    phases.append(role)
                    if pre_fail:
                        raise ValueError("PRE")
                    return entry(role)

                def spawn(argv, **_):
                    phases.append("gpu")
                    self.assertIn("--gres=dcu:8", argv)
                    self.assertTrue((root / "gpu-intent.json").is_file())
                    return SimpleNamespace(wait=lambda **_: None, returncode=1)

                fake_terminal = dict(terminal_visible=True, row=entry("gpu", "FAILED", "1:0"), observations=[])
                with patch.object(owner, "authority", return_value=AUTH), patch.object(owner, "held_lock",
                        side_effect=lambda *args: nullcontext()), patch.object(owner.time, "time", return_value=1010), \
                    patch.object(owner, "exclusive_claim"), patch.object(owner, "clear"), \
                    patch.object(owner, "account", return_value=[]), patch.object(owner, "run_short", side_effect=short), \
                    patch.object(owner, "gpu_visible", return_value=entry("gpu")[0]), \
                    patch.object(owner, "await_role", return_value=fake_terminal), \
                    patch.object(owner.subprocess, "Popen", side_effect=spawn), \
                    patch.object(owner.subprocess, "run", return_value=SimpleNamespace(returncode=2)):
                    self.assertEqual(owner.launch(root/"auth", "b"*64, root/"release", root), 2)
                self.assertEqual(phases, ["pre"] if pre_fail else ["pre", "gpu", "post"])
                result, _ = owner.document(root / "controller-outcome.json")
                self.assertFalse(result["scientific_pass"])
                self.assertEqual(result["release_proven"], not pre_fail)


if __name__ == "__main__":
    unittest.main()
