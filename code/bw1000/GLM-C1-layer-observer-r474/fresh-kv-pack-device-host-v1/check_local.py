"""Exercise the real controller with exact staged references and simulated Slurm."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import host_once as H


CHECKOUT = HERE.parents[3]
BASE = 1_795_300_000
BOOT = "1e3ff586-fd62-4b74-bf36-0388e04c761c"
PCI = [f"0000:{183+i:02x}:00.0" for i in range(8)]


class Scenario:
    def __init__(self, run, probe, variant):
        self.run, self.probe, self.variant = run, probe, variant
        self.phase_rows, self.calls = {}, []
        self.waits = 0

    def accounting(self, timeout):
        assert 0 < timeout <= 8
        parent = [H.PARENT, "parent", "RUNNING", "0:0", "6818", H.NODE, "cpu=128,gres/dcu=8"]
        return "".join("|".join(row) + "\n" for row in [parent, *self.phase_rows.values()]).encode()

    def snapshot(self, row):
        return dict(kind="training_qwen_device_snapshot", mode="host_complete", node=H.NODE, step=row[0],
                    pci=PCI, kfd_holder_count=0, full_fd_inventory_performed=True, boot_id=BOOT,
                    monotonic_ns=6_000_000_000)

    def records(self, row):
        raw = (self.run / "intent.json").read_bytes()
        memory = dict(free_bytes=32 * 1024**3, total_bytes=64 * 1024**3)
        yield dict(state="STARTED", request=H.REQUEST, label=H.LABEL, step=row[0], node=H.NODE,
                   intent_SHA256=sha256(raw).hexdigest(), planned=self.probe.PLANNED, native_SHA256=self.probe.NATIVE_SHA,
                   pid=555, boot_id=BOOT, startticks=37761)
        yield dict(state="BOUND", providers={"count": 71}, hip_SHA256=self.probe.HIP_SHA, before=memory,
                   physical_device=dict(pci=PCI[0], visible_devices=1, kfd=dict(pci=PCI[0], gpu_id=8000)))
        for group, ids in self.probe.PLANNED.items():
            for key in ids:
                yield dict(group=group, id=key, state="STARTED", output_SHA256=None)
                if self.variant == "terminal_failure" and key == [17, 128]:
                    yield dict(state="FAIL", type="ValueError", error="corrupt V")
                    return
                hashes = [sha256(raw).hexdigest() for raw in self.probe.reference(*self.probe.fixture(key[0]))] if group == "valid" else None
                yield dict(group=group, id=key, state="PASS", output_SHA256=hashes)
        yield dict(state="PASS", teardown="normal", during=memory, after=memory)

    def runner(self, argv, *, stdin, stdout, stderr, timeout, check):
        phase = next(phase for phase in H.PHASES if f"--job-name={H.label(phase)}" in argv)
        self.calls.append(phase)
        assert f"--jobid={H.PARENT}" in argv and f"--nodelist={H.NODE}" in argv and timeout > 0
        cards, cpus, _ = H.PHASES[phase]
        assert ("--gres=dcu:1" if cards else "--gres=none") in argv
        row = [H.PARENT + "." + str(980+len(self.calls)), H.label(phase), "COMPLETED", "0:0",
               "4", H.NODE, f"cpu={cpus},mem=2G,node=1" + (",gres/dcu=1" if cards else "")]
        if phase == "gpu" and self.variant == "unknown_gpu":
            row[2] = "RUNNING"
        if phase == "gpu" and self.variant in {"terminal_failure", "missing_gpu_trace", "missing_tres"}:
            row[2:4] = ["FAILED", "1:0"]
        if phase == "gpu" and self.variant == "missing_tres":
            row[6] = "(null)"
        self.phase_rows[phase] = row
        if phase == "gpu":
            if self.variant != "missing_gpu_trace":
                (self.run / "probe.jsonl").write_text("".join(json.dumps(item)+"\n" for item in self.records(row)))
        else:
            snapshot = self.snapshot(row)
            if phase == "post":
                owner = json.loads(argv[argv.index("--worker")+1]) if "--worker" in argv else None
                snapshot.update(worker_identity=owner, worker_state="OWNER_ABSENT" if owner else "NO_WORKER_IDENTITY")
                if self.variant == "changed_pci":
                    snapshot["pci"] = ["0000:71:00.0", *PCI[1:]]
            stdout.write((json.dumps(snapshot) + "\n").encode())
        return SimpleNamespace(returncode=int(row[2] != "COMPLETED"))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.probe, self.tres = H.verify_stage(CHECKOUT)[::2]

    def execute(self, variant):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            scenario = Scenario(run, self.probe, variant)

            def waiter(read, _, **kwargs):
                row = read()
                terminal = row is not None and row[2] in {"COMPLETED", "FAILED"}
                return dict(terminal_visible=terminal, row=row if terminal else None, observations=[row])

            result = H.launch(CHECKOUT, run, "a"*40, "b"*64, self.probe, SimpleNamespace(wait_terminal=waiter),
                              self.tres, query=scenario.accounting, expires=lambda: BASE+400,
                              runner=scenario.runner, wall=lambda: BASE, mono=lambda: 17,
                              mount=lambda *_: H.container(PurePosixPath("/incoming/test"), PurePosixPath("/results/test")))
            on_disk = json.loads((run / "outcome.json").read_bytes())
            self.assertEqual(on_disk, json.loads(json.dumps(result)))
            result["calls"] = scenario.calls
            if (run / "intent.json").exists():
                intent = json.loads((run / "intent.json").read_bytes())
                self.assertEqual((intent["epoch"], intent["deadline"], intent["work_deadline"]),
                                 (BASE, BASE+300, BASE+270))
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                H.launch(CHECKOUT, run, "a"*40, "b"*64, self.probe, None, self.tres)
            return result

    def test_complete_numerical_and_independent_post(self):
        result = self.execute("pass")
        self.assertEqual(result["calls"], ["pre", "gpu", "post"], result.get("error"))
        self.assertEqual(result["state"], "GPU_NUMERICAL_PASS_POST_CLEAN")
        self.assertEqual(result["probe"]["passed"], {"valid": 4, "rejected": 36})
        self.assertTrue(result["cleanup_qualified"])

    def test_failed_terminal_preserves_denominator_and_runs_post(self):
        result = self.execute("terminal_failure")
        self.assertEqual(result["calls"], ["pre", "gpu", "post"])
        self.assertEqual(result["state"], "GPU_TERMINAL_NUMERICAL_UNQUALIFIED_POST_CLEAN")
        self.assertEqual(result["probe"]["passed"], {"valid": 1, "rejected": 0})
        self.assertEqual(result["probe"]["worker_failure"]["error"], "corrupt V")
        self.assertFalse(result["numerical_pass"])

    def test_running_keeps_reservation_and_skips_post(self):
        result = self.execute("unknown_gpu")
        self.assertEqual(result["calls"], ["pre", "gpu"])
        self.assertEqual(result["actual_GPU_steps_observed"], 1)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertFalse(result["cleanup_qualified"])

    def test_cleanup_stays_unqualified_without_verified_owner_or_placement(self):
        for variant, machine_clear in (("missing_gpu_trace", True), ("changed_pci", False), ("missing_tres", True)):
            with self.subTest(variant=variant):
                result = self.execute(variant)
                self.assertEqual(result["calls"], ["pre", "gpu", "post"])
                self.assertEqual(result["post_node_clear"], machine_clear)
                self.assertFalse(result["cleanup_qualified"])
                self.assertFalse(result["numerical_pass"])

    def test_duplicate_accounting_rejected(self):
        line = f"{H.PARENT}.999|{H.LABEL}|RUNNING|0:0|4|{H.NODE}|cpu=4,gres/dcu=1\n".encode()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            H.select_rows(line + line)


if __name__ == "__main__":
    unittest.main()
