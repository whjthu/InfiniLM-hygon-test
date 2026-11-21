"""Exercise the Kimi finish contract using a real temporary direct reservation."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import link_kimi_r461_output_residual_wave as linker
import prepare_kimi_r461_output_residual_wave_finish as finish
import test_dispatch_kimi_r461_output_residual_wave_once as dispatch_fixture
from tuning_queue import Queue, digest


def raw(document):
    return json.dumps(document).encode()


class KimiScoreFinishTest(unittest.TestCase):
    def setUp(self):
        self.dispatch = dispatch_fixture.KimiOriginalScoreDispatchTest()
        self.dispatch.setUp()
        self.addCleanup(self.dispatch.doCleanups)
        self.dispatch.dispatch(lambda *_: SimpleNamespace(returncode=255))
        self.repo = self.dispatch.repo
        self.database = self.dispatch.database
        self.local = self.dispatch.intent.read_bytes()
        self.target = raw(dict(request=finish.REQUEST, label=finish.LABEL, placement=finish.PLACEMENT,
                               epoch=1795300000.5, deadline=1795300300.5, claim_SHA256=finish.CLAIM["SHA256"],
                               source_manifest_SHA256=finish.RENDERED_SHA, replay_allowed=False))
        self.row = ["1103606.712", finish.LABEL, "COMPLETED", "0:0", "83", finish.NODE, "cpu=4,mem=8G,node=1"]
        self.post_step = "1103606.713"
        self.post_batch = "b" * 32
        self.post_accounting = dict(columns=finish.COLUMNS, row=[
            self.post_step, "training80-admit-" + self.post_batch, "COMPLETED", "0:0", "1", finish.NODE,
            "cpu=1,mem=256M,node=1",
        ])
        self.snapshot = self.dispatch.fixture.verified_snapshot()
        self.snapshot["nodes"][finish.NODE].update(device_probe_step=self.post_step, kfd_process_count=0,
                                                  own_device_fd_process_count=0)
        self.snapshot["probes"] = {finish.PARENT: dict(batches=[dict(
            requested_nodes=[finish.NODE], received_nodes=[finish.NODE], state="RETURNED", request_id=self.post_batch,
            returncode=0, stdout_SHA256="a" * 64)])}
        self.score_out = self.repo / "score-fixture"
        self.score_receipt = self.receipt()
        self.verdict = dict(request=finish.REQUEST, independent_observation=True, state="COMPILE_VERIFIED",
                            columns=finish.COLUMNS, row=self.row, observations=[self.row],
                            terminal_row_SHA256=digest(self.row), intent_SHA256=sha256(self.target).hexdigest(),
                            score_receipt_SHA256=sha256(self.score_receipt).hexdigest()) | self.audit_pins

    def receipt(self):
        arms, audit = {}, {}
        for arm in finish.ARMS:
            directory = self.score_out / arm
            directory.mkdir(parents=True)
            pins = {}
            for name, key in (("fused_scores.o", "object_SHA256"), ("fused_scores.before.d", "fresh_M_SHA256"),
                              ("fused_scores.d", "actual_MD_SHA256")):
                path = directory / name
                data = (arm + "/" + name).encode()
                path.write_bytes(data)
                pins[key] = sha256(data).hexdigest()
            audit[arm] = dict(pins, symbols_stdout_SHA256="b" * 64)
            arms[arm] = dict(pins, object=str(directory / "fused_scores.o"), exports=[linker.HIDDEN])
        self.audit_pins = dict(arms=audit)
        return raw(dict(completed=True, passed=True, manifest_SHA256=linker.MANIFEST, all_fresh_M_before_compile=True,
                        actual_device_TUs=2, actual_GPU_steps=0, runtime_calls=0, original_step=self.row[0],
                        epoch=1795300000.5, deadline=1795300300.5, node=finish.NODE, arms=arms))

    def proof(self, *, capture_at=990.):
        reconcile = raw(self.verdict)
        capture = finish.terminal_receipt(reconcile, monotonic=lambda: capture_at, wall=lambda: capture_at + 3000)
        self.snapshot["local_receipt"]["payload_SHA256"] = digest(
            {key: value for key, value in self.snapshot.items() if key != "local_receipt"})
        return self.target, reconcile, raw(capture), raw(self.snapshot), raw(self.post_accounting)

    def prepare(self, *, now=1000., capture_at=990.):
        return finish.prepare(finish.claimed_run(self.database), self.local, *self.proof(capture_at=capture_at),
                              now=now, boot_anchor=3000.)

    def assert_holds(self):
        self.assertEqual(finish.claimed_run(self.database)[0], "UNKNOWN")

    def test_exact_link_kwargs_and_original_finished_pair_feed_downstream(self):
        result = self.prepare()
        self.assertEqual((result["planned"], result["completed"], result["missing"]), (2, 2, 0))
        self.assertEqual(result["evidence_SHA256"], digest(result["evidence"]))
        self.assertEqual(result["link_kwargs"], dict(
            generated_source=("/refactor/results/Kimi461-output-residual-wave-score-gfx936-CPU-v1-controller/generated"),
            score_receipt_sha256=sha256(self.score_receipt).hexdigest(),
            score_accounting=("/refactor/results/Kimi461-output-residual-wave-score-gfx936-CPU-v1-controller/"
                              "original-terminal-reconcile.json"),
        ))
        self.assertEqual(result["link_score_accounting_SHA256"], sha256(raw(self.verdict)).hexdigest())
        self.assertEqual(set(linker.qualified_scores(json.loads(self.score_receipt), self.verdict,
                                                    output=self.score_out)), set(finish.ARMS))
        self.assert_holds()
        fixture_queue = Queue(self.database)
        self.addCleanup(fixture_queue.db.close)
        fixture_queue.finish(finish.DIRECT_RUN, result)
        self.assertEqual(fixture_queue.dispatch_state(finish.DIRECT_RUN)["state"], "TERMINAL")

    def test_failed_terminal_keeps_two_unverified_arms_and_no_link(self):
        self.row[2:4] = ["FAILED", "1:0"]
        self.verdict.update(state="TERMINAL_UNVERIFIED", terminal_row_SHA256=digest(self.row))
        result = self.prepare()
        self.assertEqual((result["planned"], result["completed"], result["missing"]), (2, 0, 2))
        self.assertEqual((result["actual_compile_state"], result["actual_compile_exit_code"]), ("FAILED", "1:0"))
        self.assertIsNone(result["link_kwargs"])
        self.assertFalse(result["passed"])
        self.assert_holds()

    def test_unknown_has_no_finish_or_release(self):
        self.verdict = dict(request=finish.REQUEST, independent_observation=True, state="UNKNOWN", observations=[None])
        with self.assertRaisesRegex(ValueError, "no independently verified terminal"):
            self.prepare()
        self.assert_holds()

    def test_foreign_or_older_post_refuses_release(self):
        with self.assertRaisesRegex(ValueError, "predates reconcile receipt"):
            self.prepare(capture_at=998.)
        self.post_accounting["row"][0] = "1103606.714"
        with self.assertRaisesRegex(ValueError, "no successful Slurm step"):
            self.prepare()
        self.assert_holds()

    def test_stale_post_refuses_release(self):
        with self.assertRaisesRegex(ValueError, "not fresh"):
            self.prepare(now=1060.)
        self.assert_holds()

    def test_device_holder_or_nonzero_post_gres_refuses_release(self):
        self.snapshot["nodes"][finish.NODE].update(KFD_clean=False, kfd_process_count=1)
        with self.assertRaisesRegex(ValueError, "still has a Slurm step or a device holder"):
            self.prepare()
        self.snapshot["nodes"][finish.NODE].update(KFD_clean=True, kfd_process_count=0)
        original_tres = self.post_accounting["row"][-1]
        for resources in (",gres/dcu=1", ",gres/dcu=0,gres/gpu=1"):
            self.post_accounting["row"][-1] = original_tres + resources
            with self.assertRaisesRegex(ValueError, "zero-GRES step"):
                self.prepare()
        self.assert_holds()

    def test_altered_original_intent_or_raw_reconcile_is_rejected(self):
        inputs = list(self.proof())
        inputs[2] = raw(dict(json.loads(inputs[2]), raw_reconcile_SHA256="0" * 64))
        with self.assertRaisesRegex(ValueError, "reconcile bytes"):
            finish.prepare(finish.claimed_run(self.database), self.local, *inputs, now=1000., boot_anchor=3000.)
        with self.assertRaisesRegex(ValueError, "reservation and intent"):
            finish.prepare(finish.claimed_run(self.database), raw(dict(json.loads(self.local), publication="b" * 40)),
                           *self.proof(), now=1000., boot_anchor=3000.)
        self.assert_holds()

    def test_cli_writes_one_packet_and_leaves_queue_bytes_unchanged(self):
        names = ("target-intent", "reconcile", "terminal-receipt", "post-snapshot", "post-accounting")
        argv = ["prepare_kimi_r461_output_residual_wave_finish.py"]
        for name, data in zip(names, self.proof(), strict=True):
            path = self.dispatch.intent.with_name(name + ".fixture.json")
            path.write_bytes(data)
            argv.extend(("--" + name, str(path)))
        before = sha256(self.database.read_bytes()).hexdigest()
        with (patch.object(finish, "REPO", self.repo), patch.object(finish, "QUEUE_DB", self.database),
              patch.object(finish.time, "monotonic", return_value=1000.), patch.object(finish.time, "time", return_value=4000.),
              patch("sys.argv", argv), redirect_stdout(StringIO()) as stdout):
            finish.main()
            output = json.loads(stdout.getvalue())
            with self.assertRaises(FileExistsError):
                finish.main()
        packet_path = Path(output["finish"])
        self.assertEqual(packet_path, self.dispatch.intent.with_name(finish.FINISH_NAME))
        self.assertEqual(output["finish_SHA256"], sha256(packet_path.read_bytes()).hexdigest())
        self.assertFalse(output["queue_modified"])
        self.assertEqual(before, sha256(self.database.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
