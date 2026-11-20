import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_vggt_tum_cross_camera_dev import audit
from bind_vggt_tum_dev_rgb import sha
from score_vggt_tum_cross_camera_dev import ARMS, WINDOWS, canonical_scorer, summarize


class CrossCameraOutputAuditTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.inputs = self.root / "inputs.json"
        self.progress = self.root / "progress.jsonl"
        sequence, window = min(WINDOWS)
        self.heads_sha = "a" * 64
        truth = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
        truth[1, 0, 3] = -.05
        manifest = [dict(sequence=seq, window=w, state="RGB224_PRECONDITION_FAILED") for seq, w in sorted(WINDOWS)]
        manifest[0].update(state="RGB224_AND_RELATIVE_POSE_BOUND", rgb_file_SHA256="b" * 64,
                           world_to_camera=truth.tolist())
        raw = json.dumps(dict(windows=manifest)).encode()
        self.inputs.write_bytes(raw)
        self.inputs_sha = sha(raw)
        arms = ("base", *ARMS)
        rows = [dict(sequence=r["sequence"], window=r["window"], input_state=r["state"],
                     **{arm: dict(state="MISSING_INPUT", pair_error_deg=None) for arm in arms}) for r in manifest]
        rows[0].update(input_SHA256="b" * 64, **{arm: dict(state="NOT_RUN", pair_error_deg=None) for arm in arms})
        self.records = [dict(state="PLANNED", rows=json.loads(json.dumps(rows)),
                             inputs_SHA256=self.inputs_sha, heads_SHA256=self.heads_sha)]
        original = canonical_scorer()
        for arm in arms:
            if arm == "lowlr25":
                prediction = dict(state="INVALID_PREDICTION", forward_s=.2, pair_error_deg=None)
            else:
                prediction = dict(state="PREDICTED", forward_s=.1, extrinsics=truth.tolist(),
                                  **original["angular_errors"](truth, truth))
            rows[0][arm] = prediction
            self.records.append(dict(sequence=sequence, window=window, arm=arm, result=prediction))
        self.records.append(dict(state="COMPLETED", summary=summarize(rows, original["summarize_development"]),
                                 formal_CO3D=False, optimizer_updates=0, planned_windows=24, input_pairs=1))

    def audited(self):
        self.progress.write_text("".join(json.dumps(doc) + "\n" for doc in self.records), encoding="utf8")
        return audit(self.inputs, self.inputs_sha, self.heads_sha, self.progress)

    def test_reconstructs_24_window_denominator_and_invalid_state(self):
        out = self.audited()
        self.assertEqual(out["actual_prediction_events"], 3)
        self.assertEqual(out["raw_pose_predictions"], dict(base=1, alternate25=1, lowlr25=0))
        self.assertEqual(out["invalid_predictions"]["lowlr25"], 1)
        self.assertEqual(out["AUC"]["all_windows"]["base"]["failed_pairs"], 23)

    def test_rejects_missing_event_or_unrelated_terminal(self):
        del self.records[1]
        with self.assertRaisesRegex(ValueError, "event count"):
            self.audited()

    def test_rejects_forged_angles_and_summary(self):
        self.records[1]["result"]["rotation_deg"] += 1
        with self.assertRaisesRegex(ValueError, "original extrinsics"):
            self.audited()
        self.records[1]["result"]["rotation_deg"] -= 1
        self.records[-1]["summary"]["all_windows"]["base"]["failed_pairs"] = 0
        with self.assertRaisesRegex(ValueError, "terminal score"):
            self.audited()


if __name__ == "__main__":
    unittest.main()
