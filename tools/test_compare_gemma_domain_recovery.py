import json
from pathlib import Path
import tempfile
import unittest

from compare_gemma_domain_recovery import compare


def state(step):
    return {name: {"fingerprint": f"{name}-{step}"} for name in ("adapter", "optimizer", "scheduler")}


def fixture(root):
    for rank in range(8):
        for arm, start, folder in (("continuous", 0, "continuous-witness"), ("resumed", 20, "resume-witness")):
            path = root / arm / "train" / folder / f"rank-{rank:02d}"
            path.mkdir(parents=True)
            count = 100 - start * 4
            records = {
                "owner.json": dict(rank=rank, local_rank=rank, PID=1000 + start + rank, startticks=100 + start),
                "result.json": dict(rank=rank, world_size=8, completed_global_step=25,
                    saved_checkpoints=[25] if start else [10, 20, 25],
                    inputs=[dict(ordinal=i + 1, global_step=start + i // 4,
                                 full_input={"stamp": i + 4 * start}) for i in range(count)],
                    outputs=[dict(ordinal=i + 1, loss={"stamp": i + 4 * start}, loss_scalar=0.25)
                             for i in range(count)],
                    steps=[dict(global_step=step, state=state(step)) for step in range(start + 1, 26)]),
                "checkpoint-25-rng.json": dict(rank=rank, RNG=[rank, 25]),
            }
            if start:
                records["result.json"].update(native_resume_window_complete=True,
                    RNG_loads=[dict(global_step=20, saved_equal=True, actual=[rank, 20])])
                records["native-data-skip.json"] = dict(rank=rank, before=100, skipped=80, after=20)
                records["restored-before-RNG.json"] = dict(restored_parameter_optimizer_scheduler_equal=True, actual=state(20))
            else:
                records["checkpoint-20-rng.json"] = dict(rank=rank, RNG=[rank, 20])
            for name, value in records.items():
                (path / name).write_text(json.dumps(value), encoding="utf-8")


class GemmaRecoveryComparison(unittest.TestCase):
    def test_compares_all_fixed_rank_pairs_and_reports_divergent_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            result = compare(root)
            self.assertTrue(result["exact_paired_recovery"])
            self.assertEqual(result["compared_tail_microbatch_pairs"], 160)
            self.assertEqual(result["compared_rank_update_pairs"], 40)
            self.assertEqual(len(result["evidence_SHA256"]), 72)
            changed = root / "resumed/train/resume-witness/rank-03/result.json"
            body = json.loads(changed.read_bytes())
            body["outputs"][7]["loss_scalar"] = 0.5
            changed.write_text(json.dumps(body), encoding="utf-8")
            result = compare(root)
            self.assertFalse(result["exact_paired_recovery"])
            self.assertEqual(result["maximum_absolute_loss_gap"], 0.25)
            self.assertEqual(result["differences"], [{"rank": 3, "field": "loss", "tail_ordinal": 8}])

    def test_retains_missing_rank_and_invalid_skip_in_the_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "resumed/train/resume-witness/rank-04/result.json").unlink()
            bad = root / "resumed/train/resume-witness/rank-06/native-data-skip.json"
            bad.write_text(json.dumps(dict(rank=6, before=100, skipped=79, after=21)), encoding="utf-8")
            result = compare(root)
            self.assertEqual(result["expected_ranks"], 8)
            self.assertEqual(result["planned_tail_microbatch_pairs"], 160)
            self.assertEqual(result["compared_tail_microbatch_pairs"], 120)
            self.assertEqual([row["rank"] for row in result["differences"]], [4, 6])
            self.assertFalse(result["exact_paired_recovery"])

    def test_rejects_a_reused_process_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            owner = root / "resumed/train/resume-witness/rank-07/owner.json"
            owner.write_text(json.dumps(dict(rank=7, local_rank=7, PID=1007, startticks=100)), encoding="utf-8")
            result = compare(root)
            self.assertEqual(result["compared_ranks"], list(range(7)))
            self.assertFalse(result["exact_paired_recovery"])


if __name__ == "__main__":
    unittest.main()
