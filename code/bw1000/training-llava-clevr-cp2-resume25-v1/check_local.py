"""Regression for the actual CP2 archive, resume cursor and prospective plan."""

from copy import deepcopy
from pathlib import Path
import unittest

import contract


class Continuation(unittest.TestCase):
    def test_actual_checkpoint_inventory(self):
        repo = Path(__file__).resolve().parents[3]
        manifest = contract.archive(repo / "results/bw1000/2026-11-28/specialized-llava-4983-eight-rank-first2-cp2-and-4984-clean.json")
        self.assertEqual(len(manifest), 17)
        self.assertEqual(manifest["trainer_state.json"]["SHA256"],
                         "3a8b058d20aa0dc37979f62fb2d4982339e0b939f79ce56693dcd20ff91fb841")

    def test_epoch1_no_skip(self):
        actual = dict(global_step=2, epoch=1.0, max_steps=25)
        self.assertEqual(contract.cursor(actual)["skip_microbatches"], 0)
        for other in (dict(actual, epoch=1.5), dict(actual, global_step=3)):
            with self.assertRaisesRegex(ValueError, "EXACT_CP2_EPOCH_FROM_ORIGINAL_FILE"):
                contract.cursor(other)

    def test_assembly_and_production_guard(self):
        expected = {i: dict(ordinal=i, row_sha256=f"{i:064x}", image_sha256=f"{100+i:064x}",
                            input_sha256=f"{200+i:064x}") for i in range(1, 65)}
        actual = [[dict(epoch=e, seed=42+e, rows=[expected[i] for i in range(1 + rank * 8, 9 + rank * 8)])
                   for e in contract.EPOCHS] for rank in range(8)]
        plan = contract.assemble(actual, expected)
        self.assertEqual(sum(len(u["rows"]) for u in plan["updates"]), 736)
        self.assertEqual(plan["updates"][0]["rows"][0]["offset"], 0)
        self.assertEqual(plan["updates"][-1]["rows"][-1]["offset"], 3)
        first = dict(global_step=3, actual_epoch=1, actual_skip_microbatches=0, rows=plan["updates"][0]["rows"],
                     rank_updates=[dict(rank=rank, actual_adamw_counters_before=[2]*504, actual_adamw_counters_after=[3]*504,
                                        finite_losses=4, learning_rate=.00009, parameter_delta_l2=.4, pre_SHA256="a"*64,
                                        post_SHA256="b"*64, restored_device_rng=True) for rank in range(8)])
        self.assertTrue(contract.first_resumed_boundary(first, plan["updates"][0]))
        wrong = deepcopy(first)
        wrong["rank_updates"][0]["actual_adamw_counters_after"][0] = 4
        with self.assertRaisesRegex(ValueError, "ACTUAL_STEP3_FINITE_POSITIVE_CHANGE"):
            contract.first_resumed_boundary(wrong, plan["updates"][0])
        with self.assertRaisesRegex(ValueError, "ACTUAL_RESUMED_32_FIRST_FORWARDS"):
            contract.first_resumed_boundary(dict(first, actual_skip_microbatches=4), plan["updates"][0])
        actual[0][0]["rows"][0] = expected[9]
        with self.assertRaisesRegex(ValueError, "EXACT_OBSERVED_PHYSICAL_4878_SOURCE"):
            contract.assemble(actual, expected)


if __name__ == "__main__":
    unittest.main()
