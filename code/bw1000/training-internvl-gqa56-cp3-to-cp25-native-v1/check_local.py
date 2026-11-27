import json
from pathlib import Path
import tempfile
import unittest

import render
import series_contract as series
from series_live import SeriesCursor


RECEIPT=render.REPO/"local-only/internvl-gqa56-arrow-continuation-epochs1to12-once-1ccc01ddf76049dca2e32d9f013be644.json"


def fixture():
    updates=[]
    for step in range(4,26):
        epoch,half=divmod(step-1,2)
        slots=[]
        for micro in range(1,5):
            for rank in range(8):
                offset=half*4+micro-1
                position=rank*8+offset
                slots.append(dict(rank=rank,accumulation=micro,update=step,epoch_index=epoch,
                                  rank_offset0=offset,source_ordinal=position+1,logical=position,
                                  physical=position,row_SHA256=f"{position:064x}"))
        updates.append(dict(global_step=step,slots=slots))
    return dict(start_global_step=3,start_epoch=1.5,max_steps=25,updates=updates)


class ContinuationTests(unittest.TestCase):
    def test_production_render_with_exact_native_cpu_proof(self):
        files=render.render(RECEIPT)
        self.assertEqual(files,render.render(RECEIPT))
        claim=json.loads(files["CLAIM.json"])
        pins=json.loads(files["SOURCE_FILES.json"])["files"]
        self.assertEqual(set(pins),set(files)-{"SOURCE_FILES.json"})
        for name,pin in pins.items():
            self.assertEqual((pin["bytes"],pin["SHA256"]),(len(files[name]),render.digest(files[name])))
        self.assertEqual((claim["planned_updates"],claim["planned_input_slots"],claim["placement"]["cpus"]),(22,704,64))
        self.assertFalse(claim["GPU_GO"])
        self.assertIn(b"checkpoint-3",files["node.py"])
        self.assertIn(b"native_skip_batches=4",files["callback.py"])
        self.assertIn(b'"--ignore_data_skip", "false"',files["cp3_contract.py"])
        self.assertIn(b"actual_rank_forward_witnesses",files["native.py"])
        self.assertIn(b"EXACT_4738_ZERO_GRES_TERMINAL",files["owner_gate.py"])
        self.assertTrue(json.loads(files["SKIP_SOURCE.json"])["qualified"])
        self.assertFalse(claim["GPU_GO"])
        self.assertIn(b"QUALIFIED_DEPLOYED_SKIP_SOURCE_REQUIRED_BEFORE_OWNER",files["owner_gate.py"])

    def test_mutated_schedule_rejected_before_render(self):
        with tempfile.TemporaryDirectory() as scratch:
            altered=Path(scratch)/"changed.json"
            altered.write_bytes(RECEIPT.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"ORIGINAL_4738_CPU_TRANSPORT"):
                render.render(altered)

    def test_exact_704_update_offsets_and_boundaries(self):
        raw=fixture()
        ranks=series.parse_plan(raw)
        self.assertEqual([len(rank) for rank in ranks],[88]*8)
        self.assertEqual((ranks[0][0]["epoch_index"],ranks[0][0]["rank_offset0"]),(1,4))
        self.assertEqual((ranks[-1][-1]["epoch_index"],ranks[-1][-1]["rank_offset0"]),(12,3))
        raw["updates"][0]["slots"][0]["rank_offset0"]=0
        with self.assertRaisesRegex(ValueError,"EPOCH_CURSOR"):
            series.parse_plan(raw)

    def test_actual_cursor_rejects_early_wrong_or_unskipped_consumption(self):
        ranks=series.parse_plan(fixture())
        cursor=SeriesCursor(0,ranks[0],list(range(64)))
        source=dict(ordinal=5,row_SHA256=f"{4:064x}")
        with self.assertRaisesRegex(ValueError,"NATIVE_SKIP"):
            cursor.prepare(0,source)
        cursor.skip_consumed=True
        self.assertEqual(list(cursor.witness(([x] for x in range(8)),1)),[[i] for i in range(8)])
        with self.assertRaisesRegex(ValueError,"COLLATOR_SOURCE"):
            cursor.prepare(0,dict(source,ordinal=1))
        associated=cursor.prepare(0,source)
        self.assertEqual(cursor.forward(0,associated),ranks[0][0])
        with self.assertRaisesRegex(ValueError,"ORDERED_NATIVE"):
            list(cursor.witness(iter(()),3))

    def test_rate_remains_positive_for_22_optimizer_steps(self):
        self.assertTrue(all(series.lr(cursor)>0 for cursor in range(3,25)))
        self.assertEqual(series.lr(25),0)


if __name__=="__main__":
    unittest.main()
