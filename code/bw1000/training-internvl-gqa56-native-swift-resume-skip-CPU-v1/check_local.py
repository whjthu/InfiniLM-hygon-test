"""Check immutable source binding and the resumed position boundary without Torch."""
import json
import sys
from types import SimpleNamespace
import unittest

import prepare


sys.path.insert(0,str(prepare.BASE))
import skip_route
sys.path.remove(str(prepare.BASE))


class ResumeSkipTests(unittest.TestCase):
    def test_frozen_parent_and_new_normal_entry(self):
        files=prepare.packet()
        self.assertEqual(files,prepare.packet())
        doc=json.loads(files["SOURCE_FILES.json"])
        self.assertEqual(doc["parent_4738_source_manifest_SHA256"],prepare.BASE_SHA)
        self.assertEqual({key:val["SHA256"] for key,val in doc["files"].items()},
                         {key:prepare.digest(val) for key,val in files.items() if key!="SOURCE_FILES.json"})
        self.assertIn(b"skip_route.observe",files["probe.py"])
        self.assertIn(b"BYTE_EXACT_ALL_704_ORIGINAL_4738_SLOTS",files["probe.py"])
        self.assertIn(b"with patch(holder):",files["skip_route.py"])
        self.assertFalse(doc["actual_gpu"])

    def test_raw_sampler_and_real_fetch_order_are_independent_gates(self):
        raw=[3,19,8,12,40,52,47,61]
        order=list(range(63,-1,-1))
        typed={i:f"{i:064x}" for i in range(64)}
        retained=[dict(logical=i,physical=order[i],typed_row_SHA256=typed[order[i]]) for i in raw[4:]]
        self.assertEqual(skip_route.verify_resume_slots(raw,retained,raw[4:],raw,order,typed),(raw[:4],raw[4:]))
        for bad in ([*raw],raw[:4],raw[4:][::-1]):
            with self.assertRaisesRegex(ValueError,"BEFORE_REAL_DATASET_FETCH"):
                skip_route.verify_resume_slots(raw,retained,bad,raw,order,typed)
        with self.assertRaisesRegex(ValueError,"PRE_SKIP_POSITIONS"):
            skip_route.verify_resume_slots(raw[::-1],retained,raw[4:],raw,order,typed)

    def test_original_base_seed_unchanged_and_resumed_seed43_required(self):
        data=dict(rank=5,world_size=8,tp_size=1,total_samples=8,batch_size=1,base_seed=42,
                  curr_seed=43,shuffle=True,drop_last=False,group_by_length=False)
        skip_route.sampler_fields(SimpleNamespace(**data),5,43)
        with self.assertRaisesRegex(ValueError,"FIELDS_AND_SEED"):
            skip_route.sampler_fields(SimpleNamespace(**data),5,42)
        with self.assertRaisesRegex(ValueError,"FIELDS_AND_SEED"):
            skip_route.sampler_fields(SimpleNamespace(**dict(data,base_seed=43)),5,43)

    def test_plain_accelerate_import_is_not_a_swift_patch(self):
        def plain(*args):
            return args
        with self.assertRaisesRegex(ValueError,"LIVE_SWIFT_NESTED"):
            skip_route.captured_skip(plain,plain,object())


if __name__=="__main__":
    unittest.main()
