"""Host-only checks for the DP2 preflight contract."""
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import preflight


def fixture():
    cache, native, model = (preflight.ORIGINAL_PATHS[key] for key in ("cache","native","model"))
    return dict(weights=model,source=native,config=dict(training_cache=cache,training_cache_sha=preflight.CACHE_SHA,
                world_size=2,sp=1,seed=20260911,transformer_version="480p_t2v",warmup_steps=5,max_steps=20,
                checkpoint_steps=[10,20,25],lora_rank=4,lora_alpha=8,lora_target_modules=["img_attn_q","img_attn_v"]))


class Preflight(unittest.TestCase):
    def test_exact_frozen_overlay_source(self):
        self.assertEqual(preflight.source_check()["local_overlay_files"],preflight.PATCHES)
        self.assertFalse(preflight.source_check()["deployed_verified"])

    def test_original_shared_cache_and_dual_rank_topology(self):
        self.assertEqual(preflight.validate_recipe(fixture())[preflight.ORIGINAL_PATHS["cache"]],str(preflight.CACHE))
        for key, invalid in (("world_size",1),("sp",2),("max_steps",25),("training_cache_sha","0"*64)):
            fake=fixture()
            fake["config"][key]=invalid
            with self.assertRaises(ValueError):
                preflight.validate_recipe(fake)

    def test_exact_a100_pin_rejects_an_equivalent_synthetic_summary(self):
        with TemporaryDirectory() as td:
            path=(Path(td)/"launch.json").resolve()
            path.write_bytes(json.dumps(fixture()).encode())
            with self.assertRaisesRegex(ValueError,"Frozen original length or SHA"):
                preflight.read_pinned(path,preflight.DP2_SHA,1563)

    def test_actual_two_card_selector_and_checkpoint_scope(self):
        root=PurePosixPath("/project/results")
        original=PurePosixPath("/original/launch.json")
        asset_map=PurePosixPath("/original/assets.json")
        out=root/preflight.LABEL/"outputs/hunyuanvideo-1.5/train/continuous"
        with patch.object(preflight,"RESULTS",root):
            argv=preflight.native_argv(original,asset_map,"1,7","resume10",out/"checkpoint-10")
            self.assertEqual(argv[-2:],["--resume-from-checkpoint",str(out/"checkpoint-10")])
            for card in ("7", "7,7", "0,1,2"):
                with self.assertRaisesRegex(ValueError,"two distinct"):
                    preflight.native_argv(original,asset_map,card,"continuous")
            with self.assertRaisesRegex(ValueError,"new DP2 continuous"):
                preflight.native_argv(original,asset_map,"1,7","resume20",out/"checkpoint-10")
            with self.assertRaisesRegex(ValueError,"new DP2 continuous"):
                preflight.native_argv(original,asset_map,"1,7","resume20",out.parent/"checkpoint-20")


if __name__ == "__main__":
    unittest.main()
