"""Host-only checks for the DP2 preflight contract."""
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import preflight


def fixture():
    cache, native, model = (preflight.ORIGINAL_PATHS[key] for key in ("cache","native","model"))
    return dict(weights=model,source=native,config=dict(training_cache=cache,training_cache_sha=preflight.CACHE_SHA,
                world_size=2,sp=1,seed=20260911,transformer_version="480p_t2v",warmup_steps=5,max_steps=20,
                checkpoint_steps=[10,20,25],lora_rank=4,lora_alpha=8,lora_target_modules=["img_attn_q","img_attn_v"]))


class Preflight(unittest.TestCase):
    def test_private_framework_keeps_original_sealed_and_replaces_only_bound_provider(self):
        with TemporaryDirectory() as td:
            parent = Path(td).resolve()
            original = parent / "original"
            original.mkdir()
            (original / "provider.py").write_bytes(b"base")
            (original / "__init__.py").write_bytes(b"")
            pins = {name: {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}
                    for name, raw in (("provider.py", b"base"), ("__init__.py", b""))}
            private = parent / "private"
            observed = preflight.materialize_framework(pins, original, private, {"provider.py": b"replacement"})
            self.assertEqual(observed["provider.py"], sha256(b"replacement").hexdigest())
            self.assertEqual((private / "__init__.py").read_bytes(), b"")
            self.assertEqual((original / "provider.py").read_bytes(), b"base")
            with self.assertRaisesRegex(ValueError, "Patch targets"):
                preflight.materialize_framework(pins, original, parent / "other", {"unbound.py": b"new"})

    def test_weight_receipt_uses_size_and_checks_execution_identity(self):
        weight = Path("/refactor/transformer.safetensors")
        inode = 11_025_132_338
        prior = dict(passed=True, stable_stat=True, weight_path=str(weight),
                     weight_bytes=preflight.MODEL_BYTES, weight_SHA256=preflight.MODEL_SHA,
                     file_identity=[49, inode, preflight.MODEL_BYTES, 21, 22])
        observed = SimpleNamespace(st_mode=stat.S_IFREG, st_dev=53, st_ino=inode, st_size=preflight.MODEL_BYTES,
                                   st_mtime_ns=21, st_ctime_ns=22)
        with patch.object(preflight, "WEIGHT", weight), patch.object(preflight, "read_pinned", return_value=json.dumps(prior)), \
             patch.object(Path, "lstat", return_value=observed), patch.object(Path, "resolve", return_value=weight):
            self.assertEqual(preflight.verify_transformer(), [53, inode, preflight.MODEL_BYTES, 21, 22])
            observed.st_size = inode
            with self.assertRaisesRegex(ValueError, "transformer extent"):
                preflight.verify_transformer()
            observed.st_size = preflight.MODEL_BYTES
            observed.st_ino += 1
            with self.assertRaisesRegex(ValueError, "transformer snapshot"):
                preflight.verify_transformer()

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
