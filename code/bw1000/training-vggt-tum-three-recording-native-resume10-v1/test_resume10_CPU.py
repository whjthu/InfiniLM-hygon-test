"""Exercise the new resume comparison and original receipt gates locally."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_resume import compare, BUNDLE_SHA, ORDER_SHA, VIEW_SHA
import recovery_reference as original


HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[2] / "results/bw1000/2026-11-22/training-vggt-nine-public-RGBD-original927-native25-resume20-terminal.json"


def checkpoint(number):
    return dict(global_step=number, next_global_step=number+1, next_batch_cursor=number%9,
                file_SHA256=original.CP10_SHA if number==10 else "a"*64,
                full_boundary_SHA256=original.CP10_BOUNDARY_SHA if number==10 else "f"*64,
                model_SHA256="b"*64, optimizer_SHA256="c"*64, placements_SHA256="d"*64, rng_SHA256="e"*64)


def arm(name, start, origin):
    return dict(
        arm=name, process=dict(boot_id=f"boot-{origin}", pid=1, startticks=2), epoch=origin, deadline=origin+1700,
        bundle_SHA256=BUNDLE_SHA, backend_SHA256="1"*64, checkpoint_identity={"model":"pinned"},
        input_order_SHA256=ORDER_SHA, view_order_SHA256=VIEW_SHA, executed_optimizer_steps=25-start,
        resume_source_SHA256=original.CP10_SHA if start else None, final_state_SHA256="2"*64,
        scheduler=dict(type="constant_learning_rate_in_optimizer_param_groups",
                       mutable_scheduler=False,learning_rate=2e-5), gradient_accumulation=1,
        steps=[dict(global_optimizer_step=n,batch_cursor=(n-1)%9,loss=n/32,grad_norm=n/8,
                    camera_head_grad_norm_before_clip=n/8+1) for n in range(start+1,26)],
        checkpoints={str(n):checkpoint(n) for n in (10,20,25) if n>start},
    )


class Restore(unittest.TestCase):
    def setUp(self):
        self.baseline=arm("continuous",0,10)
        self.resumed=arm("resume10",10,2000)

    def pair(self):
        return compare({"continuous":self.baseline,"resume10":self.resumed})

    def test_all_fifteen_updates_use_the_pinned_suffix(self):
        result=self.pair()
        self.assertTrue(result["comparison_passed"])
        self.assertEqual(result["actual_update_invocations"],15)
        self.assertEqual([r["global_step"] for r in result["arms"]["resume10"]["steps"]],list(range(11,26)))
        self.assertEqual(set(result["arms"]["resume10"]["checkpoint_component_hashes_equal"]),{"20","25"})

    def test_reject_original_epoch_or_wrong_resume(self):
        self.resumed["epoch"]=self.baseline["epoch"]
        with self.assertRaisesRegex(ValueError,"fresh epoch"):
            self.pair()
        self.resumed["epoch"]=2000
        self.resumed["resume_source_SHA256"]="9"*64
        with self.assertRaisesRegex(ValueError,"original continuous checkpoint"):
            self.pair()

    def test_preserve_numeric_and_state_failures(self):
        self.resumed["steps"][0]["camera_head_grad_norm_before_clip"]+=0.01
        self.resumed["checkpoints"]["20"]["placements_SHA256"]="9"*64
        self.resumed["checkpoints"]["25"]["rng_SHA256"]="9"*64
        report=self.pair()
        self.assertFalse(report["comparison_passed"])
        observed=report["arms"]["resume10"]
        self.assertGreater(observed["maximum_camera_preclip_error"],1e-6)
        self.assertFalse(observed["checkpoint_component_hashes_equal"]["20"]["placements_SHA256"])
        self.assertFalse(observed["checkpoint_component_hashes_equal"]["25"]["rng_SHA256"])

    def test_incomplete_native_update_does_not_pass(self):
        self.resumed["steps"].pop(0)
        with self.assertRaisesRegex(ValueError,"Missing or duplicate"):
            self.pair()

    def test_public_reference_binds_checkpoint_and_source(self):
        raw=PUBLIC.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),original.ORIGINAL_PUBLIC_SHA)
        result=json.loads(raw)
        intent=dict(request=original.ORIGINAL_REQUEST,publication=original.ORIGINAL_PUBLICATION,
                    parent="1103606",node="f13r4n09",epoch=self.baseline["epoch"],deadline=self.baseline["epoch"]+1800)
        self.assertEqual(original.validate_reference(result,intent,self.baseline)["file_SHA256"],original.CP10_SHA)
        wrong=deepcopy(self.baseline)
        wrong["checkpoints"]["10"]["full_boundary_SHA256"]="0"*64
        with self.assertRaisesRegex(ValueError,"actual original checkpoint 10"):
            original.validate_reference(result,intent,wrong)
        result["actual_independent_post"][2]="FAILED"
        with self.assertRaisesRegex(ValueError,"continuous accounting"):
            original.validate_reference(result,intent,self.baseline)

    def test_checkpoint_digest_streams_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            checkpoint_file=root/original.CHECKPOINT
            checkpoint_file.parent.mkdir(parents=True)
            data=b"file"*(2<<20)
            checkpoint_file.write_bytes(data)
            binding=dict(checkpoint10_SHA256=sha256(data).hexdigest(),checkpoint10_bytes=len(data))
            # The target Linux checks include inode fields absent from Windows fstat.
            fingerprint=(lambda x:(x.st_size,x.st_mtime_ns)) if os.name=="nt" else original.signature
            with patch.object(original,"signature",fingerprint), patch.object(Path,"read_bytes",side_effect=AssertionError):
                self.assertEqual(original.check_checkpoint(root,binding),checkpoint_file)
                with checkpoint_file.open("r+b") as stream:
                    stream.write(b"wrong")
                with self.assertRaisesRegex(ValueError,"bytes differ"):
                    original.check_checkpoint(root,binding)


if __name__=="__main__":
    unittest.main()
