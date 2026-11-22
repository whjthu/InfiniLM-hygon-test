"""Local source-binding and scored head authority checks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import launch
import receive


def local_read(publication, path, wanted=None):
    raw=(launch.REPO/path).read_bytes()
    if wanted is not None and sha256(raw).hexdigest()!=wanted:
        raise ValueError("source pins differ")
    return raw


class Pair(unittest.TestCase):
    def setUp(self):
        self.raw=local_read("fixture",launch.NEAR7_PUBLIC,launch.NEAR7_PUBLIC_SHA)
        self.proof=receive.validate_public(self.raw)
        self.native=dict(
            metrics=dict(global_optimizer_step=25,executed_optimizer_steps=25,cached_valid_batches=7,
                         training_scope="camera_head",trainable_backbone_parameters=0,camera_head_sha_after="a"*64),
            checks={"frozen_aggregator_unchanged":True},
            artifacts=[{"path":"/pilot/"+receive.HEAD,"sha256":"c"*64}])

    def test_dispatch_identity_is_full_hex_and_matches_receiver(self):
        self.assertRegex(launch.REQUEST, r"^[0-9a-f]{32}$")
        self.assertEqual(launch.REQUEST, receive.REQUEST)
        self.assertEqual(launch.REQUEST, json.loads((launch.HERE/"CLAIM.json").read_bytes())["request"])

    def test_head_authority_binds_one_new_immutable_checkpoint(self):
        authority=receive.authority_from(self.proof,self.native,1792)
        self.assertEqual(set(authority),{"near7_25"})
        self.assertEqual(authority["near7_25"]["file_SHA256"],"c"*64)
        self.assertEqual(authority["near7_25"]["module_SHA256"],"a"*64)
        self.assertEqual(authority["near7_25"]["gpu_step"],"1103606.978")
        self.assertEqual(authority["near7_25"]["actual_checkpoint_bytes_read"],0)
        wrong=deepcopy(self.native)
        wrong["metrics"]["trainable_backbone_parameters"]=1
        with self.assertRaisesRegex(ValueError,"artifact or source scope"):
            receive.authority_from(self.proof,wrong,1792)

    def test_failure_denominator_and_exact_published_reference(self):
        self.assertEqual(self.proof["actual_updates"],25)
        self.assertEqual(self.proof["accounting"]["gpu"][2:4],["COMPLETED","0:0"])
        with self.assertRaisesRegex(ValueError,"published near7"):
            receive.validate_public(self.raw+b" ")
        claim=json.loads((launch.HERE/"CLAIM.json").read_bytes())
        self.assertEqual((claim["planned_windows"],claim["admitted_windows"],claim["expected_predictions"],
                          claim["original_missing_windows"],claim["optimizer_updates"]),(24,22,44,2,0))

    def test_frozen_program_has_exact_derived_arms_and_no_old_head_mounts(self):
        files=launch.controller_sources("f"*40,local_read)
        self.assertEqual(len(files),16)
        claim=json.loads((launch.HERE/"CLAIM.json").read_bytes())
        self.assertEqual(sha256(files[launch.ORIGINAL_INDEX]).hexdigest(),claim["remapped_source_index_SHA256"])
        self.assertIn(b"'/near7:ro'",files[launch.CONTROLLER])
        self.assertNotIn(b"'/lowlr:ro'",files[launch.CONTROLLER])
        scorer=files["tools/score_vggt_tum_cross_camera_dev.py"]
        self.assertIn(b'ARMS = {"near7_25":',scorer)
        self.assertIn(b'("base", *ARMS)',scorer)
        self.assertIn(b"from score_vggt_tum_cross_camera_dev import ARMS",files["tools/audit_vggt_tum_cross_camera_dev.py"])
        with TemporaryDirectory() as tmp:
            repo=Path(tmp)
            path=repo/launch.BASE/"CLAIM.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(claim,GPU_GO=True)))
            _,_,_,program=launch.frozen(repo,"f"*40,epoch=100,deadline=1900,read=local_read)
            compile(program,"<frozen bootstrap>","exec")
            self.assertLess(len(program),131072)


if __name__=="__main__":
    unittest.main()
