"""Local source-binding and scored head authority checks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import launch
import receive
import audit_terminal
import dev24_decision


def local_read(publication, path, wanted=None):
    raw=(launch.REPO/path).read_bytes()
    if wanted is not None and sha256(raw).hexdigest()!=wanted:
        raise ValueError("source pins differ")
    return raw


class Pair(unittest.TestCase):
    def setUp(self):
        self.raw=local_read("fixture",launch.NEAR7_PUBLIC,launch.NEAR7_PUBLIC_SHA)
        self.proof=receive.validate_public(self.raw)
        self.binding=json.loads((launch.HERE/"HEAD_BINDING.json").read_bytes())
        self.native=dict(
            metrics=dict(global_optimizer_step=25,executed_optimizer_steps=25,cached_valid_batches=14,domain_geometry={"verified_pairs":7},
                         training_scope="camera_head",trainable_backbone_parameters=0,camera_head_sha_after=self.binding["module_SHA256"]),
            checks={"frozen_aggregator_unchanged":True},
            artifacts=[{"path":"/pilot/"+receive.HEAD,"sha256":self.binding["file_SHA256"]}])

    def test_dispatch_identity_is_full_hex_and_matches_receiver(self):
        self.assertRegex(launch.REQUEST, r"^[0-9a-f]{32}$")
        self.assertEqual(launch.REQUEST, receive.REQUEST)
        self.assertEqual(launch.REQUEST, json.loads((launch.HERE/"CLAIM.json").read_bytes())["request"])

    def test_head_authority_binds_one_new_immutable_checkpoint(self):
        authority=receive.authority_from(self.proof,self.native,self.binding["head_bytes_stat"],self.binding)
        self.assertEqual(set(authority),{"near7_bidir25"})
        self.assertEqual(authority["near7_bidir25"]["file_SHA256"],self.binding["file_SHA256"])
        self.assertEqual(authority["near7_bidir25"]["module_SHA256"],self.binding["module_SHA256"])
        self.assertEqual(authority["near7_bidir25"]["gpu_step"],"1103606.1078")
        self.assertEqual(authority["near7_bidir25"]["actual_checkpoint_bytes_read"],0)
        with self.assertRaisesRegex(ValueError,"exact checkpoint binding"):
            receive.authority_from(self.proof,self.native,self.binding["head_bytes_stat"]-1,self.binding)
        wrong=deepcopy(self.native)
        wrong["metrics"]["trainable_backbone_parameters"]=1
        with self.assertRaisesRegex(ValueError,"artifact or source scope"):
            receive.authority_from(self.proof,wrong,self.binding["head_bytes_stat"],self.binding)

    def test_failure_denominator_and_exact_published_reference(self):
        self.assertEqual(self.proof["actual_updates"],25)
        self.assertEqual(self.proof["accounting"]["gpu"][2:4],["COMPLETED","0:0"])
        with self.assertRaisesRegex(ValueError,"published bidirectional near7"):
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
        self.assertIn(b'ARMS = {"near7_bidir25":',scorer)
        self.assertIn(b'("base", *ARMS)',scorer)
        self.assertIn(b"from score_vggt_tum_cross_camera_dev import ARMS",files["tools/audit_vggt_tum_cross_camera_dev.py"])
        self.assertIn(b"training-vggt-tum-near7-bidirectional14-native25-v1",files[launch.CONTROLLER])
        self.assertNotIn(b"training-vggt-tum-near7-sequence-holdout25-v1",files[launch.CONTROLLER])
        self.assertNotIn(b"1103606.978",scorer)
        self.assertEqual((launch.HERE/"read_vggt_dev24_roundoff_audit.py").read_bytes(),
                         (launch.REPO/"tools/read_vggt_dev24_roundoff_audit.py").read_bytes())
        with TemporaryDirectory() as tmp:
            repo=Path(tmp)
            path=repo/launch.BASE/"CLAIM.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(claim,GPU_GO=False)))
            with self.assertRaisesRegex(ValueError, "operative"):
                launch.frozen(repo,"f"*40,epoch=100,deadline=1900,read=local_read)
            path.write_text(json.dumps(dict(claim,GPU_GO=True)))
            _,_,_,program=launch.frozen(repo,"f"*40,epoch=100,deadline=1900,read=local_read)
            compile(program,"<frozen bootstrap>","exec")
            import ast
            import base64
            import zlib
            params=ast.literal_eval(ast.parse(program).body[0].value)
            payload=ast.literal_eval(ast.parse(zlib.decompress(base64.b64decode(params["body"]))).body[0].value)
            self.assertEqual((params["request"],payload["context"]["request"],audit_terminal.REQUEST),
                             (launch.REQUEST,)*3)
            self.assertEqual(set(payload["auxiliary"]),set(launch.AUXILIARY))
            self.assertEqual(payload["head_binding"],(launch.HERE/"HEAD_BINDING.json").read_bytes())
            self.assertLess(len(program),131072)


    def test_complete_invalid_prediction_counts_trigger_stop(self):
        def scores(failed):
            return dict(development_AUC_at_5=.05, development_AUC_at_15=.2,
                        development_AUC_at_30=.39, failed_pairs=failed)
        independent=dict(assessed_windows=24,input_pairs=22,actual_prediction_events=44,
                         raw_pose_predictions=dict(base=22,near7_bidir25=21),
                         invalid_predictions=dict(base=0,near7_bidir25=1),
                         AUC=dict(all_windows=dict(base=scores(2),near7_bidir25=scores(3))))
        self.assertEqual(dev24_decision.evaluate(independent)["decision"],"STOP_NEAR_ONLY_AUGMENTATION")
        independent["AUC"]["all_windows"]["near7_bidir25"]["failed_pairs"]=2
        self.assertEqual(dev24_decision.evaluate(independent)["decision"],"MISSING_OR_INCOMPLETE_DEV24")
        independent["invalid_predictions"]["near7_bidir25"]=0
        independent["raw_pose_predictions"]["near7_bidir25"]=22
        self.assertEqual(dev24_decision.evaluate(independent)["decision"],"FOLLOWUP_MAY_BE_CONSIDERED")

    def test_accounting_distinguishes_actual_failure_and_other_namespaces(self):
        intent=dict(request=launch.REQUEST,publication="f"*40,parent=launch.PARENT,node=launch.NODE)
        rows={phase:[f"{launch.PARENT}.{1300+i}",f"{launch.LABEL}-{phase}",
                     "COMPLETED","0:0","3",launch.NODE,"cpu=2,node=1"]
              for i,phase in enumerate(audit_terminal.PHASES)}
        outcome=dict(request=launch.REQUEST,publication="f"*40,stages={phase:{"row":row}
                                                                     for phase,row in rows.items()})
        self.assertTrue(audit_terminal.accounting_rows(intent,outcome,rows))
        rows["gpu"][2:4]=["FAILED","1:0"]
        self.assertFalse(audit_terminal.accounting_rows(intent,outcome,rows))
        with self.assertRaisesRegex(ValueError,"original post"):
            audit_terminal.accounting_rows(intent,outcome,{**rows,"post":rows["gpu"]})



if __name__=="__main__":
    unittest.main()
