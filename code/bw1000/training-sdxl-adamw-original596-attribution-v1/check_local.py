"""Check archived identity closure and CPU decomposition with small fixtures."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import read_original as audit

PUBLIC=HERE.parents[2]/"results/bw1000/2026-11-22/training-sdxl-sdpa-adjoint-original596-numeric-failure.json"


class Local(unittest.TestCase):
    def test_original_public_decision_and_gate_arithmetic(self):
        actual=json.loads(audit.pinned(PUBLIC,audit.PUBLIC,3739,3739))
        update=actual["metrics"]["parameter_update"]
        need=1-(.05/update["relative_l2"])**2
        self.assertAlmostEqual(need,.9335692561504141)
        self.assertEqual(actual["parameter_counts"]["parameter_update"],1120)
        self.assertEqual(actual["metrics"]["gradients_before_clip"],actual["metrics"]["gradients_after_clip"])
        helper=HERE.parent/"training-sdxl-adamw-first-step-counterfactual-v1/worker.py"
        self.assertGreater(len(audit.pinned(helper,audit.HELPER,32768)),0)

    def test_three_term_result_retains_signed_cross_terms_and_denominator(self):
        data=audit.zeros()
        data.update(native=1.5625,reference_native=4,sign_CPU=.25,opposite_native=1.0,
                    gram=[[1,.5,-.25],[.5,.25,-.125],[-.25,-.125,.0625]])
        report=audit.finalize(data,.05)
        self.assertAlmostEqual(report["native_relative_l2"],.625)
        self.assertAlmostEqual(report["clipped_gradient_only_relative_l2"],.75)
        self.assertAlmostEqual(report["cpu_sign_only_relative_l2"],.25)
        self.assertAlmostEqual(report["component_gram_over_reference_squared"][0][2],-.0625)
        self.assertAlmostEqual(report["native_error_squared_share_opposite_sign"],.64)
        with self.assertRaisesRegex(ValueError,"does not close"):
            audit.finalize(dict(data,native=1),.05)

    def test_archive_hash_and_exact_size_precede_weights_only_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"observed.pt"
            raw=b"frozen fixture bytes"
            path.write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            torch=Mock()
            with self.assertRaisesRegex(ValueError,"size"):
                audit.archive(torch,path,len(raw)-1,digest)
            with self.assertRaisesRegex(ValueError,"archive bytes"):
                audit.archive(torch,path,len(raw),"0"*64)
            torch.load.assert_not_called()
            audit.archive(torch,path,len(raw),digest)
            torch.load.assert_called_once_with(path,map_location="cpu",weights_only=True,mmap=True)

    def test_bound_current_comparator_witness_and_reject_unbound_archive(self):
        proof=json.loads(audit.pinned(PUBLIC,audit.PUBLIC,3739,3739))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run=root/"run"
            replay=run/"candidate/replay"
            replay.mkdir(parents=True)
            ref=root/"reference"
            ref.mkdir()

            def save(path, data):
                raw=json.dumps(data,sort_keys=True).encode()
                path.write_bytes(raw)
                return hashlib.sha256(raw).hexdigest(),len(raw)

            policy={"stage_id":"sdxl-sft","reference_payload_sha256":audit.REFERENCE,
                    "metrics":{"parameter_update_relative_l2_max":.05}}
            policy_sha,policy_size=save(ref/"policy.json",policy)
            terminal={"request":audit.REQUEST,"step":"1087656.596","comparison_complete":True,
                      "gpu_passed":True,"clean_post":True,
                      "runs":[{"terminal":{"row":row}} for row in proof["job_rows"]]}
            terminal_sha,_=save(run/"terminal.json",terminal)
            common=dict(stage_id="sdxl-sft",reference_sha256=audit.REFERENCE,policy_sha256=policy_sha,
                        boundary="fixed_realized_inputs")
            observed="7"*64
            summary=dict(common,observed_sha256=observed,replay_source_sha256=audit.REPLAY,
                         runtime=dict(accelerator="hip",cuda_count=1,cuda_available=True))
            summary_sha,summary_size=save(replay/"summary.json",summary)
            comparison=dict(common,comparator_sha256=audit.COMPARATOR,accelerator="hip",
                            passed=False,checks=proof["checks"],replay_receipt_sha256=summary_sha,
                            observed_sha256=observed,metrics={
                                k:dict(v,per_tensor={str(n):{} for n in range(1120)})
                                for k,v in proof["metrics"].items()})
            comp_sha,comp_size=save(run/"comparison.json",comparison)
            proof.update(terminal_SHA256=terminal_sha,full_private_comparison_SHA256=comp_sha,native_summary_sha256=summary_sha)
            actual_pin=audit.pinned
            def fixture_public(path,digest,limit,size=None):
                if path==PUBLIC:
                    return json.dumps(proof).encode()
                return actual_pin(path,digest,limit,size)
            with patch.multiple(audit,POLICY=policy_sha,POLICY_BYTES=policy_size,SUMMARY=summary_sha,
                                SUMMARY_BYTES=summary_size,OBSERVED=observed,COMPARISON_BYTES=comp_size,pinned=fixture_public):
                path,_,gate=audit.close_evidence(run,PUBLIC,ref)
                self.assertEqual((path,gate),(replay/"observed.pt",.05))
                self.assertFalse(path.exists())
                save(replay/"summary.json",dict(summary,observed_sha256="0"*64))
                with self.assertRaisesRegex(ValueError,"identity mismatch: summary"):
                    audit.close_evidence(run,PUBLIC,ref)


if __name__=="__main__":
    unittest.main(verbosity=2)
