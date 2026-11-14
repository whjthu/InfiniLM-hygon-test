"""Real CPU full-state checkpoint identity rejects silent policy changes."""
from pathlib import Path
import sys,types,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gencuda import hy_resume_checkpoint as common,zimage_training_resume_checkpoint as resume
from gencuda.zimage_attention_policy import POLICY_ID,policy_for_config


class PolicyCheckpoint(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(71);m=torch.nn.Linear(2,2);o=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        progress=common.initial_optimizer_progress(m,o)
        for _ in range(20):
            o.zero_grad(set_to_none=True);m(torch.ones(1,2)).square().sum().backward();o.step();progress=common.advance_optimizer_progress(progress,m,o)
        scheduler=types.SimpleNamespace(timesteps=torch.arange(8));spec=resume.scheduler_spec(scheduler)
        identity={k:common.json_sha256(k) for k in resume.IDENTITY_FIELDS}
        base=resume.build_contract(identity=identity,seed=71,max_steps=25,sample_order=['original-row-'+str(n) for n in range(20)],device_ids=[])
        changed=dict(identity,config=common.json_sha256(dict(attention_backward_policy=POLICY_ID)),runtime=common.json_sha256(policy_for_config(dict(attention_backward_policy=POLICY_ID))))
        candidate=resume.build_contract(identity=changed,seed=71,max_steps=25,sample_order=base['sample_order'],device_ids=[])
        payload=resume.make_payload(contract=base,global_step=20,model=m,optimizer=o,optimizer_progress=progress,scheduler=scheduler,expected_scheduler=spec)
        return m,o,progress,scheduler,spec,base,candidate,payload

    def test_baseline_cannot_silently_restore_as_candidate(self):
        m,o,progress,scheduler,spec,base,candidate,payload=self.fixture();before=common.tree_sha256(common.capture_model_state(m,o,progress));rng=common.capture_rng([])
        with self.assertRaises(resume.CheckpointValidationError):resume.restore_checkpoint(payload,candidate,m,o,scheduler,spec)
        self.assertEqual(before,common.tree_sha256(common.capture_model_state(m,o,progress)));self.assertEqual(common.tree_sha256(rng),common.tree_sha256(common.capture_rng([])))

    def test_derived_candidate_checkpoint_requires_candidate_policy(self):
        m,o,progress,scheduler,spec,base,candidate,payload=self.fixture()
        progress,_=resume.restore_checkpoint(payload,base,m,o,scheduler,spec);resume.assert_restored_boundary(payload,m,o,progress,scheduler,[])
        o.zero_grad(set_to_none=True);m(torch.ones(1,2)).square().sum().backward();o.step();progress=common.advance_optimizer_progress(progress,m,o)
        derived=resume.make_payload(contract=candidate,global_step=21,model=m,optimizer=o,optimizer_progress=progress,scheduler=scheduler,expected_scheduler=spec)
        with self.assertRaises(resume.CheckpointValidationError):resume.restore_checkpoint(derived,base,m,o,scheduler,spec)
        progress,_=resume.restore_checkpoint(derived,candidate,m,o,scheduler,spec);resume.assert_restored_boundary(derived,m,o,progress,scheduler,[])
        self.assertEqual(derived['metadata']['global_step'],21)

if __name__=='__main__':unittest.main()
