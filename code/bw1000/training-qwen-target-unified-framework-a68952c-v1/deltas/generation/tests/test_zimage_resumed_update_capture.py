"""Real CPU regression for original loss/update observers at global21."""
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_zimage_first_update_capture import original_loss,MiniModel
from gencuda.backends import zimage_training as backend
from gencuda.zimage_first_update import instrument_update
from gencuda.zimage_resumed_update import ZImageResumedUpdateRecorder,recorded_resumed_update
from gencuda import hy_resume_checkpoint as common


class ResumedCapture(unittest.TestCase):
    def fixture(self):
        native=types.SimpleNamespace(FlowMatchSFTLoss=original_loss())
        torch.manual_seed(918);model=MiniModel(native);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        progress=common.initial_optimizer_progress(model,opt)
        with patch.object(backend,'require_cuda',return_value=torch),patch.object(backend,'synchronize',return_value=None):
            for step in range(1,21):
                backend._train_step(model,opt,dict(input_latents=torch.arange(8).reshape(2,4).float()/8),900+step,1.)
                progress=common.advance_optimizer_progress(progress,model,opt)
        saved=common.capture_model_state(model,opt,progress);rng=torch.get_rng_state().clone()
        resumed=MiniModel(native);other=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=.01)
        common.validate_model_state(saved,resumed,other,20);common._copy_model_state(saved,resumed,other)
        self.assertEqual(common.tree_sha256(saved),common.tree_sha256(common.capture_model_state(resumed,other,saved['optimizer_progress'])))
        torch.set_rng_state(rng);return native,resumed,other,saved['optimizer_progress']

    def run21(self,path,record):
        native,model,opt,progress=self.fixture();original=native.FlowMatchSFTLoss
        epoch=time.time();rec=ZImageResumedUpdateRecorder(path,list(model.named_parameters()),opt,epoch=epoch,deadline=epoch+60,device_ids=[],binding=dict(CPU_fixture=True)) if record else None
        update=instrument_update(backend._train_step) if record else backend._train_step
        with patch.object(backend,'require_cuda',return_value=torch),patch.object(backend,'synchronize',return_value=None):
            if record:update.__globals__.update(require_cuda=lambda:torch,synchronize=lambda:None)
            data=dict(input_latents=torch.arange(8).reshape(2,4).float()/8)
            value=recorded_resumed_update(update,native,rec,21,model,opt,data,921,1.) if record else update(model,opt,data,921,1.)
        progress=common.advance_optimizer_progress(progress,model,opt)
        if rec:rec.guard(rec.finish,21,True)
        self.assertIs(native.FlowMatchSFTLoss,original)
        return value['loss'],common.tree_sha256(common.capture_model_state(model,opt,progress)),torch.get_rng_state(),rec

    def test_actual21_capture_preserves_loss_full_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.run21(Path(temp)/'plain',False);b=self.run21(Path(temp)/'recorded',True)
            self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))
            rec=b[3];self.assertEqual(rec.current_step,21);self.assertEqual(rec.completed_steps,21);self.assertTrue(rec.first_complete)
            self.assertEqual(rec.phase,'RESUMED21_RETURNED');self.assertEqual(len(rec.files),12)
            initial=torch.load(rec.root/'initial_optimizer.pt',weights_only=True)
            final=torch.load(rec.root/'optimizer.pt',weights_only=True)
            self.assertTrue(all(float(v['step'])==20 for v in initial['state'].values()))
            self.assertTrue(all(float(v['step'])==21 for v in final.values()))
            p=json.loads((rec.root/'progress.json').read_bytes());self.assertEqual(p['completed_global_steps'],[21])

    def test_non20_optimizer_and_wrong_globalstep_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            native,model,opt,progress=self.fixture();epoch=time.time()
            for state in opt.state.values():state['step'].fill_(19)
            with self.assertRaisesRegex(ValueError,'completed20'):ZImageResumedUpdateRecorder(Path(temp)/'bad',list(model.named_parameters()),opt,epoch=epoch,deadline=epoch+60,device_ids=[],binding={})
            self.assertFalse((Path(temp)/'bad').exists())
            for state in opt.state.values():state['step'].fill_(20)
            rec=ZImageResumedUpdateRecorder(Path(temp)/'valid',list(model.named_parameters()),opt,epoch=epoch,deadline=epoch+60,device_ids=[],binding={})
            with self.assertRaisesRegex(ValueError,'global21'):rec.guard(rec.enter,22,{})
            self.assertTrue(rec.failed);self.assertEqual(rec.completed_steps,20);self.assertFalse(rec.first_complete)

    def test_mutated_native_input_keeps_actual21_failure_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            native,model,opt,progress=self.fixture();model.mutate=True;epoch=time.time();original=native.FlowMatchSFTLoss
            rec=ZImageResumedUpdateRecorder(Path(temp)/'recorded',list(model.named_parameters()),opt,epoch=epoch,deadline=epoch+60,device_ids=[],binding={})
            update=instrument_update(backend._train_step);update.__globals__.update(require_cuda=lambda:torch,synchronize=lambda:None)
            with self.assertRaisesRegex(ValueError,'mutated'):
                recorded_resumed_update(update,native,rec,21,model,opt,dict(input_latents=torch.ones(2,4)),921,1.)
            self.assertIs(native.FlowMatchSFTLoss,original)
            p=json.loads((rec.root/'progress.json').read_bytes());self.assertTrue(p['failed']);self.assertEqual(p['planned_global_steps'],[21]);self.assertEqual(p['completed_global_steps'],[])


if __name__=='__main__':unittest.main()
