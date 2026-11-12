"""Real CPU AdamW evidence that capture preserves native loss/update math."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gencuda.zimage_first_update import ZImageFirstUpdateRecorder, instrument_loss, instrument_update, recorded_native_update
from gencuda.backends import zimage_training as backend
from gencuda import hy_resume_checkpoint as common


def original_loss():
    path=Path(os.environ['ZIMAGE_ORIGINAL_LOSS_SOURCE'])
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='523b00ccfd815c2f4c07434348ebb79ad294dfea325171b80a452db2c9943de5':
        raise ValueError('original loss source differs')
    fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='FlowMatchSFTLoss')
    module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    scope=dict(torch=torch,BasePipeline=object);exec(compile(module,str(path),'exec'),scope)
    return scope['FlowMatchSFTLoss']


class Scheduler:
    timesteps=torch.arange(1,9,dtype=torch.float32)
    def add_noise(self,latents,noise,timestep):return latents*.7+noise*.3
    def training_target(self,latents,noise,timestep):return noise-latents
    def training_weight(self,timestep):return .8


class MiniModel(torch.nn.Module):
    def __init__(self,native,mutate=False):
        super().__init__();self.layer=torch.nn.Linear(4,4);self.native=native;self.mutate=mutate
        self.pipe=types.SimpleNamespace(scheduler=Scheduler(),torch_dtype=torch.float32,device='cpu',in_iteration_models=('dit',),dit=self.layer,model_fn=self.denoise)
    def denoise(self,dit,latents,**inputs):
        value=dit(latents)
        if self.mutate:latents.add_(1)
        return value
    def forward(self,data):return self.native.FlowMatchSFTLoss(self.pipe,**data)


class RealCPUCapture(unittest.TestCase):
    def fixture(self,path,mutate=False,limit=1024**2):
        native=types.SimpleNamespace(FlowMatchSFTLoss=original_loss())
        torch.manual_seed(918);model=MiniModel(native,mutate)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        epoch=time.time();rec=ZImageFirstUpdateRecorder(path,list(model.named_parameters()),opt,
             epoch=epoch,deadline=epoch+60,device_ids=[],binding=dict(CPU_fixture=True),artifact_limit=limit,total_limit=4*1024**2)
        return native,model,opt,rec

    def run_updates(self,path,recorded):
        native,model,opt,rec=self.fixture(path);values=[]
        update=instrument_update(backend._train_step) if recorded else backend._train_step
        original=native.FlowMatchSFTLoss
        with patch.object(backend,'require_cuda',return_value=torch),patch.object(backend,'synchronize',return_value=None):
            # Compiled functions captured globals before the patch;bind only the
            # explicit CPU fixture providers,never alter the mathematical body.
            if recorded:update.__globals__.update(require_cuda=lambda:torch,synchronize=lambda:None)
            for step in range(1,26):
                data=dict(input_latents=torch.arange(8).reshape(2,4).float()/8)
                values.append(recorded_native_update(update,native,rec if recorded else None,step,model,opt,data,900+step,1)['loss'])
        if recorded:rec.guard(rec.finish,25,True)
        self.assertIs(native.FlowMatchSFTLoss,original)
        return values,{n:p.detach().clone() for n,p in model.named_parameters()},common.tree_sha256(opt.state_dict()),torch.get_rng_state(),rec

    def test_full25_exact_loss_parameters_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=self.run_updates(p/'plain',False);b=self.run_updates(p/'recorded',True)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[2],b[2]);self.assertTrue(torch.equal(a[3],b[3]))
            self.assertTrue(all(torch.equal(v,b[1][n]) for n,v in a[1].items()))
            rec=b[4];self.assertEqual(rec.completed_steps,25);self.assertTrue(rec.first_complete)
            self.assertEqual(rec.phase,'NATIVE25_RETURNED');self.assertEqual(len(rec.files),12)
            for row in rec.files:
                raw=(rec.root/row['name']).read_bytes();self.assertEqual(len(raw),row['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['SHA256'])
                torch.load(rec.root/row['name'],map_location='cpu',weights_only=True)
            prepared=torch.load(rec.root/'prepared.pt',weights_only=True)
            self.assertTrue(torch.equal(prepared['inputs']['latents'],prepared['inputs']['input_latents']*.7+prepared['noise']*.3))
            target=torch.load(rec.root/'target.pt',weights_only=True)['training_target']
            self.assertTrue(torch.equal(target,prepared['noise']-prepared['inputs']['input_latents']))
            initial=torch.load(rec.root/'initial.pt',weights_only=True);updated=torch.load(rec.root/'updated.pt',weights_only=True)
            self.assertTrue(any(not torch.equal(v,updated[n]) for n,v in initial.items()))

    def test_prepared_input_mutation_preserves_failure_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            native,model,opt,rec=self.fixture(Path(temp)/'recorded',mutate=True);original=native.FlowMatchSFTLoss
            update=instrument_update(backend._train_step);update.__globals__.update(require_cuda=lambda:torch,synchronize=lambda:None)
            with self.assertRaisesRegex(ValueError,'mutated'):
                recorded_native_update(update,native,rec,1,model,opt,dict(input_latents=torch.ones(2,4)),901,1)
            self.assertIs(native.FlowMatchSFTLoss,original);p=json.loads((rec.root/'progress.json').read_bytes())
            self.assertTrue(p['failed']);self.assertEqual(p['planned_global_steps'],25);self.assertEqual(p['completed_global_steps'],0);self.assertFalse(p['first_update_complete'])

    def test_file_budget_failure_restores_loss_and_keeps_initial_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            native,model,opt,rec=self.fixture(Path(temp)/'recorded',limit=32);original=native.FlowMatchSFTLoss
            update=instrument_update(backend._train_step);update.__globals__.update(require_cuda=lambda:torch,synchronize=lambda:None)
            with self.assertRaises(ValueError):recorded_native_update(update,native,rec,1,model,opt,dict(input_latents=torch.ones(2,4)),901,1)
            self.assertIs(native.FlowMatchSFTLoss,original);p=json.loads((rec.root/'progress.json').read_bytes())
            self.assertTrue(p['failed']);self.assertEqual(p['completed_global_steps'],0);self.assertEqual(p['planned_global_steps'],25)


if __name__=='__main__':unittest.main()
