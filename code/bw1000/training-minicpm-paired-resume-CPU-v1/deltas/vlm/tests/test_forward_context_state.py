"""CPU checks for ambient autograd/AMP and nested training-mode observations."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from forward_context_state import context_snapshot,training_mode_changes,input_layouts

class ForwardContextTest(unittest.TestCase):
    def test_root_training_does_not_imply_child_training(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Sequential(torch.nn.Linear(4,2)))
        model[1].eval()
        before=context_snapshot(model,device_type='cpu')
        self.assertTrue(model.training);self.assertFalse(model[1].training)
        model.train()
        after=context_snapshot(model,device_type='cpu')
        self.assertEqual([x['name'] for x in training_mode_changes(before,after)],['1','1.0'])
        self.assertEqual(after['training_modules'],after['total_modules'])

    def test_records_ambient_context_without_changing_parameters(self):
        model=torch.nn.Linear(3,2);original={k:v.detach().clone() for k,v in model.state_dict().items()}
        default=context_snapshot(model,device_type='cpu')
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            active=context_snapshot(model,device_type='cpu')
            self.assertFalse(active['grad_enabled']);self.assertTrue(active['autocast_enabled'])
        after=context_snapshot(model,device_type='cpu')
        self.assertEqual(default['identity_SHA256'],after['identity_SHA256'])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()))

    def test_equal_values_can_have_different_layouts(self):
        view=torch.arange(24).reshape(4,6)[:,::2]
        cloned=view.clone()
        self.assertTrue(torch.equal(view,cloned))
        a=input_layouts((),{'input':view})['kwargs/input']
        b=input_layouts((),{'input':cloned})['kwargs/input']
        self.assertEqual(a['shape'],b['shape']);self.assertEqual(a['dtype'],b['dtype'])
        self.assertNotEqual(a['stride'],b['stride'])
        self.assertNotIn('values',a)

    def test_rejects_changed_inventory(self):
        before=context_snapshot(torch.nn.Linear(2,3),device_type='cpu')
        after=context_snapshot(torch.nn.Sequential(torch.nn.Linear(2,3)),device_type='cpu')
        with self.assertRaises(ValueError):training_mode_changes(before,after)

if __name__=='__main__':unittest.main()
