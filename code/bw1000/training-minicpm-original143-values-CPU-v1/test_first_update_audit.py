import unittest

import torch

from vlm.runtime.first_update_audit import fingerprint, optimizer_steps, parameter_delta, summarize_update


class FirstUpdateAuditTest(unittest.TestCase):
    def test_fingerprint_is_key_order_invariant_and_dtype_sensitive(self):
        a = {'weight': torch.tensor([1., 2.]), 'other': None}
        b = {'other': None, 'weight': torch.tensor([1., 2.])}
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertNotEqual(fingerprint(a)['SHA256'],
                            fingerprint(dict(a, weight=a['weight'].to(torch.bfloat16)))['SHA256'])

    def test_nonfinite_and_unsupported_values_are_rejected(self):
        for value in (torch.tensor([float('nan')]), float('inf'), object()):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                fingerprint(value)

    def test_warmup_zero_change_and_nonzero_update_remain_distinct(self):
        initial = {'weight': torch.tensor([1., 2.])}
        self.assertEqual(parameter_delta(initial, initial)['changed_elements'], 0)
        self.assertEqual(parameter_delta(initial, {'weight': torch.tensor([1., 3.])})['changed_elements'], 1)
        before = dict(global_step=0, parameters=initial, optimizer={
            'state_by_name': {'weight': {}}, 'param_groups': [{'parameter_names': ['weight'], 'lr': 0.}]})
        after = dict(global_step_before_increment=0, parameters=initial,
                     optimizer={'state_by_name': {'weight': {'step': torch.tensor(1.)}}})
        preclip = dict(global_step=0, gradients={'weight': torch.tensor([.1, .2])})
        postclip = dict(preclip, reported_preclip_norm=torch.tensor(.2236068))
        result = summarize_update(before, after, preclip, postclip, ordinal=1)
        self.assertEqual(result['optimizer_after_steps']['minimum'], 1)
        self.assertEqual(result['parameter_update']['changed_elements'], 0)

    def test_bad_parameter_inventory_and_optimizer_step_are_rejected(self):
        with self.assertRaises(ValueError):
            parameter_delta({'a': torch.ones(1)}, {'b': torch.ones(1)})
        for step in (float('nan'), -1, 1.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                optimizer_steps({'state_by_name': {'weight': {'step': step}}})


if __name__ == '__main__':
    unittest.main()
