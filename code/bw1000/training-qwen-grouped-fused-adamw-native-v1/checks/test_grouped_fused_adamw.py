"""Exercise the real native optimizer, scheduler and restart on CPU."""
import copy
import unittest
from unittest.mock import patch

import torch

from vlm.runtime import grouped_fused_adamw as grouped


def parameters():
    return [torch.nn.Parameter(torch.linspace(-.02, .05, 4) + index / 1000)
            for index in range(72)]


def training_pair(params, candidate):
    groups = [{"params": params[:68], "weight_decay": .1},
              {"params": params[68:], "weight_decay": 0.0}]
    factory = grouped.GroupedFusedAdamW if candidate else torch.optim.AdamW
    kwargs = {"tensor_group_width": 64} if candidate else {"fused": True, "foreach": False}
    optimizer = factory(groups, lr=.001, betas=(.9, .95), **kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    return optimizer, scheduler


def step(params, pair, index):
    optimizer, scheduler = pair
    optimizer.zero_grad(set_to_none=True)
    for ordinal, parameter in enumerate(params):
        if not (index > 0 and ordinal % 7 == 0):
            parameter.grad = torch.full_like(parameter, (index + ordinal + 1) / 1000)
    optimizer.step()
    scheduler.step()


def equal(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second, strict=True):
            equal(test, a, b)
    else:
        test.assertEqual(first, second)


class GroupedAdamWTests(unittest.TestCase):
    def test_native_function_list_width_and_scheduler_horizon(self):
        params = parameters()
        pair = training_pair(params, True)
        log = []
        pair[0].register_step_post_hook(lambda *_: log.append(True))
        with patch.object(grouped, "adam", wraps=grouped.adam) as native:
            step(params, pair, 0)
        self.assertEqual([len(call.args[0]) for call in native.call_args_list], [64, 4, 4])
        self.assertEqual(log, [True])
        self.assertEqual([len(group["params"]) for group in pair[0].param_groups], [68, 4])
        self.assertEqual(len(pair[1].get_last_lr()), 2)

    def test_native_fused_multistep_parity(self):
        base, candidate = parameters(), parameters()
        pair_a, pair_b = training_pair(base, False), training_pair(candidate, True)
        for index in range(4):
            step(base, pair_a, index)
            step(candidate, pair_b, index)
            equal(self, base, candidate)
            equal(self, pair_a[0].state_dict(), pair_b[0].state_dict())
            equal(self, pair_a[1].state_dict(), pair_b[1].state_dict())

    def test_fresh_optimizer_and_scheduler_recovery(self):
        continuous = parameters()
        original = training_pair(continuous, True)
        for index in range(2):
            step(continuous, original, index)
        resumed = [torch.nn.Parameter(p.detach().clone()) for p in continuous]
        restored = training_pair(resumed, True)
        saved_optimizer = copy.deepcopy(original[0].state_dict())
        saved_scheduler = copy.deepcopy(original[1].state_dict())
        restored[1].load_state_dict(saved_scheduler)
        restored[0].load_state_dict(saved_optimizer)
        for index in range(2, 4):
            step(continuous, original, index)
            step(resumed, restored, index)
        equal(self, continuous, resumed)
        equal(self, original[1].state_dict(), restored[1].state_dict())
        equal(self, original[0].state_dict(), restored[0].state_dict())

    def test_incompatible_group_refused_before_state_mutation(self):
        params = parameters()
        optimizer, _ = training_pair(params, True)
        before = copy.deepcopy(optimizer.state_dict())
        bad = copy.deepcopy(before)
        bad["param_groups"][1]["fused"] = False
        with self.assertRaises(ValueError):
            optimizer.load_state_dict(bad)
        equal(self, before, optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
