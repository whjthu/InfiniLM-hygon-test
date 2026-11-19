"""Exercise the installed Swift optimizer callback and actual Trainer factory."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from swift.optimizers import OptimizerCallback, optimizers_map
from transformers import Trainer, TrainingArguments

from vlm.plugins.grouped_adamw import KEY
from vlm.runtime.grouped_fused_adamw import GroupedFusedAdamW


def fixture(path, grouped):
    torch.manual_seed(1124)
    model = torch.nn.Sequential(*(torch.nn.Linear(2, 2) for _ in range(70)))
    args = TrainingArguments(output_dir=str(path), use_cpu=True, optim="adamw_torch",
                             weight_decay=.1, learning_rate=.001, warmup_steps=1,
                             lr_scheduler_type="cosine", report_to=[])
    trainer = Trainer(model=model, args=args)
    trainer.args.bf16 = True
    cls = optimizers_map[KEY] if grouped else OptimizerCallback
    callback = cls(trainer.args, trainer)
    callback.create_optimizer_and_scheduler(25)
    return model, trainer


def group_names(model, optimizer):
    lookup = {id(tensor): name for name, tensor in model.named_parameters()}
    return [([lookup[id(p)] for p in group["params"]], group["weight_decay"])
            for group in optimizer.param_groups]


class NativeSwiftFactoryTests(unittest.TestCase):
    def test_actual_swift_factory_parameter_groups_and_updates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_a, a = fixture(root / "baseline", False)
            model_b, b = fixture(root / "candidate", True)
            self.assertIs(type(a.optimizer), torch.optim.AdamW)
            self.assertIs(type(b.optimizer), GroupedFusedAdamW)
            self.assertIsNone(b.optimizer_cls_and_kwargs)
            self.assertEqual(group_names(model_a, a.optimizer), group_names(model_b, b.optimizer))
            self.assertEqual([len(g["params"]) for g in b.optimizer.param_groups], [70, 70])
            self.assertIs(a.scheduler.optimizer, a.optimizer)
            self.assertIs(b.scheduler.optimizer, b.optimizer)
            for ordinal in range(3):
                for model, trainer in ((model_a, a), (model_b, b)):
                    for index, parameter in enumerate(model.parameters()):
                        parameter.grad = torch.full_like(parameter, (ordinal + index + 1) / 1000)
                    trainer.optimizer.step()
                    trainer.scheduler.step()
                    trainer.optimizer.zero_grad(set_to_none=True)
                self.assertEqual(a.scheduler.get_last_lr(), b.scheduler.get_last_lr())
            for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
                torch.testing.assert_close(parameter_a, parameter_b, atol=1e-7, rtol=1e-6)
                for moment in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(a.optimizer.state[parameter_a][moment],
                                               b.optimizer.state[parameter_b][moment],
                                               atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
