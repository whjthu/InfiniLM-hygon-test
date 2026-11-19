"""Register an opt-in optimizer in the pinned native Swift entry."""
import hashlib
import importlib
from pathlib import Path

import torch
from swift.optimizers import OptimizerCallback, optimizers_map

from vlm.runtime.grouped_fused_adamw import GroupedFusedAdamW


PROVIDERS = {
    "torch.optim.adam": "691246e104a54c3ff985922a4f551aea088c06c3358a18861e59c34a920d21e1",
    "torch.optim.adamw": "54299056b7745c162192132bb6028f3387c05ff4203518ff0240058584968312",
    "swift.optimizers.base": "ecaba5a3e1cf8c3bc18bcfbe5856d319e4d228fa77cead7570189bb71b51b25b",
    "swift.trainers.mixin": "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9",
    "transformers.trainer": "c1a56423fcfcf9cfec6847467ffb2e2c8a9a9e8cc1836b82c87ed0c81e504be0",
}


def verify_providers():
    for module, expected in PROVIDERS.items():
        provider = importlib.import_module(module)
        if hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError(f"grouped AdamW provider differs: {module}")


class GroupedAdamWCallback(OptimizerCallback):
    tensor_group_width = 64

    def create_optimizer(self, model=None):
        trainer = self.trainer
        args = trainer.args
        if not (torch.version.hip and args.bf16 and not args.fp16
                and not getattr(args, "deepspeed", None) and not getattr(args, "fsdp", None)):
            raise ValueError("grouped AdamW candidate requires the original HIP BF16 replicated recipe")
        if trainer.optimizer is not None:
            if not isinstance(trainer.optimizer, GroupedFusedAdamW):
                raise ValueError("trainer has a different optimizer")
            return trainer.optimizer
        verify_providers()
        if trainer.optimizer_cls_and_kwargs is not None:
            raise ValueError("grouped AdamW requires the original optimizer selection")
        opt_model = trainer.model if model is None else model
        cls, kwargs = trainer.get_optimizer_cls_and_kwargs(args, opt_model)
        if cls is not torch.optim.AdamW or {"params", "model", "optimizer_dict"}.intersection(kwargs):
            raise ValueError("installed Trainer did not select ordinary native AdamW")
        options = dict(kwargs, tensor_group_width=self.tensor_group_width)
        trainer.optimizer_cls_and_kwargs = (GroupedFusedAdamW, options)
        try:
            return super().create_optimizer(model=model)
        finally:
            trainer.optimizer_cls_and_kwargs = None


KEY = "hygon-grouped-adamw-64"
if KEY in optimizers_map:
    raise ValueError("grouped AdamW optimizer key is already registered")
optimizers_map[KEY] = GroupedAdamWCallback
