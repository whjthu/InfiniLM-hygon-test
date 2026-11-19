"""Fused AdamW list partitioning with native Swift checkpoint groups."""
from torch import enable_grad, no_grad
from torch.optim import AdamW
from torch.optim.adam import adam


class GroupedFusedAdamW(AdamW):
    """Keep scheduler/state-dict groups intact and partition only kernel inputs."""

    def __init__(self, params, *, tensor_group_width, **kwargs):
        if type(tensor_group_width) is not int or tensor_group_width not in (32, 64, 128):
            raise ValueError("grouped AdamW requires a measured tensor group width")
        if kwargs.pop("fused", None) not in (None, True) or kwargs.pop("foreach", None) not in (None, False):
            raise ValueError("grouped AdamW requires the native fused backend")
        self.tensor_group_width = tensor_group_width
        super().__init__(params, fused=True, foreach=False, **kwargs)
        self._check_groups(self.param_groups)

    @staticmethod
    def _check_groups(groups):
        for group in groups:
            if (group.get("fused") is not True or group.get("foreach") is not False
                    or group.get("decoupled_weight_decay") is not True
                    or any(group.get(k) for k in ("capturable", "differentiable", "amsgrad", "maximize"))):
                raise ValueError("grouped AdamW requires eager, non-differentiable, decoupled fused groups")
            if any(type(value) not in (int, float) for value in (group["lr"], *group["betas"])):
                raise ValueError("grouped AdamW requires scalar learning rate and betas")

    def load_state_dict(self, state_dict):
        self._check_groups(state_dict["param_groups"])
        return super().load_state_dict(state_dict)

    @no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        self._check_groups(self.param_groups)
        loss = None
        if closure is not None:
            with enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameters, gradients, averages, squares, maximums, counters = [], [], [], [], [], []
            has_complex = self._init_group(group, parameters, gradients, averages, squares, maximums, counters)
            if has_complex:
                raise ValueError("grouped AdamW requires real trainable parameters")
            beta1, beta2 = group["betas"]
            width = self.tensor_group_width
            for start in range(0, len(parameters), width):
                chunk = slice(start, start + width)
                adam(parameters[chunk], gradients[chunk], averages[chunk], squares[chunk], [],
                     counters[chunk], amsgrad=False, has_complex=False, beta1=beta1, beta2=beta2,
                     lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
                     maximize=False, foreach=False, capturable=False, differentiable=False, fused=True,
                     grad_scale=getattr(self, "grad_scale", None), found_inf=getattr(self, "found_inf", None),
                     decoupled_weight_decay=True)
        return loss
