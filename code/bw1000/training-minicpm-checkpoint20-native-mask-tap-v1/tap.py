"""Observe full native LoRA dropout masks without replacing any sampler output."""

from hashlib import sha256
from sys import exc_info
import time
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode


EXPECTED_DROPOUTS = 186


def require(ok, message):
    if not ok:
        raise ValueError(message)


class NativeMaskMode(TorchDispatchMode):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.mask = None

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        name = operation._schema.name
        result = operation(*args, **(kwargs or {}))
        if name == "aten::native_dropout":
            require(self.mask is None and len(result) == 2 and args[0] is self.value,
                    "expected exactly one native dropout of the original input")
            mask = result[1]
            require(mask.dtype == torch.bool and tuple(mask.shape) == tuple(self.value.shape)
                    and mask.device == self.value.device and not mask.requires_grad,
                    "native returned mask metadata differs")
            self.mask = mask
        elif name == "aten::bernoulli_":
            raise ValueError("native bool-mask path required")
        return result


class MaskTap:
    """One real forward, all ordered LoRA masks, and no host copies during compute."""

    def __init__(self, model, *, max_retained_bytes, expected_calls=EXPECTED_DROPOUTS):
        require(type(max_retained_bytes) is int and max_retained_bytes > 0
                and type(expected_calls) is int and expected_calls > 0, "invalid mask retention scope")
        self.bound = [(name, module) for name, module in model.named_modules() if ".lora_dropout." in name]
        self.count = expected_calls
        require(len(self.bound) == self.count and len({id(module) for _, module in self.bound}) == self.count,
                "LoRA dropout module inventory differs")
        for _, module in self.bound:
            require(type(module) is torch.nn.Dropout and module.p == .05 and not module.inplace
                    and module.training and "forward" not in vars(module)
                    and getattr(module.forward, "__func__", None) is torch.nn.Dropout.forward,
                    "LoRA dropout is not the unmodified p=.05 native module")
        self.limit = max_retained_bytes
        self.retained_bytes = 0
        self.records = []
        self.names = set()
        self.storage_ids = set()
        self.installed = []
        self.state = "new"

    def _patch(self, name, module):
        original = module.forward

        def forward(owner, value):
            require(self.state == "active" and name not in self.names and owner.training
                    and owner.p == .05 and not owner.inplace, "native dropout call or module mode differs")
            with NativeMaskMode(value) as observe:
                out = original(value)
            mask = observe.mask
            require(mask is not None, "forward emitted no observable native bool mask")
            storage = mask.untyped_storage()
            charge = storage.nbytes()
            identity = (str(mask.device), storage.data_ptr())
            require(identity not in self.storage_ids and charge >= mask.numel() > 0,
                    "native mask storage identity or extent differs")
            require(self.retained_bytes + charge <= self.limit, "native mask retention budget exhausted")
            self.retained_bytes += charge
            self.names.add(name)
            self.storage_ids.add(identity)
            self.records.append((name, mask))
            return out

        module.forward = types.MethodType(forward, module)
        self.installed.append(module)

    def __enter__(self):
        require(self.state == "new", "mask tap is single-use")
        self.state = "active"
        try:
            for name, module in self.bound:
                self._patch(name, module)
        except BaseException:
            self.__exit__(*exc_info())
            raise
        return self

    def __exit__(self, kind, _error, _trace):
        for module in self.installed:
            del module.forward
        self.installed.clear()
        self.state = "failed" if kind else "returned"
        if kind:
            self.records.clear()

    def seal(self):
        require(self.state == "returned", "original forward did not return")
        if len(self.records) != self.count:
            self.state = "failed"
            self.records.clear()
            raise ValueError("native mask inventory incomplete")
        self.state = "sealed"
        rows = []
        started = time.monotonic()
        try:
            for ordinal, (name, mask) in enumerate(self.records):
                cpu = mask.detach().cpu().contiguous()
                raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                require(len(raw) == mask.numel(), "complete CPU mask has wrong size")
                rows.append({"module": name, "ordinal": ordinal, "shape": list(mask.shape),
                             "mask_stride": list(mask.stride()),
                             "mask_SHA256": sha256(raw).hexdigest(),
                             "mask_bytes": len(raw), "provider": "aten::native_dropout"})
        finally:
            self.records.clear()
        return {"native_mask_calls": self.count, "charged_device_storage_bytes": self.retained_bytes,
                "cap_device_storage_bytes": self.limit, "mask_fingerprints": rows,
                "logical_mask_bytes_moved_to_host": sum(row["mask_bytes"] for row in rows),
                "post_forward_copy_and_digest_seconds": time.monotonic() - started}


def compare(left, right):
    one, two = left["mask_fingerprints"], right["mask_fingerprints"]
    require(len(one) == len(two) == left["native_mask_calls"] == right["native_mask_calls"] == EXPECTED_DROPOUTS,
            "cannot compare incomplete native mask inventories")
    differences = []
    names = set()
    for ordinal, (a, b) in enumerate(zip(one, two, strict=True)):
        keys = ("module", "ordinal", "shape", "mask_stride", "mask_bytes", "provider")
        require(all(a[key] == b[key] for key in keys), "native mask call order or extent differs")
        require(a["ordinal"] == ordinal and a["module"] not in names and a["provider"] == "aten::native_dropout"
                and all(type(row["mask_SHA256"]) is str and len(row["mask_SHA256"]) == 64
                        and all(c in "0123456789abcdef" for c in row["mask_SHA256"]) for row in (a, b)),
                "duplicate native mask receipt or invalid identity")
        names.add(a["module"])
        if a["mask_SHA256"] != b["mask_SHA256"]:
            differences.append({"module": a["module"], "ordinal": a["ordinal"]})
    return {"compared_native_masks": EXPECTED_DROPOUTS, "different_native_masks": len(differences),
            "first_different_module": differences[0] if differences else None}
