"""Record ordered native forward boundaries without retaining tensor bodies."""
from collections import Counter
from contextlib import contextmanager


def layout(dit):
    named = dict(dit.named_modules())
    core = ("noise_refiner", "context_refiner", "layers")
    if any(not hasattr(dit, group) or not len(getattr(dit, group)) for group in core):
        raise ValueError("Missing native refinement or transformer blocks")
    tracked = {"t_embedder", "cap_embedder"}
    for group in core:
        tracked.update(f"{group}.{index}" for index in range(len(getattr(dit, group))))
    for group in ("all_x_embedder", "all_final_layer"):
        members = getattr(dit, group, None)
        if not members:
            raise ValueError(f"Missing native {group} modules")
        tracked.update(f"{group}.{key}" for key in members)
    if getattr(dit, "siglip_embedder", None) is not None:
        tracked.add("siglip_embedder")
        tracked.update(f"siglip_refiner.{index}" for index in range(len(dit.siglip_refiner)))
    if not tracked <= named.keys():
        raise ValueError("Missing named forward boundaries")
    return {name: named[name] for name in sorted(tracked)}


@contextmanager
def observe(dit, tensor_type, tensor_digest, deadline, progress):
    modules = layout(dit)
    rows = []
    counts = Counter()
    hooks = []

    def signature(value):
        if isinstance(value, tensor_type):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "SHA256": tensor_digest(value)}
        if isinstance(value, (tuple, list)):
            return [signature(item) for item in value]
        if isinstance(value, dict):
            return {key: signature(value[key]) for key in sorted(value)}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError("Unexpected native forward boundary argument type")

    def hook(name):
        def record(_module, args, kwargs, output):
            deadline()
            counts[name] += 1
            limit = 2 if name == "t_embedder" else 1
            if counts[name] > limit:
                raise ValueError(f"Unexpected forward occurrence: {name} #{counts[name]}")
            row = {"boundary": name, "occurrence": counts[name], "args": signature(args),
                   "kwargs": signature(kwargs), "output": signature(output)}
            rows.append(row)
            progress(row)
        return record

    try:
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(hook(name), with_kwargs=True))
        yield rows
    finally:
        for handle in reversed(hooks):
            handle.remove()


def validate(rows, dit):
    counts = Counter(row["boundary"] for row in rows)
    specified = layout(dit)
    optional = {name for name in specified if name.startswith("siglip_")}
    required = {name for name in specified if not name.startswith(("all_x_embedder.", "all_final_layer."))
                and name not in optional and name != "t_embedder"}
    if (not counts.keys() <= specified.keys() or any(counts[name] != 1 for name in required)
            or counts["t_embedder"] != 2
            or any(sum(value for name, value in counts.items() if name.startswith(group)) != 1
                   for group in ("all_x_embedder.", "all_final_layer."))
            or (optional.intersection(counts) and any(counts[name] != 1 for name in optional))):
        raise ValueError("Incomplete native Omni forward trace")
    for name, count in counts.items():
        if [row["occurrence"] for row in rows if row["boundary"] == name] != list(range(1, count + 1)):
            raise ValueError("Nonconsecutive forward occurrence")
