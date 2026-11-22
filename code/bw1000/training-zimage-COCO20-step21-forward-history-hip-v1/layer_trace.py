"""Record ordered native forward boundaries without retaining tensor bodies."""
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
    if not tracked <= named.keys():
        raise ValueError("Missing named forward boundaries")
    return {name: named[name] for name in sorted(tracked)}


@contextmanager
def observe(dit, tensor_type, tensor_digest, deadline):
    modules = layout(dit)
    rows = []
    active = set()
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
            if name in active:
                raise ValueError(f"Repeated forward boundary: {name}")
            active.add(name)
            rows.append({"boundary": name, "args": signature(args), "kwargs": signature(kwargs),
                         "output": signature(output)})
        return record

    try:
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(hook(name), with_kwargs=True))
        yield rows
    finally:
        for handle in reversed(hooks):
            handle.remove()


def validate(rows, dit):
    names = [row["boundary"] for row in rows]
    required = {name for name in layout(dit) if not name.startswith(("all_x_embedder.", "all_final_layer."))}
    if (len(names) != len(set(names)) or not required <= set(names)
            or any(sum(name.startswith(group) for name in names) != 1
                   for group in ("all_x_embedder.", "all_final_layer."))):
        raise ValueError("Incomplete native forward trace")
