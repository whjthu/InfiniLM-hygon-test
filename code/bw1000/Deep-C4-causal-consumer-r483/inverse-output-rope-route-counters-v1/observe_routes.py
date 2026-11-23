"""Read cumulative native route counts at synchronized request boundaries."""

import ctypes


FIELDS = ("legacy", "fallback_geometry", "fallback_no_stream", "pack32", "pack64", "failed")
ARM_ROUTE = {"legacy": "legacy", "h1-pack32": "pack32", "h1-pack64": "pack64"}


class RouteCounts:
    def __init__(self, library):
        self.read = library.deep_inverse_output_rope_route_counts_v1
        self.read.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
        self.read.restype = ctypes.c_int

    def snapshot(self):
        values = (ctypes.c_uint64 * len(FIELDS))()
        status = self.read(values, len(values))
        if status != 0:
            raise ValueError(f"Native route counting is unavailable (status {status})")
        return dict(zip(FIELDS, values, strict=True))


def verified_delta(before, after, arm, *, require_selected):
    if arm not in ARM_ROUTE or set(before) != set(FIELDS) or set(after) != set(FIELDS):
        raise ValueError("Native route arm or snapshot schema differs")
    output = {}
    for name in FIELDS:
        start, end = before[name], after[name]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < 2**64:
            raise ValueError("A native route counter is malformed or moved backwards")
        output[name] = end - start
    selected = ARM_ROUTE[arm]
    forbidden = set(ARM_ROUTE.values()) - {selected}
    if arm == "legacy":
        forbidden.update(("fallback_geometry", "fallback_no_stream"))
    if output["failed"] or any(output[key] for key in forbidden):
        raise ValueError("Native inverse RoPE took an unexpected or failed route")
    if require_selected and output[selected] == 0:
        raise ValueError("The selected native inverse RoPE route was never observed")
    return output
