"""Verify cumulative input audit counters from the Deep483 guarded probe."""

GUARD_BYTES_EACH_SIDE = 256
FROZEN_WEIGHT_BYTES = 13_144_320
COUNTERS = ("input_audits", "input_buffers_checked", "input_bytes_checked")


class InputAudit:
    """Check observed guard coverage without copying any device buffers."""

    def __init__(self, metadata):
        sizes = [entry["bytes"] for entry in metadata["weights"].values()]
        if len(sizes) != 7 or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("seven checkpoint payload sizes required")
        if sum(sizes) != FROZEN_WEIGHT_BYTES:
            raise ValueError("checkpoint payload bytes differ")
        self.totals = (7, 7, sum(sizes) + 7 * 2 * GUARD_BYTES_EACH_SIDE)
        self.seconds = None

    def check(self, observed, *, prepared_payloads=None):
        """The first call is postload. Only a successful prepare adds buffers."""
        expected = self.totals
        if prepared_payloads is not None:
            if len(prepared_payloads) != 3:
                raise ValueError("three prepare inputs required")
            lengths = tuple(map(len, prepared_payloads))
            rows, remainder = divmod(lengths[2], 8)
            if remainder or not 1 <= rows <= 512 or lengths[:2] != (rows * 8192, rows * 2048):
                raise ValueError("prepare audit input geometry differs")
            audits, buffers, nbytes = self.totals
            expected = (audits + 1, buffers + 3,
                        nbytes + sum(lengths) + 3 * 2 * GUARD_BYTES_EACH_SIDE)
        guard = observed.get("input_guard_bytes_each_side")
        if type(guard) is not int or guard != GUARD_BYTES_EACH_SIDE:
            raise ValueError("input guard width differs")
        for field, value in zip(COUNTERS, expected):
            actual = observed.get(field)
            if type(actual) is not int or actual != value:
                raise ValueError(f"{field} differs from guarded input schedule")
        seconds = observed.get("input_audit_seconds")
        if type(seconds) not in (int, float) or not 0 <= seconds < float('inf'):
            raise ValueError("invalid cumulative input audit time")
        if self.seconds is not None and seconds < self.seconds:
            raise ValueError("cumulative input audit time decreased")
        if prepared_payloads is None and self.seconds is not None and seconds != self.seconds:
            raise ValueError("input audit time changed without prepare")
        self.totals = expected
        self.seconds = seconds
        return dict(zip(COUNTERS, expected), input_audit_seconds=seconds,
                    input_guard_bytes_each_side=guard)


def check_before_load(observed):
    """Require an empty guarded owner before reading any checkpoint ranges."""
    if observed.get("ready") is not False or observed.get("pending") is not False:
        raise ValueError("guarded owner is not fresh")
    geometry = {
        "past": 0, "prepares": 0, "commits": 0, "aborts": 0, "resets": 0,
        "local_heads": 64, "compressed_capacity": 8192, "full_context": 32768,
        "input_guard_bytes_each_side": GUARD_BYTES_EACH_SIDE,
    }
    for field, expected in geometry.items():
        actual = observed.get(field)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"invalid guarded owner pre-load field: {field}")
    for field in COUNTERS:
        actual = observed.get(field)
        if type(actual) is not int or actual != 0:
            raise ValueError(f"invalid guarded owner pre-load field: {field}")
    seconds = observed.get("input_audit_seconds")
    if type(seconds) not in (int, float) or seconds != 0:
        raise ValueError("input audit clock must start at zero")
    free = observed.get("driver_free")
    if type(free) is not int or free < 16 * 1024**3:
        raise ValueError("strict16GiB floor before checkpoint load")
