"""Exercise staging order with the pinned metadata and synthetic byte buffers."""
import ctypes
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import checkpoint


meta = json.loads(Path(__file__).with_name("CHECKPOINT_METADATA.json").read_bytes())
names = sorted(meta["weights"])
parameters = [
    dict(name=name, shape=entry["shape"], dtype=meta["dtype_codes"][entry["dtype"]],
         bytes=entry["bytes"], contiguous=True, loaded=False)
    for name, entry in meta["weights"].items()
]
payloads = {name: bytes([i + 1]) * meta["weights"][name]["bytes"] for i, name in enumerate(names)}
digests = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
last = names[-1]


def check_case(case, expected_exception=None):
    events = []
    pins = {k: meta[k] for k in ("layer", "original_config_SHA256", "original_index_SHA256")}
    pins["weights"] = {name: dict(bytes=len(payloads[name]), SHA256=digests[name]) for name in names}
    if case == "bad_last_sha":
        pins["weights"][last]["SHA256"] = "0" * 64

    class Recorder:
        def read(self, name):
            assert name == "parameters"
            return parameters

        def call(self, operation, *args):
            events.append(operation)
            if operation == "load":
                name, buffer, length = args
                assert hashlib.sha256(ctypes.string_at(buffer, length)).hexdigest() == digests[name.decode()]
                if case == "native_failure":
                    raise RuntimeError("recorded load failure")

    def read_weight(root, entry):
        name = entry["checkpoint_key"].removeprefix("layers.2.attn.indexer.")
        events.append("read:" + name)
        payload = payloads[name]
        return payload[:-1] if case == "short_last" and name == last else payload

    def deadline(stage):
        if case == "late_read_deadline" and stage == "checkpoint.read." + last:
            raise TimeoutError("recorded deadline")

    with patch.object(checkpoint, "headers", return_value=None), patch.object(checkpoint, "read_weight", read_weight):
        try:
            result = checkpoint.load_checkpoint(Recorder(), meta, pins, None, deadline)
        except (ValueError, TimeoutError, RuntimeError) as error:
            assert type(error) is expected_exception, str(error)
        else:
            assert expected_exception is None
            assert result["loaded"] == names
            assert result["bytes"] == 13144320

    reads = ["read:" + name for name in names]
    expected = {
        "success": reads + ["load"] * 7 + ["postload"],
        "bad_last_sha": reads,
        "short_last": reads,
        "late_read_deadline": reads[:-1],
        "native_failure": reads + ["load"],
    }
    assert events == expected[case], (case, events)
    return {key: events.count(key) for key in ("load", "postload")}


print(json.dumps({
    "success": check_case("success"),
    "bad_last_sha": check_case("bad_last_sha", ValueError),
    "short_last": check_case("short_last", ValueError),
    "late_read_deadline": check_case("late_read_deadline", TimeoutError),
    "native_failure": check_case("native_failure", RuntimeError),
}))
