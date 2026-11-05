"""Persist one C4 host packet only after the original owner and inner reads."""
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from consume import read_after_original


def consume_window(read, original_window, destination):
    inner = original_window["scalar_packets"]["inner"]
    data = Path(inner["path"]).read_bytes()
    if len(data) != inner["bytes"] or sha256(data).hexdigest() != inner["SHA256"]:
        raise ValueError("original inner packet changed before C4 consumption")
    start = perf_counter()
    result, raw = read_after_original(read, data, original_window["geometry"]["past"][0])
    with destination.open("xb") as stream:
        stream.write(raw)
    result["consume_and_IO_wall_ms"] = (perf_counter() - start) * 1000
    result["packet"] = dict(path=str(destination), bytes=len(raw), SHA256=sha256(raw).hexdigest())
    return result
