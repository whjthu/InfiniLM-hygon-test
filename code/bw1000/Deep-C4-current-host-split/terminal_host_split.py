"""Check original C4 host packets against the original inner windows at terminal."""
from pathlib import Path

from consume import consume
from support import actual, sha


def audit_window(output, window):
    records = []
    for key in ("inner", "host"):
        record = (window["scalar_packets"]["inner"] if key == "inner"
                  else window["C4_host_split"]["packet"])
        path = actual(record["path"])
        assert path.resolve().is_relative_to(Path(output).resolve())
        assert path.stat().st_size == record["bytes"] and sha(path) == record["SHA256"]
        records.append(path.read_bytes())
    geometry = window["geometry"]
    sample = geometry["decode_call_1based"]
    assert sample in (11, 12) and geometry["past"] == [8489 + sample - 1]
    joined = consume(records[1], records[0], geometry["past"][0])
    original = window["C4_host_split"]
    assert all(original[key] == value for key, value in joined.items())
    return dict(sample=sample, packet=original["packet"], **joined,
                read_wall_ms=original["read_wall_ms"],
                consume_and_IO_wall_ms=original["consume_and_IO_wall_ms"])
