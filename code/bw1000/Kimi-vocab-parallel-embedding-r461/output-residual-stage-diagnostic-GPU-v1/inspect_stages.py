"""Read a bounded prefix without turning an interrupted operation into a pass."""
import argparse
import json
from pathlib import Path
import re


def summarize(raw):
    events = []
    counters = {}
    times = {}
    for line in raw.splitlines(keepends=True):
        if not line.startswith(b"kimi-stage "):
            continue
        if not line.endswith(b"\n"):
            continue
        match = re.fullmatch(rb"kimi-stage (python|native) ([0-9]+) ([0-9]+) ([a-z0-9_.]+)\r?\n", line)
        if not match:
            raise ValueError("malformed complete stage record")
        lane, seq, ns, name = match.groups()
        seq, ns = int(seq), int(ns)
        if seq != counters.get(lane, 0) or ns < times.get(lane, 0):
            raise ValueError("unordered stage sequence or clock")
        counters[lane] = seq + 1
        times[lane] = ns
        events.append(dict(lane=lane.decode(), name=name.decode(), monotonic_ns=ns))
    outstanding = {}
    for event in events:
        name = event["name"]
        lane = event["lane"]
        if name.endswith(".enter"):
            outstanding.setdefault(lane, []).append(name[:-6])
        elif name.endswith(".return"):
            key = name[:-7]
            pending = outstanding.setdefault(lane, [])
            if not pending or pending.pop() != key:
                raise ValueError("stage return without its matching entry")
    return dict(events=events, completed_records=len(events),
                latest=events[-1]["name"] if events else None,
                innermost_unreturned={lane: stack[-1] for lane, stack in outstanding.items() if stack})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    with args.log.open("rb") as file:
        raw = file.read(512*1024 + 1)
    if len(raw) > 512*1024:
        raise ValueError("diagnostic stage log exceeds 512 KiB")
    print(json.dumps(summarize(raw), allow_nan=False))


if __name__ == "__main__":
    main()
