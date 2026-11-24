"""Stream-hash the official CLIP-L text ranges already present by offset in the retained prefix."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener

from probe_hymotion_official_clip_range import BASE, CHECKPOINT_SIZE, OfficialRedirects, REVISION


INTERVALS = ((71856, 72472), (74532, 472383488))
HEADER_SHA = "c36dcc32f7c435dd670a1ba5aee76c176d66c1fa87cb50a3e0fec6bc107aa08f"
CHUNK = 512 * 1024


def fingerprint(session, start, stop, deadline):
    if (start, stop) not in INTERVALS:
        raise ValueError("outside the two officially tracked retained text ranges")
    size = stop-start
    req = Request(BASE + "model.safetensors", headers={
        "Range": f"bytes={start}-{stop-1}", "Accept-Encoding": "identity"})
    sig = sha256()
    with session.open(req, timeout=9) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{CHECKPOINT_SIZE}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or response.headers.get("Content-Length") not in (None, str(size))):
            raise ValueError("official content did not bind the exact retained interval")
        left = size
        while left:
            if time.monotonic() >= deadline:
                raise TimeoutError("original retained-range verification budget expired")
            data = response.read(min(left, CHUNK))
            if not data:
                raise EOFError("official retained interval truncated")
            sig.update(data)
            left -= len(data)
        if response.read(1):
            raise ValueError("official retained interval contains trailing bytes")
    return dict(interval=[start, stop], bytes=size, SHA256=sig.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("header_only_copy", type=Path)
    args = parser.parse_args()
    with args.header_only_copy.open("rb") as stream:
        header = stream.read(71857)
    if len(header) != 71856 or sha256(header).hexdigest() != HEADER_SHA:
        raise ValueError("official tensor offsets are not bound to the observed header")
    t = time.monotonic()
    opener = build_opener(OfficialRedirects())
    record = dict(revision=REVISION, official_header_SHA256=HEADER_SHA,
                  completed=[], scope="publisher hashes only; target prefix yet to be read",
                  local_weight_files_written=0, cluster_connections=0, encoder_rows=0, target_partial_verified=False)
    try:
        for start, stop in INTERVALS:
            record["completed"].append(fingerprint(opener, start, stop, t+145))
    except (EOFError, OSError, ValueError) as error:
        record["failure_type"] = type(error).__name__
    record.update(wall_seconds=time.monotonic()-t,
                  state="OFFICIAL_TEXT_PREFIX_EXPECTATIONS_PINNED" if len(record["completed"]) == 2 else "PARTIAL")
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, sort_keys=True))
    return int(record["state"] != "OFFICIAL_TEXT_PREFIX_EXPECTATIONS_PINNED")


if __name__ == "__main__":
    raise SystemExit(main())
