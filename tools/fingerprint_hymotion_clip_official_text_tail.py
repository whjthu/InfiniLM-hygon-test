"""Independently fingerprint the exact official CLIP-L text suffix plus bound header."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener

import inspect_hymotion_clip_text_weight_ranges as layout
from probe_hymotion_official_clip_range import BASE, CHECKPOINT_SIZE, OfficialRedirects, REVISION


HEADER_SHA256 = "c36dcc32f7c435dd670a1ba5aee76c176d66c1fa87cb50a3e0fec6bc107aa08f"
START = 472383488
END = 492316452
CHUNK = 256 * 1024
WALL_LIMIT = 80


def digest_range(opener, start, end, deadline):
    if not START <= start < end <= END or deadline <= time.monotonic():
        raise ValueError("unexpected official text-tail scope")
    req = Request(BASE + "model.safetensors", headers={
        "Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity"})
    size = end - start
    digest = sha256()
    with opener.open(req, timeout=9) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end-1}/{CHECKPOINT_SIZE}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or response.headers.get("Content-Length") not in (None, str(size))):
            raise ValueError("publisher did not return the exact text-tail interval")
        remaining = size
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError("original local text-tail read expired")
            block = response.read(min(remaining, CHUNK))
            if not block:
                raise EOFError("official suffix truncated")
            digest.update(block)
            remaining -= len(block)
        if response.read(1):
            raise ValueError("official suffix contains excess bytes")
    return {"bytes": size, "SHA256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("header_only_copy", type=Path)
    args = parser.parse_args()
    opener = build_opener(OfficialRedirects())
    t = time.monotonic()
    header = layout.request_header(opener)
    if sha256(header).hexdigest() != HEADER_SHA256:
        raise ValueError("official header differs from the established immutable observation")
    with args.header_only_copy.open("xb") as handle:
        handle.write(header)
    evidence = dict(revision=REVISION, official_header_SHA256=HEADER_SHA256, interval=[START, END],
                    local_weight_files_written=0, cluster_connections=0, scope="source digest only",
                    target_partial_SHA256_verified=False, install_ready=False, encoder_rows=0, gpu=0)
    try:
        evidence.update(state="OFFICIAL_TEXT_TAIL_DIGESTED",
                        official_tail=digest_range(opener, START, END, t + WALL_LIMIT))
    except (OSError, ValueError, EOFError) as error:
        evidence.update(state="OFFICIAL_TEXT_TAIL_UNVERIFIED", error_class=type(error).__name__)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(evidence, sort_keys=True))
    return int(evidence["state"] != "OFFICIAL_TEXT_TAIL_DIGESTED")


if __name__ == "__main__":
    raise SystemExit(main())
