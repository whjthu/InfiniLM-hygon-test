"""Fetch one immutable 4 MiB source piece for the durable SCNet split receiver."""
from hashlib import sha256
import time
from urllib.request import Request, build_opener

from wire import PIECE


def fetch(row, offset, count, remaining, publisher):
    end = offset + count - 1
    if not (0 <= offset <= end < row["bytes"] and 0 < count <= PIECE and remaining > 0):
        raise ValueError("Publisher wire range is out of bounds")
    url = row["publisher_URL"]
    if not publisher.official_endpoint(url):
        raise ValueError("Only the frozen official HTTPS endpoint is permitted")
    deadline = time.monotonic() + remaining
    request = Request(url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
    with build_opener(publisher.OfficialRedirect()).open(request, timeout=min(20, remaining)) as response:
        if not (publisher.official_endpoint(response.geturl()) and response.status == 206
                and response.headers.get("Content-Range") == f"bytes {offset}-{end}/{row['bytes']}"
                and response.headers.get("Content-Length") == str(count)
                and response.headers.get("Content-Encoding", "identity").lower() == "identity"):
            raise ValueError("Publisher did not honor the exact wire range")
        body = bytearray()
        while len(body) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError("Wire fetch budget exhausted")
            part = response.read1(min(1 << 20, count - len(body)))
            if not part:
                raise ValueError("Short publisher response")
            body.extend(part)
        if response.read(1) != b"" or time.monotonic() >= deadline:
            raise ValueError("Publisher wire response exceeds its bound")
        return body


def advance(row, transport, publisher, max_pieces, deadline, record):
    if type(max_pieces) is not int or not 1 <= max_pieces <= 8:
        raise ValueError("One wire window is limited to eight pieces")
    asset = row["asset_id"]

    def call(operation, data=None, **arguments):
        left = int(deadline - time.monotonic())
        if left < 6:
            raise TimeoutError("Wire window exhausted")
        argv = [operation, "--asset", str(asset), "--budget", str(min(150, left - 2))]
        for name, value in arguments.items():
            argv.extend(["--" + name.replace("_", "-"), str(value)])
        reply = transport(argv, data, left)
        if reply.get("file_id") != asset:
            raise ValueError("Wire receiver replied for a different asset")
        record(operation, reply)
        return reply

    status = call("inspect")
    original = status["confirmed_bytes"]
    if original == row["bytes"]:
        return status
    length = status["segment_bytes"]
    offset = status["wire_bytes"]
    for _ in range(max_pieces):
        if offset == length:
            break
        count = min(PIECE, length - offset)
        data = fetch(row, original + offset, count, min(90, deadline - time.monotonic()), publisher)
        digest = sha256(data).hexdigest()
        status = call("append", data, file_offset=original, wire_offset=offset, digest=digest)
        if status["confirmed_bytes"] != original or status["wire_bytes"] != offset + count or status["piece_SHA256"] != digest:
            raise ValueError("Target persisted a different wire piece")
        offset += count
    if offset == length:
        status = call("commit", file_offset=original)
        if status["confirmed_bytes"] != original + length:
            raise ValueError("Original segment did not commit at its expected extent")
    return status
