"""Overlap frozen publisher reads while retaining ordered durable target writes."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import time


PIECE_BYTES = 4 * 1024 * 1024
MAX_PARALLEL_READS = 3


def advance(row, transport, publisher, max_pieces, deadline, record, *, fetch):
    """Copy up to eight consecutive pieces of the target's current segment."""
    if type(max_pieces) is not int or not 1 <= max_pieces <= 8:
        raise ValueError("Expected a window of one to eight pieces")
    asset = row["asset_id"]

    def call(operation, data=None, **arguments):
        remaining = int(deadline - time.monotonic())
        if remaining < 6:
            raise TimeoutError("Wire window exhausted")
        argv = [operation, "--asset", str(asset), "--budget", str(min(150, remaining - 2))]
        for name, value in arguments.items():
            argv.extend(["--" + name.replace("_", "-"), str(value)])
        started = time.monotonic()
        answer = transport(argv, data, remaining)
        if answer.get("file_id") != asset:
            raise ValueError("Target replied for a different asset")
        record(operation, answer)
        record("target_wall", {"operation": operation, "seconds": time.monotonic() - started})
        return answer

    status = call("inspect")
    committed = status["confirmed_bytes"]
    if committed == row["bytes"]:
        return status
    length, offset = status["segment_bytes"], status["wire_bytes"]
    if not (0 <= committed < row["bytes"] and
            0 <= offset <= length <= min(8 * PIECE_BYTES, row["bytes"] - committed)
            and (offset == length or offset % PIECE_BYTES == 0)):
        raise ValueError("Target returned invalid piece boundaries")

    def read(segment_offset, count):
        started = time.monotonic()
        left = min(90, deadline - started)
        if left <= 0:
            raise TimeoutError("Publisher read budget exhausted")
        data = fetch(row, committed + segment_offset, count, left, publisher)
        if len(data) != count:
            raise ValueError("Publisher piece has wrong length")
        return data, time.monotonic() - started

    offsets = iter(range(offset, min(length, offset + max_pieces * PIECE_BYTES), PIECE_BYTES))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_READS) as executor:
        pending = deque()

        def schedule():
            start = next(offsets, None)
            if start is not None:
                count = min(PIECE_BYTES, length - start)
                pending.append((start, count, executor.submit(read, start, count)))

        for _ in range(MAX_PARALLEL_READS):
            schedule()
        try:
            while pending:
                start, count, future = pending.popleft()
                data, elapsed = future.result(timeout=max(0.0, deadline - time.monotonic()))
                record("publisher_wall", {"offset": committed + start, "bytes": count, "seconds": elapsed})
                digest = sha256(data).hexdigest()
                status = call("append", data, file_offset=committed, wire_offset=start, digest=digest)
                offset = start + count
                if (status["confirmed_bytes"], status["wire_bytes"], status["piece_SHA256"]) != (
                        committed, offset, digest):
                    raise ValueError("Target did not attest this exact ordered piece")
                schedule()
        finally:
            for _, _, future in pending:
                future.cancel()
    if offset == length:
        status = call("commit", file_offset=committed)
        if status["confirmed_bytes"] != committed + length:
            raise ValueError("Target did not commit the expected full segment")
    return status
