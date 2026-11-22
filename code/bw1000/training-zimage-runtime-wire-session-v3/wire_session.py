"""Bounded framed transport for a single source-pinned Z-Image asset."""

import argparse
from io import BytesIO
import json
import queue
import re
import signal
import sys
import threading
import time


MAX_HEADER = 2048
MAX_RECEIPT = 8192
PIECE_BYTES = 4 << 20
MAX_OPERATIONS = 48
HEX = re.compile(r"[0-9a-f]{64}")


def integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def emit(stdout, value):
    stdout.write((json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii"))
    stdout.flush()


def read_frame(stdin):
    line = stdin.readline(MAX_HEADER+1)
    if not line:
        return None
    if len(line) > MAX_HEADER or not line.endswith(b"\n"):
        raise ValueError("Wire request header exceeds its bound")
    spec = json.loads(line)
    if not isinstance(spec, dict):
        raise ValueError("Wire frame must be a mapping")
    return spec


def operation(stdin, spec, pieces, asset):
    op = spec.get("op")
    expected = {"op", "seq"}
    if op == "inspect" and spec.keys() == expected:
        return pieces.state(asset)
    if op == "commit" and spec.keys() == expected|{"file_offset"}:
        if not integer(spec["file_offset"], 0, asset["bytes"] - 1):
            raise ValueError("Invalid commit offset")
        return pieces.commit(asset, spec["file_offset"])
    if op == "append" and spec.keys() == expected|{"file_offset", "wire_offset", "digest", "count"}:
        if not (
            integer(spec["file_offset"], 0, asset["bytes"] - 1)
            and integer(spec["wire_offset"], 0, (32 << 20) - 1)
            and integer(spec["count"], 1, PIECE_BYTES)
            and isinstance(spec["digest"], str) and HEX.fullmatch(spec["digest"])
        ):
            raise ValueError("Invalid append extent or digest")
        raw = stdin.read(spec["count"])
        if len(raw) != spec["count"]:
            raise ValueError("Wire payload interrupted before staging")
        return pieces.append(asset, spec["file_offset"], spec["wire_offset"], spec["digest"], BytesIO(raw))
    raise ValueError("Unexpected wire operation or request keys")


def serve(stdin, stdout, pieces, asset, nonce, source_sha):
    emit(stdout, {"state": "SESSION_READY", "asset": asset["asset_id"], "nonce": nonce, "source_SHA256": source_sha})
    for ordinal in range(1, MAX_OPERATIONS+1):
        spec = read_frame(stdin)
        if spec is None:
            return 0
        if type(spec.get("seq")) is not int or spec["seq"] != ordinal:
            raise ValueError("Wire sequence was repeated or skipped")
        try:
            receipt = operation(stdin, spec, pieces, asset)
        except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            emit(stdout, {"seq": ordinal, "state": "SESSION_STOPPED", "error_type": type(exc).__name__})
            return 1
        emit(stdout, {"seq": ordinal, "state": "ACK", "receipt": receipt})
    return 0


class Client:
    """One opened SSH channel. Timeout or disconnect stops sending further frames."""

    def __init__(self, child, asset, nonce, source_sha, ready_timeout):
        self.child, self.asset, self.sequence = child, asset, 0
        self.inbox = queue.Queue()
        self.last = {"operation": "handshake", "frame_attempted": False}
        self.reader = threading.Thread(target=self.read_messages, daemon=True)
        self.reader.start()
        expected = dict(state="SESSION_READY", asset=asset, nonce=nonce, source_SHA256=source_sha)
        if self.read(ready_timeout) != expected:
            raise ValueError("Authenticated session marker differs")

    def read_messages(self):
        try:
            while True:
                line = self.child.stdout.readline(MAX_RECEIPT+1)
                if not line:
                    self.inbox.put(EOFError("Authenticated wire session closed"))
                    return
                if len(line) > MAX_RECEIPT or not line.endswith(b"\n"):
                    raise ValueError("Wire receipt exceeds its bound")
                self.inbox.put(json.loads(line))
        except (ValueError, OSError) as exc:
            self.inbox.put(exc)

    def read(self, timeout):
        try:
            value = self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Wire session receipt deadline") from None
        if isinstance(value, Exception):
            raise value
        return value

    def transport(self, args, payload, timeout):
        op, *opts = args
        options = dict(zip(opts[::2], opts[1::2], strict=True))
        if len(options)*2 != len(opts):
            raise ValueError("Repeated wire invocation option")
        if int(options.pop("--asset")) != self.asset or op not in {"inspect", "append", "commit"}:
            raise ValueError("Wire invocation is outside the authenticated asset scope")
        remaining = min(int(options.pop("--budget")), timeout)
        if remaining <= 0 or self.sequence == MAX_OPERATIONS:
            raise TimeoutError("Wire session has no remaining operations")
        if op in {"append", "commit"}:
            options["file_offset"] = int(options.pop("--file-offset"))
        if op == "append":
            options["wire_offset"] = int(options.pop("--wire-offset"))
            options["digest"] = options.pop("--digest")
            options["count"] = len(payload)
        self.sequence += 1
        header = (json.dumps({**options, "op": op, "seq": self.sequence}, separators=(",", ":"))+"\n").encode("ascii")
        sent = queue.Queue()
        self.last = {"operation": op, "frame_attempted": False, "payload_bytes": len(payload) if payload else 0}

        def write():
            try:
                self.last["frame_attempted"] = True
                self.child.stdin.write(header)
                if payload:
                    self.child.stdin.write(payload)
                self.child.stdin.flush()
                sent.put(None)
            except OSError as exc:
                sent.put(exc)

        deadline = time.monotonic()+remaining
        threading.Thread(target=write, daemon=True).start()
        try:
            outgoing_error = sent.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("Wire session write deadline") from None
        if outgoing_error:
            raise outgoing_error
        response = self.read(max(0, deadline-time.monotonic()))
        if response.get("seq") != self.sequence or response.get("state") != "ACK":
            self.last["remote_response"] = response
            raise RuntimeError("Wire receiver stopped without acknowledging this frame")
        self.last["frame_confirmed"] = True
        return response["receipt"]


def main():
    from pathlib import Path
    from hashlib import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--wire", type=Path, required=True)
    parser.add_argument("--wire-sha256", required=True)
    parser.add_argument("--asset", type=int, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if not (0 < args.seconds <= 420 and re.fullmatch(r"[a-f0-9]{32}", args.nonce)):
        raise ValueError("Wire session duration or nonce differs")
    wire_file = args.wire/"wire.py"
    if wire_file.resolve(strict=True) != wire_file or sha256(wire_file.read_bytes()).hexdigest() != args.wire_sha256:
        raise ValueError("Original wire source differs")
    sys.path.insert(0, str(args.wire))
    import wire

    receiver, _, stage = wire.load_legacy(args.legacy)
    rows = [row for row in receiver.assets() if row["asset_id"]==args.asset]
    if len(rows) != 1:
        raise ValueError("Unknown asset identity")
    pieces = wire.Pieces(receiver.TARGET, receiver, stage)

    def expired(*_):
        raise TimeoutError("Original wire session deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    source_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    raise SystemExit(serve(sys.stdin.buffer, sys.stdout.buffer, pieces, rows[0], args.nonce, source_sha))


if __name__ == "__main__":
    main()
