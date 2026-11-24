"""One authenticated host invocation for T07 zero-GRES evidence and the guarded bridge."""
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import time


PYTHON = "/usr/local/python3.12/bin/python3"
FRAME_LIMIT = 8192
PAYLOAD_LIMIT = 262144


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def emit(row):
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def receive(limit):
    line = sys.stdin.buffer.readline(limit + 2)
    need(1 < len(line) <= limit + 1 and line.endswith(b"\n"), "missing or unbounded client frame")
    return line[:-1]


def execute(observe_bytes, observer_pin, session_pin, bridge_pin):
    observer = b64decode(observe_bytes, validate=True)
    need(hash_bytes(observer) == observer_pin and all(re.fullmatch(r"[a-f0-9]{64}", p)
         for p in (observer_pin, session_pin, bridge_pin)), "frozen one-session program differs")
    module = {"__name__": "published_device_observer"}
    exec(compile(observer, "published-t07-observe.py", "exec"), module)
    obs, gpu = (module[k] for k in ("REQUEST", "GPU_REQUEST"))
    scope = dict(device_request=obs, GPU_request=gpu)
    emit(dict(stage="HOST_AUTHORITY", **scope, observer_SHA256=observer_pin, session_SHA256=session_pin))
    got = json.loads(receive(FRAME_LIMIT))
    need(got == dict(stage="START_ZERO_GRES", **scope), "client did not request this exact observation")
    original = sys.argv
    try:
        sys.argv = ["published-t07-observe.py", "observe", observer_pin]
        result = module["run"]()
    finally:
        sys.argv = original
    candidate_file = module["NAMESPACE"] / "candidate.json"
    raw = candidate_file.read_bytes()
    need(0 < len(raw) < FRAME_LIMIT and json.loads(raw) == result, "durable raw candidate differs")
    certificate = result["certified_observation"]
    receipt_pin = hash_bytes(raw)
    emit(dict(stage="DEVICE_CANDIDATE", **scope, candidate_SHA256=receipt_pin, candidate=result))
    offer = json.loads(receive(FRAME_LIMIT))
    need(offer["stage"] == "OFFER_GPU_PAYLOAD" and offer["device_request"] == obs and offer["GPU_request"] == gpu
         and offer["candidate_SHA256"] == receipt_pin and re.fullmatch(r"[a-f0-9]{64}", offer["payload_SHA256"])
         and type(offer["payload_bytes"]) is int and 0 < offer["payload_bytes"] <= PAYLOAD_LIMIT
         and 0 <= time.time() - certificate["observed_epoch"] <= 40, "no fresh matching GPU offer")
    amounts = dict(candidate_SHA256=receipt_pin, payload_SHA256=offer["payload_SHA256"], payload_bytes=offer["payload_bytes"])
    emit(dict(stage="GPU_AUTHORITY", **scope, **amounts))
    packet = receive(PAYLOAD_LIMIT)
    need(len(packet) == amounts["payload_bytes"] and hash_bytes(packet) == amounts["payload_SHA256"],
         "GPU payload differs from the offered digest and length")
    envelope = json.loads(packet)
    bridge = b64decode(envelope["bridge"], validate=True)
    need(envelope["request"] == gpu and envelope["t07_successor_device"] == certificate
         and envelope["bridge_SHA256"] == bridge_pin == hash_bytes(bridge)
         and 0 <= time.time() - certificate["observed_epoch"] <= 40, "GPU payload or original candidate expired")
    emit(dict(stage="HOST_RECEIVED_GPU_PAYLOAD", **scope, **amounts))
    launched = subprocess.run([PYTHON, "-I", "-S", "-B", "-c", bridge.decode(), "start", "--launch-only"],
                              input=packet, capture_output=True, timeout=37)
    need(len(launched.stdout) <= FRAME_LIMIT and len(launched.stderr) <= FRAME_LIMIT, "bounded bridge reply")
    reply = json.loads(launched.stdout) if launched.returncode == 0 else None
    emit(dict(stage="BRIDGE_REPLY", **scope, rc=launched.returncode, reply=reply,
              stdout_SHA256=hash_bytes(launched.stdout), stderr_SHA256=hash_bytes(launched.stderr)))
    return launched.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(execute(*sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        emit(dict(stage="HOST_ERROR", error_type=type(error).__name__,
                  detail_SHA256=hash_bytes(str(error).encode("utf8", "replace"))))
        raise SystemExit(1)
