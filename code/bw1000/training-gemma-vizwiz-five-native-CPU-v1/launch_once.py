"""Run the published five-row Gemma VizWiz input audit through the existing session."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from types import ModuleType


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
PRIVATE = REPO / "local-only"
TRANSPORT = PRIVATE / "run-minicpm-sdpa-target-asset-preflight-original.py"
TRANSPORT_SHA256 = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"


def verified_bytes(path, pin):
    with path.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    if len(raw) != pin["bytes"] or sha256(raw).hexdigest() != pin["SHA256"]:
        raise ValueError(f"bytes changed: {path.name}")
    return raw


def packet():
    claim_raw = (PACKAGE / "CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    offered = {}
    for remote_name, pin in claim["payload_files"].items():
        raw = verified_bytes(REPO / pin["source"], pin)
        offered[remote_name] = b64encode(raw).decode("ascii")
    claim_pin = sha256(claim_raw).hexdigest()
    receiver = verified_bytes(PACKAGE / "receive_once.py", claim["receiver"])
    return claim, claim_pin, b"".join((
        b"CLAIM = " + repr(claim).encode() + b"\n",
        b"CLAIM_SHA256 = " + repr(claim_pin).encode() + b"\n",
        b"FILES = " + repr(offered).encode() + b"\n",
        receiver,
    ))


def published_claim(revision, claim):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("exact published Git revision required")
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=8)
    paths = {PACKAGE / name for name in ("CLAIM.json", "launch_once.py", "receive_once.py")}
    paths.update(REPO / pin["source"] for pin in claim["payload_files"].values() if not pin["public_data"])
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        blob = subprocess.check_output(["git", "cat-file", "blob", revision + ":" + relative], cwd=REPO,
                                       stderr=subprocess.PIPE, timeout=8)
        if blob != path.read_bytes():
            raise ValueError("published source differs from the current source")


def save(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-local", action="store_true")
    parser.add_argument("--publication")
    args = parser.parse_args()
    claim, claim_pin, body = packet()
    record = dict(request=claim["request"], claim_SHA256=claim_pin, files=len(claim["payload_files"]),
                  payload_SHA256=sha256(body).hexdigest(), payload_bytes=len(body), replay_allowed=False)
    if args.check_local:
        compile(body, "<exact-scope>", "exec")
        print(json.dumps(dict(record, dispatched=False), sort_keys=True))
        return 0
    published_claim(args.publication or "", claim)
    raw = TRANSPORT.read_bytes()
    if sha256(raw).hexdigest() != TRANSPORT_SHA256:
        raise ValueError("existing transport source differs")
    transport = ModuleType("existing_transport")
    exec(compile(raw, str(TRANSPORT), "exec"), vars(transport))
    argv = [*transport.ssh(REPO.parent), shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-"])]
    sleeve = PRIVATE / ("gemma-vizwiz-five-native-cpu-" + claim["request"])
    sleeve.mkdir()
    save(sleeve / "intent.json", dict(record, publication=args.publication))
    try:
        result = subprocess.run(argv, input=body, capture_output=True, timeout=330)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as failure:
        rc, stdout, stderr = None, failure.stdout or b"", failure.stderr or b""
    receipt = []
    if len(stdout) <= 32768:
        for line in stdout.splitlines():
            if line.startswith(b"{"):
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("request") == claim["request"]:
                    receipt.append(message)
    terminal_report = next((message for message in receipt if "terminal" in message), None)
    state = "ORIGINAL_RETURNED" if terminal_report else "UNKNOWN"
    pin = save(sleeve / "result.json", dict(record, publication=args.publication, transport_rc=rc,
                                            remote_outcome=state, receipt=receipt, stdout_SHA256=sha256(stdout).hexdigest(),
                                            stderr_SHA256=sha256(stderr).hexdigest()))
    print(json.dumps(dict(request=claim["request"], transport_rc=rc, remote_outcome=state,
                          saved_result_SHA256=pin, terminal=terminal_report and terminal_report["terminal"],
                          passed=bool(terminal_report and terminal_report["passed"])), sort_keys=True))
    return int(rc != 0 or not terminal_report or not terminal_report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
