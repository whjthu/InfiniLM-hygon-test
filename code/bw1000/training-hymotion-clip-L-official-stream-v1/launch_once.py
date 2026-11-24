"""Stream one pinned snapshot through the existing authenticated SSH route."""

import argparse
import base64
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from urllib.request import Request, build_opener

from receive_once import BUDGET_SECONDS, CHUNK_BYTES, MEMBERS, completion_marker, save_json, validate_manifest


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
SSH_HELPER = REPO / "local-only/run-minicpm-sdpa-target-asset-preflight-original.py"
SSH_HELPER_SHA256 = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
HTTP_HELPER = REPO / "tools/probe_hymotion_official_clip_range.py"
HTTP_HELPER_SHA256 = "12b7bfad90fee397e6f2541993e73fa1634e473a1a3abc6ab8ab0bd3871eddaf"
METADATA = REPO / "results/bw1000/2026-11-24/training-hymotion-clip-L-official-text-metadata-pins.json"


def load_pinned_module(name, path, pin):
    source = path.read_bytes()
    if sha256(source).hexdigest() != pin:
        raise ValueError("helper digest differs")
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    exec(compile(source, str(path), "exec"), vars(module))
    return module


def published_package(commit):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("full publication commit required")
    package = PACKAGE.relative_to(REPO).as_posix()
    trusted = ("claim.json", "launch_once.py", "receive_once.py")
    for name in trusted:
        published = subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "show", f"{commit}:{package}/{name}"],
            cwd=REPO, capture_output=True, check=True, timeout=8,
        ).stdout
        if published != (PACKAGE / name).read_bytes():
            raise ValueError("installed source differs from recorded publication")


def event_line(stream, request):
    data = stream.readline(4097)
    if not data:
        raise EOFError("receiver returned no event")
    if len(data) > 4096 or not data.endswith(b"\n"):
        raise ValueError("receiver event exceeds its bound")
    event = json.loads(data)
    if not isinstance(event, dict) or event.get("request") != request:
        raise ValueError("receiver returned an unexpected request")
    return event


def verify_metadata(claim):
    raw = METADATA.read_bytes()
    if sha256(raw).hexdigest() != claim["metadata_evidence_SHA256"]:
        raise ValueError("metadata evidence digest differs")
    evidence = json.loads(raw)["observed_official_assets"]
    expected = {name: claim["files"][name] for name in MEMBERS[:-1]}
    if evidence["revision"] != claim["revision"] or evidence["asset_identity"] != expected:
        raise ValueError("official text asset identities differ")


def copy_member(opener, base_url, name, pin, target, deadline):
    remaining = pin["bytes"]
    digest = sha256()
    req = Request(base_url + name, headers={"Accept-Encoding": "identity"})
    timeout = min(15, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("streaming budget expired")
    with opener.open(req, timeout=timeout) as source:
        length = source.headers.get("Content-Length")
        if (source.status != 200 or source.headers.get("Content-Encoding", "identity") != "identity"
                or source.headers.get("Content-Range") is not None
                or (length is not None and length != str(remaining))):
            raise ValueError("unexpected official response")
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError("streaming budget expired")
            block = source.read(min(remaining, CHUNK_BYTES))
            if not block:
                raise EOFError("incomplete official member")
            if target.write(block) != len(block):
                raise OSError("incomplete SSH pipe write")
            remaining -= len(block)
            digest.update(block)
        if source.read(1) or digest.hexdigest() != pin["SHA256"]:
            raise ValueError("official member length or digest differs")


def receive_command(claim_bytes):
    embedded = base64.b64encode(claim_bytes).decode("ascii")
    claim_pin = sha256(claim_bytes).hexdigest()
    prefix = ("from base64 import b64decode\nimport json\n"
              f"CLAIM = json.loads(b64decode({embedded!r}))\nCLAIM_SHA256 = {claim_pin!r}\n")
    code = prefix + (PACKAGE / "receive_once.py").read_text(encoding="utf-8")
    return "/usr/local/python3.12/bin/python3 -I -S -B -c " + shlex.quote(code)


def transfer(command, claim_bytes, claim, opener, base_url, report):
    deadline = time.monotonic() + BUDGET_SECONDS
    expected = sum(pin["bytes"] for pin in claim["files"].values())
    request = claim["request"]
    process = subprocess.Popen(
        [*command, receive_command(claim_bytes)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    watchdog = threading.Timer(max(0, deadline - time.monotonic()), process.kill)
    watchdog.daemon = True
    watchdog.start()
    try:
        ready = event_line(process.stdout, request)
        if ready != {"event": "READY", "request": request, "require_bytes": expected}:
            raise ValueError("receiver did not reserve expected scope")
        report["remote_ready_observed"] = True
        send_error = None
        try:
            for name in MEMBERS:
                copy_member(opener, base_url, name, claim["files"][name], process.stdin, deadline)
                report["sender_verified_members"].append(name)
            marker = completion_marker(request)
            if process.stdin.write(marker) != len(marker):
                raise OSError("incomplete SSH completion marker")
        except (OSError, ValueError, EOFError) as error:
            send_error = type(error).__name__
        try:
            process.stdin.close()
        except OSError as error:
            send_error = send_error or type(error).__name__
        report["sender_error_type"] = send_error
        terminal = event_line(process.stdout, request)
        report["ssh_exit"] = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if (terminal.get("event") == "COMPLETE" and report["ssh_exit"] == 0 and send_error is None
                and terminal.get("installed_bytes") == expected
                and re.fullmatch(r"[a-f0-9]{64}", str(terminal.get("receipt_SHA256")))):
            report.update(state="RECEIVER_REPORTED_COMPLETE", remote_receipt_SHA256=terminal["receipt_SHA256"])
        elif terminal.get("event") == "FAILED_RETAINED" and report["ssh_exit"] == 1:
            report["state"] = "RECEIVER_REPORTED_FAILED_RETAINED"
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    options = parser.parse_args()
    published_package(options.publication)
    claim_bytes = (PACKAGE / "claim.json").read_bytes()
    claim = json.loads(claim_bytes)
    validate_manifest(claim)
    verify_metadata(claim)
    http = load_pinned_module("clip_http", HTTP_HELPER, HTTP_HELPER_SHA256)
    ssh = load_pinned_module("hygon_ssh", SSH_HELPER, SSH_HELPER_SHA256)
    command = ssh.ssh(REPO.parent)
    command.remove("-v")
    report_dir = REPO / "local-only" / ("hymotion-clip-L-install-" + claim["request"])
    report_dir.mkdir()
    identity = dict(request=claim["request"], claim_SHA256=sha256(claim_bytes).hexdigest(),
                    publication=options.publication)
    save_json(report_dir / "intent.json", identity)
    report = dict(identity, state="UNKNOWN", remote_ready_observed=False, sender_verified_members=[])
    try:
        transfer(command, claim_bytes, claim, build_opener(http.OfficialRedirects()), http.BASE, report)
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as error:
        report["observed_error_type"] = type(error).__name__
    finally:
        save_json(report_dir / "terminal.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["state"] == "RECEIVER_REPORTED_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
