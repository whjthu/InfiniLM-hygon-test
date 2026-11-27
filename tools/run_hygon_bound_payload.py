"""Send a published project payload only after fresh authority on the same SSH session."""

from base64 import b64encode
from hashlib import sha256
import json
import os
from queue import Empty, Queue
import re
import shlex
import subprocess
from threading import Thread
import time
from uuid import uuid4


CAP = 560000
GATE = r'''
from base64 import b64decode
from hashlib import sha256
import io, json, os, sys
from pathlib import Path

assert os.getuid() == 154459
assert Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results').is_dir()
print(json.dumps({'event':'BOUND', **BINDING},sort_keys=True),flush=True)
line = sys.stdin.buffer.readline(CAP + 2)
assert 1 < len(line) <= CAP + 1 and line.endswith(b'\n')
raw = b64decode(line[:-1],validate=True)
assert sha256(raw).hexdigest() == BINDING['payload_sha256']
packet = json.loads(raw)
assert set(packet) == {'program','data'}
program = packet['program']
data = b64decode(packet['data'],validate=True)
print(json.dumps({'event':'RECEIVED', **BINDING},sort_keys=True),flush=True)
sys.stdin = io.TextIOWrapper(io.BytesIO(data),encoding='utf-8')
namespace = {'__name__':'__main__'}
exec(compile(program,'<bound project payload>','exec'),namespace)
'''


def digest(value):
    return sha256(value).hexdigest()


def document(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def send(ssh_argv, *, request, program, data, send_token, authority_seconds=18, outcome_seconds=160,
         packet_cap=CAP, write_seconds=20):
    """Return explicit authority, send attempt, receipt and original outcome boundaries."""
    if not re.fullmatch(r"[0-9a-f]{32}", request) or send_token.exists():
        raise ValueError("request identity is invalid or already committed for send")
    compile(program, "<project payload>", "exec")
    packet = document({"program": program, "data": b64encode(data).decode("ascii")})
    payload = b64encode(packet)
    if not 0 < packet_cap <= 32 << 20 or not 0 < write_seconds <= 180 or len(payload) > packet_cap:
        raise ValueError("project payload exceeds the authenticated packet cap")
    binding = {"request": request, "nonce": uuid4().hex, "payload_sha256": digest(packet)}
    remote = f"BINDING={binding!r}\nCAP={packet_cap}\n" + GATE
    command = [*ssh_argv, shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", remote])]
    result = {"request": request, "authority_verified": False, "payload_send_attempted": False,
              "payload_flushed": False, "remote_received": False, "response": None,
              "payload_sha256": binding["payload_sha256"], "remote_outcome": "UNOBSERVED"}
    stdout = Queue()
    stderr = sha256()

    def scan_stdout(stream):
        with stream:
            while line := stream.readline(131073):
                stdout.put(line)
            stdout.put(None)

    def scan_stderr(stream):
        with stream:
            for block in iter(lambda: stream.read(16384), b""):
                stderr.update(block)

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_thread = Thread(target=scan_stdout, args=(process.stdout,), daemon=True)
    err_thread = Thread(target=scan_stderr, args=(process.stderr,), daemon=True)
    out_thread.start()
    err_thread.start()
    transcript = sha256()
    marker = {"event": "BOUND", **binding}
    accepted = {"event": "RECEIVED", **binding}
    writer = None

    def next_message(deadline):
        while time.monotonic() < deadline:
            wait = max(0, deadline - time.monotonic())
            try:
                line = stdout.get(timeout=wait)
            except Empty:
                return None
            if line is None:
                return None
            transcript.update(line)
            if len(line) > 131072:
                raise ValueError("response exceeded the structured result cap")
            try:
                row = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            if isinstance(row, dict):
                return row
        return None

    try:
        end = time.monotonic() + authority_seconds
        while (message := next_message(end)) is not None:
            if message == marker:
                result["authority_verified"] = True
                break
        if result["authority_verified"]:
            with send_token.open("xb") as f:
                f.write(document(binding))
                f.flush()
                os.fsync(f.fileno())
            result["payload_send_attempted"] = True
            send_status = Queue()

            def write_payload():
                try:
                    process.stdin.write(payload + b"\n")
                    process.stdin.flush()
                except (OSError, ValueError) as error:
                    send_status.put(error)
                else:
                    send_status.put(None)

            writer = Thread(target=write_payload, daemon=True)
            writer.start()
            try:
                send_error = send_status.get(timeout=write_seconds)
            except Empty:
                raise TimeoutError("authenticated payload write timed out") from None
            if send_error:
                raise send_error
            result["payload_flushed"] = True
            process.stdin.close()
            end = time.monotonic() + outcome_seconds
            while (message := next_message(end)) is not None:
                if message == accepted:
                    result["remote_received"] = True
                elif result["remote_received"] and message.get("request") == request:
                    if result["response"] is not None:
                        raise ValueError("multiple original outcome records")
                    result["response"] = message
        else:
            result["remote_outcome"] = "PAYLOAD_NOT_SENT"
    except (OSError, ValueError) as error:
        result["local_error_type"] = type(error).__name__
    finally:
        if writer is None or not writer.is_alive():
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            result["rc"] = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            result["rc"] = None
        if writer is not None:
            writer.join(timeout=2)
        out_thread.join(timeout=2)
        err_thread.join(timeout=2)
    if result["remote_received"] and result["response"] is not None and result["rc"] == 0 and "local_error_type" not in result:
        result["remote_outcome"] = "RESPONSE_OBSERVED"
    elif result["payload_send_attempted"]:
        result["remote_outcome"] = "UNKNOWN"
    else:
        result["remote_outcome"] = "PAYLOAD_NOT_SENT"
    result["stdout_sha256"] = transcript.hexdigest()
    result["stderr_sha256"] = stderr.hexdigest()
    return result
