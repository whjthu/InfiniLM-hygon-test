"""Send the frozen household VGGT public archive through the installed carrier."""

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path, PurePosixPath
import queue
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import threading
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "code/bw1000/training-vggt-household-native-input-v1"
CLAIM_SHA = "440f1db54d2cbb04e78c0df793bd84709f42fd900b9ce79cfff82609b48f729b"
ORIGINAL_NO_SEND_RECEIPT_SHA = "dbe9d68ffab11f838e12c7584f652ac858d6934a85f3e8ac0ed11fa572442884"
RESPONSE_CAP = 8192
PREEXISTING = (
    "transfer-intent.json",
    "assets",
    "assets.partial",
    "assets-receipt.json",
    "cpu-intent.json",
    "cpu-receipt.json",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def original_package():
    raw = (PACKAGE / "CLAIM.json").read_bytes()
    require(sha256(raw).hexdigest() == CLAIM_SHA, "original public claim changed")
    claim = json.loads(raw)
    for name, expected in claim["files"].items():
        data = (PACKAGE / name).read_bytes()
        require(len(data) == expected["bytes"] and sha256(data).hexdigest() == expected["SHA256"],
                f"original package file changed: {name}")
    sys.path.insert(0, str(PACKAGE))
    try:
        module = importlib.import_module("transfer")
    finally:
        sys.path.pop(0)
    require(Path(module.__file__).resolve() == PACKAGE / "transfer.py", "unexpected sender import")
    return claim, module


def remote_carrier(carrier):
    path = PurePosixPath(carrier.RUN.as_posix()) / "carrier.py"
    require(path.is_absolute() and "\\" not in path.as_posix(), "remote carrier path is not POSIX")
    return path


def require_original_no_send(local, packet):
    directory = local / ("vggt-household-selected-transfer-" + packet["request"])
    raw = (directory / "receipt.json").read_bytes()
    require(sha256(raw).hexdigest() == ORIGINAL_NO_SEND_RECEIPT_SHA, "original no-send receipt differs")
    receipt = json.loads(raw)
    original = json.loads((directory / "intent.json").read_bytes())
    require(not (directory / "send.json").exists() and not (directory / "send.json").is_symlink()
            and receipt["outcome"] == "PAYLOAD_NOT_SENT"
            and receipt["request"] == packet["request"]
            and receipt["payload_send_attempted"] is False and receipt["authority_verified"] is False
            and all(original.get(key) == value for key, value in packet.items() if key != "nonce"),
            "previous execution or payload send is unresolved")
    return sha256(raw).hexdigest()


def stderr_pin(spool):
    digest = sha256()
    total = 0
    spool.seek(0)
    for part in iter(lambda: spool.read(65536), b""):
        total += len(part)
        digest.update(part)
    return {"bytes": total, "SHA256": digest.hexdigest()}


def exchange(argv, payload, packet, carrier, destination, *, authority_timeout=22, reply_timeout=125):
    expected_authority = dict(packet, event="BOUND")
    outcome = dict(request=packet["request"], authority_verified=False, payload_send_attempted=False,
                   outcome="PAYLOAD_NOT_SENT", response=None, rc=None)
    carrier.write(destination / "intent.json", carrier.encode(packet))
    reply = b""
    with tempfile.TemporaryFile(mode="w+b") as errors:
        process = None
        reader = None
        try:
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            events = queue.Queue(maxsize=1)

            def read_authority():
                try:
                    events.put(process.stdout.readline(RESPONSE_CAP + 1))
                except OSError as error:
                    events.put(error)

            reader = threading.Thread(target=read_authority, daemon=True)
            reader.start()
            line = events.get(timeout=authority_timeout)
            reader.join(timeout=1)
            require(not reader.is_alive() and isinstance(line, bytes) and 0 < len(line) <= RESPONSE_CAP,
                    "bounded fresh authority missing")
            require(json.loads(line) == expected_authority, "same-session authority differs")
            outcome["authority_verified"] = True
            carrier.write(destination / "send.json", carrier.encode(expected_authority))
            outcome["payload_send_attempted"] = True
            outcome["outcome"] = "UNKNOWN"
            reply, _ = process.communicate(input=payload, timeout=reply_timeout)
            require(process.returncode == 0 and 0 < len(reply) <= RESPONSE_CAP, "transfer receipt missing")
            answer = json.loads(reply)
            require(isinstance(answer, dict) and all(answer.get(k) == v for k, v in packet.items())
                    and answer.get("passed") is True
                    and answer.get("verified_objects") == 79
                    and type(answer.get("actual_GPU")) is int and answer["actual_GPU"] == 0
                    and answer.get("verified_manifest_bytes") == carrier.EXPECTED_BYTES,
                    "transfer receipt identity differs")
            outcome.update(outcome="ASSETS_REHASHED_ON_TARGET", response=answer)
        except (OSError, ValueError, KeyError, TypeError, queue.Empty, subprocess.SubprocessError) as error:
            outcome["error_type"] = type(error).__name__
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    outcome["rc"] = process.wait(timeout=5)
                except (OSError, subprocess.SubprocessError) as error:
                    outcome["cleanup_error_type"] = type(error).__name__
                if reader is not None:
                    reader.join(timeout=1)
                process.stdin.close()
                process.stdout.close()
        outcome["stderr_pin"] = stderr_pin(errors)
    outcome["stdout_pin"] = carrier.pin(reply)
    pin = carrier.write(destination / "receipt.json", carrier.encode(outcome))
    return outcome, pin


def check_installed(ssh, carrier, packet):
    wrapper = """
import json, runpy, sys
from hashlib import sha256
from pathlib import Path
path = Path(sys.argv[1])
with path.open("rb") as stream:
    source = stream.read(32769)
if not 0 < len(source) <= 32768 or sha256(source).hexdigest() != sys.argv[2]:
    raise ValueError("installed carrier source changed")
api = runpy.run_path(str(path))
receipt = api["installation"](sys.argv[3])
markers = {name: (path.parent / name).exists() or (path.parent / name).is_symlink()
           for name in sys.argv[5:]}
print(json.dumps(dict(publication=receipt["publication"], installed_SHA256=sys.argv[3],
                      request=sys.argv[4], markers=markers), sort_keys=True))
"""
    posix_path = remote_carrier(carrier).as_posix()
    argv = [carrier.MANAGED, "-I", "-S", "-B", "-c", wrapper, posix_path,
            sha256((PACKAGE / "carrier.py").read_bytes()).hexdigest(), packet["installed_SHA256"],
            packet["request"], *PREEXISTING]
    result = subprocess.run([*ssh, "exec " + shlex.join(argv)], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=28)
    require(result.returncode == 0 and 0 < len(result.stdout) <= RESPONSE_CAP, "original reconciliation unavailable")
    actual = json.loads(result.stdout)
    expected = {key: packet[key] for key in ("publication", "installed_SHA256", "request")}
    require(actual == dict(expected, markers=dict.fromkeys(PREEXISTING, False)),
            "original namespace or installed identity differs; reconcile only")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-path", action="store_true")
    parser.add_argument("--installed")
    parser.add_argument("--publication")
    args = parser.parse_args()
    claim, original = original_package()
    carrier = original.carrier
    path = remote_carrier(carrier).as_posix()
    if args.inspect_path:
        legacy_path = str(carrier.RUN / "carrier.py")
        print(json.dumps({"portable_absolute_remote_path": True,
                          "legacy_path_is_absolute_on_posix": PurePosixPath(legacy_path).is_absolute(),
                          "legacy_contains_backslash": "\\" in legacy_path,
                          "remote_operations": 0}))
        return 0
    if not re.fullmatch(r"[a-f0-9]{64}", args.installed or "") or not re.fullmatch(r"[a-f0-9]{40}", args.publication or ""):
        parser.error("supply the frozen installed receipt digest and its source publication")
    local = REPO / "local-only"
    destination = local / ("vggt-household-file-stderr-transfer-" + carrier.TRANSFER)
    require(not destination.exists() and not destination.is_symlink(),
            "local original request must be reconciled")
    payload = original.make_tar()
    require(carrier.pin(payload) == claim["selected_ustar"], "frozen public archive bytes differ")
    require(carrier.pin(original.SSH_HELPER.read_bytes())["SHA256"] == original.SSH_HELPER_SHA,
            "the existing authenticated route changed")
    ssh = runpy.run_path(str(original.SSH_HELPER))["ssh"](REPO.parent)
    packet = dict(request=carrier.TRANSFER, nonce=uuid4().hex, publication=args.publication,
                  input_SHA256=carrier.INPUT_SHA, tar_SHA256=claim["selected_ustar"]["SHA256"],
                  bytes=len(payload), installed_SHA256=args.installed)
    prior_receipt_sha = require_original_no_send(local, packet)
    reconciliation = check_installed(ssh, carrier, packet)
    destination.mkdir(mode=0o700)
    carrier.write(destination / "original-reconciliation.json",
                  carrier.encode(dict(reconciliation, local_prior_receipt_SHA256=prior_receipt_sha)))
    argv = [carrier.MANAGED, "-I", "-S", "-B", "-u", path, "transfer", packet["request"],
            packet["nonce"], packet["installed_SHA256"], packet["tar_SHA256"], str(packet["bytes"])]
    outcome, pin = exchange([*ssh, "exec " + shlex.join(argv)], payload, packet, carrier, destination)
    print(json.dumps(dict(outcome=outcome["outcome"], request=packet["request"], receipt_pin=pin)))
    return 0 if outcome["outcome"] == "ASSETS_REHASHED_ON_TARGET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
