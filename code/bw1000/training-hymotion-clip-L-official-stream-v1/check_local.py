"""Exercise the real stream and publication boundaries with small in-memory fixtures."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import launch_once as sender
import receive_once as receiver


def pin(data):
    return {"bytes": len(data), "SHA256": sha256(data).hexdigest()}


class Response(BytesIO):
    status = 200

    def __init__(self, body, length, *, encoding="identity"):
        super().__init__(body)
        self.headers = {"Content-Length": str(length), "Content-Encoding": encoding}


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        assert request.get_header("Accept-encoding") == "identity" and 0 < timeout <= 15
        return self.response


def reject_copy(response, expected, target, deadline, kind):
    try:
        sender.copy_member(Opener(response), "https://example.org/", "fixture", expected, target, deadline)
    except kind:
        return
    raise AssertionError("sender accepted invalid stream")


def main():
    claim = json.loads((sender.PACKAGE / "claim.json").read_bytes())
    receiver.validate_manifest(claim)
    sender.verify_metadata(claim)
    try:
        receiver.validate_manifest(dict(claim, budget_seconds=receiver.BUDGET_SECONDS + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("receiver accepted the altered scope")
    assets = {name: f"fixture for {name}".encode("ascii") for name in receiver.MEMBERS}
    pins = {name: pin(payload) for name, payload in assets.items()}
    payload = b"".join(assets.values())
    marker = receiver.completion_marker(claim["request"])
    cases = {
        "exact": (payload + marker, True),
        "truncated": (payload[:-1], False),
        "different_member": (b"?" + payload[1:] + marker, False),
        "unconfirmed": (payload, False),
        "trailing_bytes": (payload + marker + b"?", False),
    }
    for label, (body, accepted) in cases.items():
        with tempfile.TemporaryDirectory() as fixture, redirect_stdout(StringIO()) as output:
            project = Path(fixture)
            parent = project / "candidates"
            parent.mkdir()
            destination = parent / receiver.LEAF
            data = dict(claim, destination=destination.as_posix(), files=pins)
            with patch.object(receiver, "PROJECT", project), patch.object(
                receiver, "MODEL_PIN", pins["model.safetensors"]
            ):
                rc = receiver.install(data, "0" * 64, BytesIO(body))
                assert (rc == 0) == accepted, label
                assert destination.is_dir() == accepted, label
                notices = BytesIO(output.getvalue().encode("ascii"))
                assert sender.event_line(notices, claim["request"])["event"] == "READY", label
                final = sender.event_line(notices, claim["request"])
                if accepted:
                    receipt = (destination / "receipt.json").read_bytes()
                    assert final["receipt_SHA256"] == sha256(receipt).hexdigest()
                    assert json.loads(receipt)["files"] == pins
                else:
                    stage = parent / ("." + receiver.LEAF + ".incoming-" + claim["request"])
                    assert (stage / "failure.json").is_file()
                    assert final["event"] == "FAILED_RETAINED"
                try:
                    receiver.install(data, "0" * 64, BytesIO(payload + marker))
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("receiver replay was accepted")
    body = assets["config.json"]
    expected = pin(body)
    passed = BytesIO()
    active = time.monotonic() + 5
    sender.copy_member(Opener(Response(body, len(body))), "https://example.org/", "fixture", expected,
                       passed, active)
    assert passed.getvalue() == body
    reject_copy(Response(body, len(body), encoding="gzip"), expected, BytesIO(), active, ValueError)
    reject_copy(Response(body, len(body) - 1), expected, BytesIO(), active, ValueError)
    reject_copy(Response(body[:-1], len(body)), expected, BytesIO(), active, EOFError)
    reject_copy(Response(body + b"?", len(body)), expected, BytesIO(), active, ValueError)
    reject_copy(Response(b"?" + body[1:], len(body)), expected, BytesIO(), active, ValueError)
    reject_copy(Response(body, len(body)), expected, BytesIO(), 0, TimeoutError)
    print(json.dumps(dict(receiver_cases=len(cases), duplicate_rejections=len(cases),
                          sender_cases=7, bad_scope_rejections=1, network_calls=0, slurm_steps=0)))


if __name__ == "__main__":
    main()
