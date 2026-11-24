"""Capture the exact binary install packet and exercise the receiver on temporary paths."""

from __future__ import annotations

from base64 import b64encode
import json
from pathlib import Path
import shlex
import sys
import tempfile

import install
import prepare


STAGE_ID = "0c7ca351c21c48b4a8b803aa5a2979d2"
RAW_ENVELOPE = prepare.REPO / "local-only/O01-pack16-preflight-v3-c0243062d1f24b70aee7e3092c24e8f0/original-outcome.json"


def reject(call, state):
    try:
        call()
    except ValueError as error:
        assert str(error) == state, error
    else:
        raise AssertionError("invalid stage accepted: " + state)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one new private local receipt path")
    expected_work = prepare.REPO / "local-only/o01-pack16-2930-target-source-packet-20261125"
    raw = prepare.packet("reuse")
    digest = install.pin(raw)["SHA256"]
    remote = prepare.remote_argv(STAGE_ID, raw)
    assert (expected_work / "packet.json").read_bytes() == raw
    assert shlex.split((expected_work / "remote-command.txt").read_text(encoding="utf-8")) == remote
    assert remote[:8] == ["/usr/bin/timeout", "--kill-after=5s", "55s", "/usr/local/python3.12/bin/python3",
                          "-I", "-S", "-B", "-c"] and remote[-4:] == ["--stage-id", STAGE_ID, "--packet-sha256", digest]
    files, closure, closure_raw, mode = install.decode_packet(raw, digest)
    assert len(files) == 20 and mode == "reuse" and len(closure["members"]) == 26
    assert not any(name.startswith(("runtime/", "local-only/")) for name in files)
    assert install.pin(closure_raw)["SHA256"] == install.CLOSURE_SHA
    logical = install.ROOT / "incoming" / install.stage_name(STAGE_ID)
    original = install.read_exact(RAW_ENVELOPE, 4664, install.HISTORICAL_ENVELOPE_SHA)
    with tempfile.TemporaryDirectory(prefix="pack16-shared-host-install-capture-") as directory:
        root = Path(directory).resolve()
        response = root / "launch-and-accounting.json"
        response.write_bytes(install.encode(json.loads(original)["response"]))
        assert install.pin(response.read_bytes())["SHA256"] == install.HISTORICAL_RESPONSE_SHA
        result = install.install(raw, digest, root / "reuse", logical, prepare.RENDERED, response)
        report = result["result"]
        assert report["runtime_copied_members"] == 0 and not (root / "reuse/runtime").exists()
        assert report["historical_dependencies"]["target_response"] == "EXACT_ARCHIVED_TARGET_RESPONSE"
        assert report["historical_dependencies"]["local_transport_envelope"] == "ABSENT"
        assert report["host_static"]["original_full_frozen_host"] == "BLOCKED_EXACT_TRANSPORT_ENVELOPE_UNAVAILABLE"
        assert report["host_static"]["source_loader_verified"] and report["host_static"]["actual_HOST_help_checked"]
        assert not report["host_static"]["host_argv_dispatched"]
        assert not report["ordinary_coordinator_adopted"] and not report["target_compute_node_checked"]
        reject(lambda: install.install(raw, digest, root / "reuse", logical, prepare.RENDERED, response), "STAGE_NOT_EXCLUSIVE")

        strict = install.install(raw, digest, root / "exact-local-archived-envelope",
                                 logical, prepare.RENDERED, response, RAW_ENVELOPE)
        assert strict["result"]["host_static"]["original_full_frozen_host"] == "EXACT_ORIGINAL_FROZEN_HOST_CHECK_PASSED"
        assert not strict["result"]["original_transport_envelope_transferred"]

        inclusion = prepare.packet("include")
        staged = install.install(inclusion, install.pin(inclusion)["SHA256"], root / "include", logical,
                                 root / "unused", root / "absent-response")
        assert staged["result"]["runtime_copied_members"] == 6 and staged["result"]["decoded_source_members"] == 26

        polluted = json.loads(raw)
        polluted["files"]["local-only/private-original-outcome.json"] = b64encode(b"fixture only").decode("ascii")
        mutation = install.encode(polluted)
        reject(lambda: install.install(mutation, install.pin(mutation)["SHA256"], root / "unexpected",
                                       logical, prepare.RENDERED, response), "PACKET_MEMBER_SET")
        malformed_runtime = root / "malformed"
        malformed_runtime.mkdir()
        reject(lambda: install.install(raw, digest, root / "bad-runtime", logical, malformed_runtime, response),
               "RUNTIME_DIRECTORY_SCOPE")
        assert not (root / "unexpected").exists() and not (root / "bad-runtime").exists()
        assert not any("original-outcome" in str(path.relative_to(root)) for path in root.rglob("*"))

    receipt = {"schema": install.SCHEMA + "-local-test", "closure_SHA256": install.CLOSURE_SHA,
               "packet": install.pin(raw), "captured_remote_command": install.pin((expected_work / "remote-command.txt").read_bytes()),
               "reuse_result": result, "strict_original_envelope_local_only": strict,
               "included_runtime_fixture": staged,
               "checks": ["actual intake/readback", "no copied reuse runtime", "actual HOST parser and argv",
                          "separate exact historical response and original envelope", "exclusive stage/no repeat",
                          "unexpected private member and incomplete runtime rejected before writes"],
               "actual_ssh": 0, "actual_Slurm": 0, "actual_GPU": 0}
    data = install.encode(receipt)
    install.save(Path(sys.argv[1]), data)
    print(json.dumps(install.pin(data)))


if __name__ == "__main__":
    main()
