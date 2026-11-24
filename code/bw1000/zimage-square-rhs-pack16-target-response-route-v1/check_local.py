"""Independently compare the true local envelope and archived-response HOST routes."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile

import host as route


ROOT = Path(__file__).resolve().parents[3]
PROOF = ROOT / "coordination/tasks/O01-pack16-2662-target-response-identity-proof-v1.json"
ARCHIVE = ROOT / route.ARCHIVE_PATH
ENCODER = ROOT / route.ENCODER_PATH
ORIGINAL = ROOT / "local-only/O01-pack16-preflight-v3-c0243062d1f24b70aee7e3092c24e8f0/original-outcome.json"
RUNTIME = ROOT / "local-only/O01-pack16-2930-device-runtime-candidate-20261125"
FIXTURE = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/fixture-historical-response"
EPOCH = 1795500000
BOOT = "00000000-0000-4000-8000-000000000001"
NODE = "f99r9n99"


def rejects(call, gate):
    try:
        call()
    except ValueError as error:
        assert str(error) == gate, error
    else:
        raise AssertionError("unexpected acceptance")


def put(path, doc, host):
    raw = host.encode(doc)
    with path.open("xb") as stream:
        stream.write(raw)
    return route.pin(raw)["SHA256"]


def dry_run(entry, root, source, host):
    pcis = [f"0000:{idx:02x}:00.0" for idx in range(8)]
    pre_step = "1103606.901"
    pre = {"phase": "pre", "observed_epoch": EPOCH - 2, "assigned_PCI": pcis[0], "gfx": "gfx936",
           "exclusive_1_DCU_lease": True,
           "accounting": [pre_step, "fixture", "COMPLETED", "0:0", "1", NODE, "cpu=1,mem=256M,node=1"],
           "inventory": {"node": NODE, "step": pre_step, "PCIs": pcis,
                         "boot_id_SHA256": route.pin((BOOT + "\n").encode())["SHA256"],
                         "kfd_process_count": 0, "own_device_fd_process_count": 0}}
    pre_path, scope_path, protection_path = (root / item for item in ("pre.json", "scope.json", "protection.json"))
    protection = {"parent": host.PARENT, "selected_node": NODE, "protected_nodes": ["f12r2n11"],
                  "protected_steps": ["1103606.2559"], "observed_epoch": EPOCH - 2,
                  "valid_until_epoch": EPOCH + 50}
    if not pre_path.exists():
        pre_sha = put(pre_path, pre, host)
        protection_sha = put(protection_path, protection, host)
        scope = {"request": host.REQUEST, "label": host.LABEL, "parent": host.PARENT, "node": NODE, "gfx": "gfx936",
                 "source_publication": host.PUBLICATION, "runtime_SHA256": host.RUNTIME_SHA, "root_admitted": True,
                 "torch_version": "2.10.0", "torch_hip": "6.3.26113", "protected_nodes": ["f12r2n11"],
                 "PCI": pcis[0], "boot_id": BOOT, "epoch": EPOCH, "deadline_epoch": EPOCH + 240,
                 "pre_step": pre_step, "pre_SHA256": pre_sha, "protection_authority_SHA256": protection_sha}
        put(scope_path, scope, host)
    scope_pin = route.pin(scope_path.read_bytes())["SHA256"]
    outer = sys.argv
    stream = io.StringIO()
    try:
        sys.argv = ["fixture-host", "dry-run", "--repo", str(ROOT), "--runtime", str(RUNTIME),
                    "--original-dependencies", str(source), "--scope", str(scope_path), "--pre", str(pre_path),
                    "--protection", str(protection_path), "--scope-sha256", scope_pin,
                    "--target-runtime", FIXTURE + "/runtime", "--target-scope", FIXTURE + "/scope.json",
                    "--target-output", FIXTURE + "/output", "--observed-now", str(EPOCH + 1)]
        with redirect_stdout(stream):
            assert entry.main() == 0
    finally:
        sys.argv = outer
    return json.loads(stream.getvalue())


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one absent private receipt path")
    raw = route.exact(ORIGINAL, 4664, route.RAW_WRAPPER_SHA)
    wire = json.loads(raw)
    assert all(wire[key] is True for key in ("authority_verified", "payload_flushed",
                                            "payload_send_attempted", "remote_received"))
    encode = route.original_encoder(route.exact(ENCODER, 5191, route.ENCODER_SHA))
    actual_response = encode(wire["response"])
    assert route.pin(actual_response) == {"bytes": 3049, "SHA256": route.RESPONSE_SHA}
    proof_doc = json.loads(route.exact(PROOF, 2288, route.PROOF_SHA))
    public = json.loads(route.exact(ROOT / route.PUBLIC_PATH, 4342, route.PUBLIC_SHA))
    archive = json.loads(route.exact(ARCHIVE, 3390, route.ARCHIVE_SHA))
    response = json.loads(actual_response)
    route.evidence_fields(proof_doc, public, response, archive)
    rejects(lambda: route.evidence_fields(dict(proof_doc, coordinator_verified_local_original=dict(
        proof_doc["coordinator_verified_local_original"], payload_flushed=False)), public, response, archive),
        "EXACT_COORDINATOR_WIRE_FLAGS")
    rejects(lambda: route.evidence_fields(proof_doc, public, dict(response, request="x"), archive),
            "EXACT_ORIGINAL_2662_IDENTITY")
    wrong_row = list(route.ACCOUNTING)
    wrong_row[6] += ",gres/dcu=1"
    rejects(lambda: route.evidence_fields(proof_doc, public, dict(response, accounting=[wrong_row]), archive),
            "EXACT_ORIGINAL_2662_IDENTITY")
    rejects(lambda: route.evidence_fields(proof_doc, public, dict(response, step_result=dict(
        response["step_result"], Slurm_GPU_resources=1)), archive), "EXACT_ZERO_GRES_STEP_RESULT")
    with tempfile.TemporaryDirectory(prefix="pack16-response-route-capture-") as scratch:
        path = Path(scratch).resolve()
        response_file = path / "launch-and-accounting.json"
        response_file.write_bytes(actual_response)
        original = route.load_entry(ROOT)
        baseline = original.load_baseline()
        primary = dry_run(original, path, ORIGINAL, baseline)
        alternate = route.load_entry(ROOT)
        route.attach(alternate, response_file, PROOF, ARCHIVE, ENCODER)
        candidate = dry_run(alternate, path, response_file, baseline)
        fields = ("frozen_pins", "runtime", "independent_reader")
        assert {key: candidate["sources"][key] for key in fields} == {key: primary["sources"][key] for key in fields}
        assert candidate["argv"] == primary["argv"] and candidate["planned"] == primary["planned"]
        assert primary["target_calls"] == candidate["target_calls"] == 0
        assert candidate["sources"]["private_transport_wrapper_read_here"] is False
        assert not candidate["sources"]["permits_new_device_step"]
        wrong_proof = path / "proof.json"
        wrong_proof.write_bytes(PROOF.read_bytes() + b"\n")
        rejects(lambda: route.frozen_target(baseline, ROOT, RUNTIME, response_file, wrong_proof, ARCHIVE, ENCODER),
                "ALTERNATE_EVIDENCE_BYTES")
        response_file.write_bytes(actual_response[:-1] + b" ")
        rejects(lambda: route.frozen_target(baseline, ROOT, RUNTIME, response_file, PROOF, ARCHIVE, ENCODER),
                "ALTERNATE_EVIDENCE_BYTES")
    receipt = {"schema": "pack16-2662-alternate-response-local-verification-v1",
               "primary_and_alternate_full_HOST_dry_run": True,
               "common_static_fields": fields, "identical_captured_argv": True,
               "argv_SHA256": route.pin(baseline.encode(primary["argv"]))["SHA256"],
               "response": route.pin(actual_response), "actual_coordinator_proof_SHA256": route.PROOF_SHA,
               "expected_independent_readback_SHA256": route.INDEPENDENT_SHA,
               "original_2662_accounting": route.ACCOUNTING,
               "negative_checks": ["response bytes", "proof bytes", "request", "one-DCU substituted row",
                                   "step_result mutation", "coordinator false wire flag"],
               "new_remote_calls": 0, "new_Slurm_calls": 0, "new_GPU_calls": 0}
    data = baseline.encode(receipt)
    with Path(sys.argv[1]).open("xb") as stream:
        stream.write(data)
    print(json.dumps(route.pin(data)))


if __name__ == "__main__":
    main()
