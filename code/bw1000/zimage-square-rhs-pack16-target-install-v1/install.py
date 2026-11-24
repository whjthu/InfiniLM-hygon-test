"""Install the exact Pack16 HOST/source closure without any Slurm operation."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from base64 import b64decode
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


SCHEMA = "bw1000.o01.pack16.target-install-closure-v1"
CLOSURE_SHA = "8d8ad7fa2594ac949ca7e3475cc22def5c489b8dca39158f8677876e3030bb84"
PUBLICATION = "358b255028a1ae961e5137ae14ca98a7efbb6dee"
ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
HISTORICAL_RUNTIME = ROOT / "incoming/O01-pack16-deps-json-c0243062d1f2-reviewed/runtime"
HISTORICAL_RESPONSE = ROOT / "results/O01-pack16-deps-json-c0243062d1f2/launch-and-accounting.json"
HISTORICAL_RESPONSE_SHA = "d12faee2177a9e7e82076e333b1bba03ce55151381413b2d2d9767b035bf2edf"
HISTORICAL_ENVELOPE_SHA = "f13cc42439079e0dde3b74d6fa2e371559a433ba42910cd04eb08bb613584262"
RUNTIME_NAMES = {"SCOPED_FILES.json", "baseline.py", "candidate.py", "contract.py", "hipprof_trace.py", "worker.py"}
RUNTIME_SHA = "999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
HOST_PATH = "code/bw1000/zimage-square-rhs-pack16-host-v1"
ARBITER_PATH = "code/bw1000/zimage-square-rhs-pack16-ordinary-handoff-v1"
CURRENT_EPOCH = 1795500000  # Deliberately historical and confined to the fixture argv.
MAX_PACKET = 262144
SHA = re.compile("[0-9a-f]{64}")
STAGE_ID = re.compile("[0-9a-f]{32}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(data):
    return {"bytes": len(data), "SHA256": sha256(data).hexdigest()}


def encode(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()


def read_exact(path, maximum, digest=None):
    require(path.is_file() and path.resolve(strict=True) == path.absolute() and not path.is_symlink(),
            "SOURCE_ABSENT_OR_ALIASED")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    require(0 < len(payload) <= maximum and (digest is None or pin(payload)["SHA256"] == digest), "SOURCE_PIN")
    return payload


def stage_name(stage_id):
    require(STAGE_ID.fullmatch(stage_id), "STAGE_ID")
    return "O01-pack16-2930-host-install-" + stage_id


def decode_packet(raw, digest):
    require(SHA.fullmatch(digest) and len(raw) <= MAX_PACKET and pin(raw)["SHA256"] == digest, "PACKET_PIN")
    packet = json.loads(raw)
    require(set(packet) == {"closure", "files", "runtime_mode"} and packet["runtime_mode"] in ("reuse", "include"),
            "PACKET_SCHEMA")
    closure_raw = b64decode(packet["closure"], validate=True)
    require(pin(closure_raw)["SHA256"] == CLOSURE_SHA, "CLOSURE_PIN")
    closure = json.loads(closure_raw)
    require(closure["schema"] == SCHEMA and closure["publication"] == PUBLICATION
            and len(closure["members"]) == 26, "CLOSURE_SCOPE")
    runtime = {name for name in closure["members"] if name.startswith("runtime/")}
    require(runtime == {"runtime/" + name for name in RUNTIME_NAMES}, "CLOSURE_RUNTIME_SCOPE")
    wanted = set(closure["members"]) if packet["runtime_mode"] == "include" else set(closure["members"]) - runtime
    require(set(packet["files"]) == wanted, "PACKET_MEMBER_SET")
    files = {}
    for name, encoded in packet["files"].items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and str(part) == name and ".." not in part.parts,
                "PACKET_PATH_SCOPE")
        data = b64decode(encoded, validate=True)
        expected = closure["members"][name]
        require(pin(data) == {key: expected[key] for key in ("bytes", "SHA256")}, "PACKET_MEMBER_PIN")
        if part.suffix == ".py":
            ast.parse(data, filename=name, feature_version=(3, 10))
        files[name] = data
    return files, closure, closure_raw, packet["runtime_mode"]


def exact_runtime(runtime, closure):
    require(runtime.is_dir() and runtime.resolve(strict=True) == runtime.absolute()
            and set(item.name for item in runtime.iterdir()) == RUNTIME_NAMES, "RUNTIME_DIRECTORY_SCOPE")
    for name in RUNTIME_NAMES:
        expected = closure["members"]["runtime/" + name]
        data = read_exact(runtime / name, expected["bytes"], expected["SHA256"])
        require(len(data) == expected["bytes"], "RUNTIME_MEMBER_SIZE")
    return {"runtime_manifest_SHA256": RUNTIME_SHA, "runtime_members": 5, "runtime_manifest_members": 1}


def history(closure, files, target_response, envelope):
    public_path = next(name for name in closure["members"] if name.endswith("original2662-zero-GRES-dependencies-pass.json"))
    public = json.loads(files[public_path])
    require(public["raw_transport"]["sha256"] == HISTORICAL_ENVELOPE_SHA
            and public["accounting"][0:4] == ["1103606.2662", "O01-pack16-deps-json-c0243062d1f2", "COMPLETED", "0:0"],
            "HISTORICAL_PUBLIC_IDENTITY")
    observed = {"public_dependencies_SHA256": closure["members"][public_path]["SHA256"],
                "target_response": "ABSENT", "local_transport_envelope": "ABSENT"}
    if target_response.exists() or target_response.is_symlink():
        try:
            response = read_exact(target_response, 4096, HISTORICAL_RESPONSE_SHA)
            target = json.loads(response)
            require(target["request"] == public["request"] and target["srun_rc"] == 0
                    and target["GPU_GO"] is False and target["accounting"] == [public["accounting"]]
                    and target["step_result"] == public["precheck"], "HISTORICAL_TARGET_CONTENT")
            observed.update(target_response="EXACT_ARCHIVED_TARGET_RESPONSE", target_response_pin=pin(response),
                            historical_step="1103606.2662", historical_GPU=0)
        except (OSError, ValueError, KeyError, TypeError):
            observed["target_response"] = "PRESENT_WITHOUT_POSITIVE_PROOF"
    if envelope is not None:
        payload = read_exact(envelope, 4664, HISTORICAL_ENVELOPE_SHA)
        require(len(payload) == 4664, "HISTORICAL_ENVELOPE_EXTENT")
        observed.update(local_transport_envelope="EXACT_ENVELOPE_AVAILABLE_IN_PLACE", envelope_pin=pin(payload))
    return observed


def load_source(path, digest, name):
    source = read_exact(path, 65536, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), vars(module))
    return module


def check_host(stage, runtime, target_runtime, logical_stage, closure, envelope):
    path = stage / HOST_PATH / "host.py"
    host = load_source(path, closure["members"][HOST_PATH + "/host.py"]["SHA256"], "installed_pack16_host")
    gate = stage / HOST_PATH / "worker_gate.py"
    read_exact(gate, 16384, host.GATE_SHA)
    arbiter_dir = stage / ARBITER_PATH
    sys.path.insert(0, str(arbiter_dir))
    previous = sys.modules.pop("arbiter", None)
    try:
        entry_path = arbiter_dir / "pack16_host.py"
        entry = load_source(entry_path, closure["members"][ARBITER_PATH + "/pack16_host.py"]["SHA256"],
                            "installed_pack16_entry")
        require(entry.load_baseline().PUBLICATION == host.PUBLICATION, "ADOPTED_HOST_BASELINE")
        previous_argv = sys.argv
        help_text = io.StringIO()
        try:
            sys.argv = [str(entry_path), "--help"]
            with redirect_stdout(help_text):
                try:
                    entry.main()
                except SystemExit as exit:
                    require(exit.code == 0, "ACTUAL_HOST_HELP")
            require("dry-run,run,finish" in help_text.getvalue(), "ACTUAL_HOST_HELP")
        finally:
            sys.argv = previous_argv
    finally:
        sys.modules.pop("arbiter", None)
        if previous is not None:
            sys.modules["arbiter"] = previous
        sys.path.pop(0)
    runtime_proof = host.helper(stage, host.ORIGINAL + "/precheck.py").runtime(runtime)
    strict = "BLOCKED_EXACT_TRANSPORT_ENVELOPE_UNAVAILABLE"
    if envelope is not None:
        host.frozen(stage, runtime, envelope)
        strict = "EXACT_ORIGINAL_FROZEN_HOST_CHECK_PASSED"
    scope = {
        "request": host.REQUEST, "label": host.LABEL, "parent": host.PARENT,
        "source_publication": host.PUBLICATION, "runtime_SHA256": host.RUNTIME_SHA,
        "root_admitted": True, "gfx": "gfx936", "torch_version": "2.10.0", "torch_hip": "6.3.26113",
        "protected_nodes": ["f12r2n11", "f12r2n10"], "node": "f99r9n99", "PCI": "0000:01:00.0",
        "boot_id": "00000000-0000-4000-8000-000000000001", "pre_step": "1103606.901",
        "pre_SHA256": "0" * 64, "protection_authority_SHA256": "0" * 64,
        "epoch": CURRENT_EPOCH, "deadline_epoch": CURRENT_EPOCH + 240,
    }
    argv, limit = host.build_argv(scope, "0" * 64, str(target_runtime),
                                  str(logical_stage / "fixture-scope-never-dispatch.json"),
                                  str(ROOT / "results/fixture-pack16-never-dispatch"),
                                  CURRENT_EPOCH + 1, str(logical_stage / HOST_PATH / "worker_gate.py"))
    require(argv.count("srun") == 1 and argv.count("--gres=dcu:1") == 1 and "--nodelist=f99r9n99" in argv,
            "HOST_FIXTURE_ARGV")
    return {"actual_HOST_static_argv_SHA256": pin(encode(argv))["SHA256"], "fixture_node": "f99r9n99",
            "fixture_client_limit_seconds": limit, "original_full_frozen_host": strict,
            "runtime_proof": runtime_proof, "host_argv_dispatched": False,
            "business_grant_read": False, "source_loader_verified": True, "actual_HOST_help_checked": True}


def save(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def sync(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def install(raw, digest, stage, logical_stage, prior_runtime, target_response, envelope=None):
    files, closure, closure_raw, runtime_mode = decode_packet(raw, digest)
    require(stage.parent.is_dir() and stage.parent.resolve(strict=True) == stage.parent.absolute()
            and not stage.exists() and not stage.is_symlink(), "STAGE_NOT_EXCLUSIVE")
    if runtime_mode == "reuse":
        exact_runtime(prior_runtime, closure)
    historical = history(closure, files, target_response, envelope)
    stage.mkdir(mode=0o700)
    sync(stage.parent)
    save(stage / "INSTALL_INTENT.json", encode({"packet_SHA256": digest, "closure_SHA256": CLOSURE_SHA,
                                                "publication": PUBLICATION, "replay_allowed": False}))
    for name, data in files.items():
        member = stage.joinpath(*PurePosixPath(name).parts)
        member.parent.mkdir(parents=True, exist_ok=True)
        save(member, data)
    save(stage / "SOURCE_CLOSURE.json", closure_raw)
    for name, data in files.items():
        member = stage.joinpath(*PurePosixPath(name).parts)
        require(read_exact(member, len(data), pin(data)["SHA256"]) == data, "STAGE_MEMBER_READBACK")
    runtime = prior_runtime if runtime_mode == "reuse" else stage / "runtime"
    target_runtime = HISTORICAL_RUNTIME if runtime_mode == "reuse" else logical_stage / "runtime"
    runtime_result = exact_runtime(runtime, closure)
    host = check_host(stage, runtime, target_runtime, logical_stage, closure, envelope)
    receipt = {"state": "SHARED_HOST_SOURCE_STAGE_VERIFIED", "packet_SHA256": digest,
               "closure_SHA256": CLOSURE_SHA, "publication": PUBLICATION, "decoded_source_members": len(files),
               "source_readback_bytes": sum(map(len, files.values())), "runtime_mode": runtime_mode,
               "runtime_copied_members": 0 if runtime_mode == "reuse" else 6,
               "runtime": runtime_result, "historical_dependencies": historical, "host_static": host,
               "new_Slurm_calls": 0, "new_GPU_calls": 0, "target_compute_node_checked": False,
               "ordinary_coordinator_adopted": False, "private_calendar_transferred": False,
               "original_transport_envelope_transferred": False, "business_send_grant_issued": False}
    data = encode(receipt)
    save(stage / "INSTALL_RESULT.json", data)
    sync(stage)
    return {"result": receipt, "receipt": pin(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--existing-original-envelope", type=Path)
    args = parser.parse_args()
    require(sys.platform == "linux" and sys.flags.isolated and sys.flags.dont_write_bytecode,
            "ISOLATED_SHARED_LINUX_PROCESS_REQUIRED")
    incoming = ROOT / "incoming"
    stage = incoming / stage_name(args.stage_id)
    original = args.existing_original_envelope
    if original is not None:
        require(original.is_absolute() and original.name == "original-outcome.json"
                and original.is_relative_to(Path(str(ROOT))) and ".." not in original.parts, "ORIGINAL_ENVELOPE_SCOPE")
    os.umask(0o077)
    raw = sys.stdin.buffer.read(MAX_PACKET + 1)
    print(encode(install(raw, args.packet_sha256, Path(str(stage)), stage, Path(str(HISTORICAL_RUNTIME)),
                         Path(str(HISTORICAL_RESPONSE)), original)).decode(), end="")


if __name__ == "__main__":
    main()
