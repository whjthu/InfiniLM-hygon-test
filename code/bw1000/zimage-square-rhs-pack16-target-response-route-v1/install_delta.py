"""Install and validate only the alternate historical proof route on the shared HOST."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PREVIOUS = ROOT / "incoming/O01-pack16-2930-host-install-0c7ca351c21c48b4a8b803aa5a2979d2"
RUNTIME = ROOT / "incoming/O01-pack16-deps-json-c0243062d1f2-reviewed/runtime"
ORIGINAL_RESPONSE = ROOT / "results/O01-pack16-deps-json-c0243062d1f2/launch-and-accounting.json"
PREVIOUS_RESULT_SHA = "85aedc91e4e54d0bcb16df67f51a4e4a8234dfb880eb176659b38e2ce17fe396"
SOURCES = {
    "host.py": "32d24be88e324a0f7b47dc7243c67ea2a3c69d8f72bb4c88ffc26e10e676d627",
    "proof.json": "3726ac745f9d10a45e718e60edc25ca438d28acf9094b41535b4f0d3f5e25fcb",
    "round9.json": "a8e65bf909b11fbcf0c858e7b02e25fef78e67560d106294c99a9f093bc19050",
    "publisher.py": "d200fc8dabed0c3bed3074f3bca262e663a4afe47b35a254073cf1e72e8b9ca9",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def put(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_packet(raw, expected):
    require(0 < len(raw) <= 32768 and digest(raw) == expected, "EXACT_PACKET")
    packet = json.loads(raw)
    require(set(packet) == set(SOURCES), "EXACT_PACKET_MEMBERS")
    decoded = {name: base64.b64decode(data, validate=True) for name, data in packet.items()}
    require(all(digest(decoded[name]) == pin for name, pin in SOURCES.items()), "EXACT_SOURCE")
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--packet-sha256", required=True)
    args = parser.parse_args()
    require(sys.platform == "linux" and sys.flags.isolated and sys.flags.dont_write_bytecode
            and re.fullmatch("[a-f0-9]{32}", args.request), "ISOLATED_LINUX_REQUEST")
    files = verify_packet(sys.stdin.buffer.read(32769), args.packet_sha256)
    old = PREVIOUS / "INSTALL_RESULT.json"
    require(old.resolve(strict=True) == old and old.stat().st_size == 1700
            and digest(old.read_bytes()) == PREVIOUS_RESULT_SHA,
            "PREVIOUS_SHARED_INSTALL")
    stage = ROOT / "incoming" / ("O01-pack16-2930-response-route-" + args.request)
    os.umask(0o077)
    stage.mkdir(mode=0o700)
    put(stage / "INTENT.json", encode({"request": args.request, "packet_SHA256": args.packet_sha256,
                                      "replay_allowed": False}))
    for name, data in files.items():
        put(stage / name, data)
        require(digest((stage / name).read_bytes()) == SOURCES[name], "STAGED_SOURCE_READBACK")
    spec = importlib.util.spec_from_file_location("audited_pack16_route", stage / "host.py")
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)
    entry = route.load_entry(PREVIOUS)
    host = entry.load_baseline()
    checked = route.frozen_target(host, PREVIOUS, RUNTIME, ORIGINAL_RESPONSE, stage / "proof.json",
                                  stage / "round9.json", stage / "publisher.py")
    result = {"state": "SHARED_TARGET_HISTORICAL_GATE_PASSED", "request": args.request,
              "packet_SHA256": args.packet_sha256, "sources": SOURCES,
              "historical_dependency_route": checked["historical_dependency_route"],
              "original_response": checked["archived_target_response"],
              "coordinator_proof_SHA256": checked["coordinator_proof_SHA256"],
              "runtime": checked["runtime"], "private_transport_wrapper_read_here": False,
              "actual_new_GPU_steps": 0, "actual_new_Slurm_steps": 0,
              "group6_adopted": False, "compute_node_checked": False}
    data = encode(result)
    put(stage / "RESULT.json", data)
    print(json.dumps({"result": result, "receipt": {"SHA256": digest(data), "bytes": len(data)}}))


if __name__ == "__main__":
    main()
