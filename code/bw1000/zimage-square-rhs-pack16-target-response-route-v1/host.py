"""Use an explicitly authorized archived target response for Pack16 historical dependencies."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys


PROOF_SHA = "3726ac745f9d10a45e718e60edc25ca438d28acf9094b41535b4f0d3f5e25fcb"
RESPONSE_SHA = "d12faee2177a9e7e82076e333b1bba03ce55151381413b2d2d9767b035bf2edf"
RAW_WRAPPER_SHA = "f13cc42439079e0dde3b74d6fa2e371559a433ba42910cd04eb08bb613584262"
WRAPPER = "code/bw1000/zimage-square-rhs-pack16-ordinary-handoff-v1/pack16_host.py"
WRAPPER_SHA = "9c9460d1dc94915931cf24b8b20e95217373ff71747f59ac061c47a2bcdf8ad0"
ARBITER_SHA = "0b5e78f78556abb8a352ea7560dacf6634b723c3b3b70c1054ad9b16228075f4"
BASE_SHA = "316a5ea88ca53484e8e8f10dcf2df1e779b02b041c04445c234ec9e5d6010f59"
PUBLIC_PATH = "results/bw1000/2026-11-24/O01-pack16-json-v3-original2662-zero-GRES-dependencies-pass.json"
PUBLIC_SHA = "1bfb518c35e1dc58f277345d33cf48fe591a51097ef59fd27102d0afcfb5b353"
ARCHIVE_PATH = "results/bw1000/2026-11-25/training-astra-sol-round9-pack16-original-shared-host-install-and-readback.json"
ARCHIVE_SHA = "a8e65bf909b11fbcf0c858e7b02e25fef78e67560d106294c99a9f093bc19050"
ENCODER_PATH = "code/bw1000/zimage-square-rhs-pack16-live-preflight-v3/dispatch_host.py"
ENCODER_SHA = "d200fc8dabed0c3bed3074f3bca262e663a4afe47b35a254073cf1e72e8b9ca9"
INDEPENDENT_SHA = "07e246081415f6bf31f21007260ca833af55b500f312a4d562a5a3885792c627"
RUNTIME_SHA = "999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
REQUEST = "c0243062d1f24b70aee7e3092c24e8f0"
TARGET = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/O01-pack16-deps-json-c0243062d1f2/launch-and-accounting.json"
ACCOUNTING = ("1103606.2662", "O01-pack16-deps-json-c0243062d1f2", "COMPLETED", "0:0", "2", "f12r2n10",
              "cpu=1,mem=256M,node=1", "2026-11-24T19:31:54", "2026-11-24T19:31:56")
RESPONSE_FIELDS = {"request", "scope", "srun_rc", "accounting", "standby_current", "step_result",
                   "srun_stdout_sha256", "srun_stderr_sha256", "GPU_GO"}


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def exact(path, size, sha):
    require(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path.absolute(),
            "ALTERNATE_EVIDENCE_PATH")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    require(pin(raw) == {"bytes": size, "SHA256": sha}, "ALTERNATE_EVIDENCE_BYTES")
    return raw


def original_encoder(raw):
    tree = ast.parse(raw)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "encode"]
    require(len(funcs) == 1, "ORIGINAL_RESPONSE_ENCODER")
    scope = {"json": json}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "<sealed .2662 encoder>", "exec"), scope)
    return scope["encode"]


def evidence_fields(proof, public, target, archive):
    row = list(ACCOUNTING)
    wrapper = proof["coordinator_verified_local_original"]
    require(wrapper == {
        "SHA256": RAW_WRAPPER_SHA, "bytes": 4664, "authority_verified": True, "payload_flushed": True,
        "payload_send_attempted": True, "rc": 0, "remote_received": True,
    } and all(wrapper[field] is True for field in ("authority_verified", "payload_flushed",
                                                   "payload_send_attempted", "remote_received"))
            and type(wrapper["rc"]) is int, "EXACT_COORDINATOR_WIRE_FLAGS")
    require(proof["schema"] == "bw1000.pack16.original-2662.response-binding.v1"
            and proof["request"] == public["request"] == target["request"] == target["scope"] == REQUEST
            and proof["original_step"] == row[0] and proof["expected_target_path"] == TARGET
            and proof["original_accounting"] == public["accounting"] == row
            and target["accounting"] == [row] and proof["target_original_response_read"] is True
            and proof["permits_new_device_step"] is False and proof["expected_GPU_request"] == 0
            and proof["expected_runtime_manifest_SHA256"] == RUNTIME_SHA, "EXACT_ORIGINAL_2662_IDENTITY")
    require(set(target) == RESPONSE_FIELDS and type(target["srun_rc"]) is int and target["srun_rc"] == 0
            and target["GPU_GO"] is False and target["standby_current"] == [public["standby_accounting"]]
            and all(re.fullmatch("[0-9a-f]{64}", target[key]) for key in (
                "srun_stdout_sha256", "srun_stderr_sha256")), "EXACT_TARGET_RESPONSE_FIELDS")
    observed = target["step_result"]
    require(observed == public["precheck"] and observed["step"] == row[0]
            and observed["request"] == REQUEST and observed["node"] == row[5]
            and row[6] == "cpu=1,mem=256M,node=1"
            and observed["Slurm_GPU_resources"] == observed["GPU_context_initializations"] == observed["hipprof_invocations"] == 0
            and observed["precheck_passed"] is True and observed["target_GPU_GO"] is False
            and observed["same_standby_unchanged"] is True and observed["torch"]["package_imported"] is False
            and observed["runtime"] == {"manifest_SHA256": RUNTIME_SHA, "members": 5}, "EXACT_ZERO_GRES_STEP_RESULT")
    historical = archive["historical_target_response"]
    require(proof["public_dependency_record"] == {"path": PUBLIC_PATH, "bytes": 4342, "SHA256": PUBLIC_SHA}
            and public["raw_transport"]["sha256"] == RAW_WRAPPER_SHA
            and proof["independent_shared_host_source_archive"] == {
                "path": ARCHIVE_PATH, "bytes": 3390, "SHA256": ARCHIVE_SHA}
            and proof["round9_publication"] == "2a18d22b606c5b12df5248ddf3eb7abe77c32076"
            and proof["independent_target_readback_SHA256"] == archive["independent_target_readback"]["SHA256"] == INDEPENDENT_SHA
            and historical["target_response"] == "EXACT_ARCHIVED_TARGET_RESPONSE"
            and historical["target_response_pin"] == {"bytes": 3049, "SHA256": RESPONSE_SHA}
            and historical["historical_step"] == row[0] and historical["historical_GPU"] == 0
            and archive["actual_new_GPU_steps"] == archive["actual_new_Slurm_steps"] == 0,
            "INDEPENDENT_TARGET_RESPONSE_PROVENANCE")
    response = proof["response_identity"]
    require({key: response[key] for key in ("bytes", "SHA256")} == {"bytes": 3049, "SHA256": RESPONSE_SHA}
            and response["publisher_encoder"] == {"path": ENCODER_PATH, "bytes": 5191, "SHA256": ENCODER_SHA},
            "ORIGINAL_TARGET_RESPONSE_PIN")


def frozen_target(host, repo, runtime, response, proof, archive, encoder):
    proof_doc = json.loads(exact(proof, 2288, PROOF_SHA))
    public = json.loads(exact(repo / PUBLIC_PATH, 4342, PUBLIC_SHA))
    response_raw = exact(response, 3049, RESPONSE_SHA)
    response_doc = json.loads(response_raw)
    archive_doc = json.loads(exact(archive, 3390, ARCHIVE_SHA))
    encode = original_encoder(exact(encoder, 5191, ENCODER_SHA))
    require(encode(response_doc) == response_raw, "ORIGINAL_TARGET_RESPONSE_SERIALIZATION")
    evidence_fields(proof_doc, public, response_doc, archive_doc)
    frozen_pins = {item: host.pin(host.read(repo / item, 131072, want)) for item, want in host.FIXED.items()}
    source = json.loads(host.read(repo / host.SOURCE / "SOURCE_FILES.json", 16384,
                                  host.FIXED[host.SOURCE + "/SOURCE_FILES.json"]))
    for name, wanted in source["files"].items():
        data = host.read(repo / host.SOURCE / name, 65536, wanted["sha256"])
        require(len(data) == wanted["bytes"], "FROZEN_SOURCE_SIZE")
    checked_runtime = host.helper(repo, host.ORIGINAL + "/precheck.py").runtime(runtime)
    return {"frozen_pins": frozen_pins, "runtime": checked_runtime,
            "independent_reader": host.FIXED[host.ORIGINAL + "/read_result.py"],
            "historical_dependency_route": "EXACT_TARGET_RESPONSE_WITH_INDEPENDENT_COORDINATOR_PROOF",
            "archived_target_response": pin(response_raw), "coordinator_proof_SHA256": PROOF_SHA,
            "coordinator_verified_original_wrapper": proof_doc["coordinator_verified_local_original"],
            "private_transport_wrapper_read_here": False, "permits_new_device_step": False}


def load_entry(repo):
    path = repo / WRAPPER
    raw = exact(path, 2904, WRAPPER_SHA)
    exact(path.with_name("arbiter.py"), 13126, ARBITER_SHA)
    sys.path.insert(0, str(path.parent))
    previous = sys.modules.pop("arbiter", None)
    try:
        spec = importlib.util.spec_from_file_location("authorized_pack16_host_adapter", path)
        entry = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(path), "exec"), vars(entry))
    finally:
        sys.modules.pop("arbiter", None)
        if previous is not None:
            sys.modules["arbiter"] = previous
        sys.path.pop(0)
    return entry


def attach(entry, response, proof, archive, encoder):
    original_loader = entry.load_baseline

    def load():
        host = original_loader()
        require(pin(Path(host.__file__).read_bytes())["SHA256"] == BASE_SHA, "ORIGINAL_HOST_SOURCE")

        def select(repo, runtime, selected_response):
            require(selected_response == response, "HISTORICAL_DEPENDENCY_SELECTOR")
            return frozen_target(host, repo, runtime, response, proof, archive, encoder)

        host.frozen = select
        return host

    entry.load_baseline = load


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--target-response", type=Path)
    parser.add_argument("--historical-proof", type=Path)
    parser.add_argument("--historical-proof-sha256")
    parser.add_argument("--round9-archive", type=Path)
    parser.add_argument("--publisher-encoder", type=Path)
    special, original_arguments = parser.parse_known_args()
    repo_arg = argparse.ArgumentParser(add_help=False)
    repo_arg.add_argument("--repo", type=Path, required=True)
    original, _ = repo_arg.parse_known_args(original_arguments)
    entry = load_entry(original.repo.resolve(strict=True))
    options = (special.target_response, special.historical_proof, special.historical_proof_sha256,
               special.round9_archive, special.publisher_encoder)
    if any(item is not None for item in options):
        require(all(item is not None for item in options) and special.historical_proof_sha256 == PROOF_SHA
                and special.target_response.as_posix() == TARGET
                and not any(arg == "--original-dependencies" or arg.startswith("--original-dependencies=")
                            for arg in original_arguments), "EXPLICIT_HISTORICAL_ROUTE")
        attach(entry, special.target_response, special.historical_proof, special.round9_archive, special.publisher_encoder)
        # The frozen parser has one historical input slot. Only this explicit route changes its verifier.
        original_arguments.extend(["--original-dependencies", str(special.target_response)])
    sys.argv[1:] = original_arguments
    return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
