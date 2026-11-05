"""Classify every original GLM952 object for the private prefix access rebuild."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from derive_accepted_prefix_headers import HEADERS

MD_SHA = "9c073163fdc389fcc12fd9dbfefc9402c1f769e0614987dd80b27e84062ce301"
GPU_SHA = "1c2b1b276c1ee296b7cdf6d97e2564ee31cdac7fd82726f45f8208c2a9aaaf23"
SNAPSHOT_SHA = "55a3ba9626fb462f051a5b6eaa5ba8f2c5af9fa9b85ae183604af5f9f996e2bf"
RECEIPT_SHA = "241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32"
PRODUCER_SHA = "ca3a6350700c18303a1ad317cbc1ed12141a68cc914b619cfcb8f016e0d5d870"
NATIVE_SHA = "9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b"
OBJECT_ROOT = "/refactor/candidates/"
GPU_ROOT = "glm-grouped-native-model-r5117/helpers/"
CUDA_OR_STDLIB = {"<cuda_bf16.h>", "<cuda_runtime.h>", "<cmath>", "<cstddef>", "<cstdint>", "<atomic>"}


def checked(raw, digest):
    if sha256(raw).hexdigest() != digest:
        raise ValueError("unexpected GLM952 closure evidence bytes")
    return json.loads(raw)


def plan(md, gpu, receipt):
    if (md["receipt_sha256"] != RECEIPT_SHA or gpu["native_receipt_sha256"] != RECEIPT_SHA
            or gpu["compiler_script_sha256"] != PRODUCER_SHA or receipt["library_SHA256"] != NATIVE_SHA):
        raise ValueError("foreign GLM952 build receipts")
    raw = receipt["final_objects_SHA256"]
    if len(raw) != 29 or any(not name.startswith(OBJECT_ROOT) for name in raw):
        raise ValueError("unexpected original object census")
    objects = {name.removeprefix(OBJECT_ROOT): digest for name, digest in raw.items()}
    affected = md["objects_with_exact_class_headers"]
    rebuild = {row["object"]: row for row in affected}
    if len(rebuild) != 8 or len(affected) != 8:
        raise ValueError("unexpected changed-header object census")
    for name, row in rebuild.items():
        if (name not in objects or not row["headers"] or not row["source"]
                or any(header not in HEADERS or pins != [HEADERS[header]]
                       for header, pins in row["headers"].items())):
            raise ValueError("unbound class dependency")
    missing = {row["object"] for row in md["missing_exact_MD"]}
    for row in gpu["objects"]:
        name = GPU_ROOT + row["name"] + ".o"
        includes = row["includes"]
        if (name not in missing or not includes or len(includes) != row["include_directive_count"]
                or not set(includes) <= CUDA_OR_STDLIB
                or not all(row[key] is True for key in ("source_matches_original_stage",
                    "original_receipt_object_match", "same_current_and_original", "current_matches_GLM952",
                    "no_class_name_mentions"))
                or objects[name] != row["current_object_sha256"]
                or objects[name] != row["original_object_sha256"]):
            raise ValueError("legacy CUDA helper source or binary proof differs")
        missing.remove(name)
    if missing or len(gpu["objects"]) != 4 or md["linked_objects"] != len(objects):
        raise ValueError("unresolved GLM952 original object dependencies")
    return {
        "parent_Native_SHA256": NATIVE_SHA, "parent_object_count": len(objects),
        "recompile": [{**row, "original_object_SHA256": objects[name]} for name, row in rebuild.items()],
        "retain": {name: dict(sha256=digest, evidence=(
            "r3675 compiled CUDA object and direct include census" if name.startswith(GPU_ROOT)
            else "original dependency list has none of the four changed class headers"))
                   for name, digest in objects.items() if name not in rebuild},
        "build_authorized": False,
        "scope": "Changed-class rebuild partition only. Original dependency and object checks do not establish "
                 "candidate SDK compile, link, CUDA runtime identity or device parity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dependency_read", "legacy_GPU_origin", "receipt_snapshot"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    md = checked(args.dependency_read.read_bytes(), MD_SHA)
    gpu = checked(args.legacy_GPU_origin.read_bytes(), GPU_SHA)
    snapshot = checked(args.receipt_snapshot.read_bytes(), SNAPSHOT_SHA)["files"]
    original, = [row["rawtext"] for path, row in snapshot.items()
                 if path.endswith("/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json")]
    receipt = checked(original.encode("utf-8"), RECEIPT_SHA)
    result = plan(md, gpu, receipt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
