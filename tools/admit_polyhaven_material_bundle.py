"""Admit a bounded Poly Haven glTF bundle for geometry and material data preparation."""

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fetch_member(entry):
    url = entry["url"]
    parsed = urlsplit(url)
    size = entry["size"]
    require(parsed.scheme == "https" and parsed.netloc == "dl.polyhaven.org"
            and parsed.path.startswith("/file/ph-assets/Models/") and not parsed.query
            and type(size) is int and 0 < size <= 2_000_000, "unexpected public member URL or extent")
    with urlopen(Request(url, headers={"User-Agent": "InfiniLM-Domain-Training/1.0"}), timeout=12) as response:
        require(response.status == 200 and response.geturl() == url, "unexpected public member response")
        raw = response.read(size + 1)
    require(len(raw) == size and md5(raw, usedforsecurity=False).hexdigest() == entry["md5"],
            "publisher member extent or checksum mismatch")
    return raw


def admit(asset, metadata, output):
    require(re.fullmatch(r"[a-z0-9_]+", asset), "invalid asset name")
    metadata_bytes = metadata.read_bytes()
    require(0 < len(metadata_bytes) <= 2_000_000, "unexpected public metadata extent")
    review = json.loads(metadata_bytes)["gltf"]["1k"]["gltf"]
    root_name = asset + ".gltf"
    members = {root_name: review, **review["include"]}
    require(all(name == PurePosixPath(name).as_posix() and re.fullmatch(r"(?:textures/)?[a-zA-Z0-9_.-]+", name)
                and name.split("/")[-1] not in {".", ".."} for name in members), "unexpected glTF member path")
    require(2 <= len(members) <= 12 and sum(value["size"] for value in members.values()) <= 6_000_000,
            "unexpected bounded glTF closure size")

    raw_documents = {name: fetch_member(entry) for name, entry in members.items()}
    scene = json.loads(raw_documents[root_name])
    require(scene["asset"]["version"] == "2.0" and scene["meshes"] and scene["scenes"], "expected glTF 2.0 geometry")
    references = {entry["uri"] for entry in [*scene["buffers"], *scene["images"]]}
    require(references == set(members) - {root_name}, "referenced glTF files do not match exact bundle")
    require(all(item["byteLength"] == len(raw_documents[item["uri"]]) for item in scene["buffers"]),
            "glTF buffer lengths disagree with publisher members")
    textures = scene["textures"]
    images = scene["images"]

    def image(texture):
        return images[textures[texture["index"]]["source"]]["uri"]

    mappings = []
    for material in scene["materials"]:
        pbr = material.get("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr:
            mappings.append({
                "material": material.get("name"), "alpha_mode": material.get("alphaMode", "OPAQUE"),
                "albedo": image(pbr["baseColorTexture"]),
                "roughness_G_metallic_B": image(pbr["metallicRoughnessTexture"]),
                "base_color_factor": pbr.get("baseColorFactor", [1, 1, 1, 1]),
                "roughness_factor": pbr.get("roughnessFactor", 1),
                "metallic_factor": pbr.get("metallicFactor", 1),
            })
    require(mappings, "no material has both albedo and metallic-roughness texture bindings")
    output.mkdir(mode=0o700)
    for name, raw in raw_documents.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    report = {
        "asset": asset, "publisher": "Poly Haven", "license": "CC0-1.0",
        "upstream_metadata_SHA256": sha256(metadata_bytes).hexdigest(),
        "files": {name: {"bytes": len(raw), "SHA256": sha256(raw).hexdigest(), "publisher_MD5": members[name]["md5"]}
                  for name, raw in raw_documents.items()},
        "mesh_count": len(scene["meshes"]), "material_count": len(scene["materials"]),
        "material_texture_bindings": mappings, "geometry_rendered": False,
        "native_training_rows": 0, "train_dev_blind_assignment": None,
    }
    with (output / "admission.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.asset, args.metadata, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
