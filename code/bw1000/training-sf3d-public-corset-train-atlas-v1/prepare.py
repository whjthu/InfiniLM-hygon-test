"""Acquire the public Corset base/ORM atlases for the independent SF3D train split."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


HOME = Path(__file__).resolve().parent
REPO = HOME.parents[2]
OUTPUT = REPO / "local-only/training-sf3d-public-corset-train-atlas-v1-original"
REQUEST = "cd58e27654924c5899b970c6994ae186"
BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/"
ASSETS = (
    ("LICENSE.md", 65536),
    ("metadata.json", 65536),
    ("glTF/Corset.gltf", 65536),
    ("glTF/Corset_baseColor.png", 8 * 1024**2),
    ("glTF/Corset_occlusionRoughnessMetallic.png", 8 * 1024**2),
)
BUDGET = 120
PBR_SOURCE = "code/bw1000/training-sf3d-public-pbr-atlas2-v1"
PREVIOUS = "local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original/complete-data-binding-private.json"
PREVIOUS_SHA = "209bd0d1eec590589b39bc3481cecd4a08fb0a901afb5caf85932f09af5e5185"


def pin(raw):
    return {"bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}


def exact(path, expected):
    raw = path.read_bytes()
    if pin(raw)["SHA256"] != expected:
        raise ValueError("frozen dependency bytes differ")
    return raw


def write_json(path, doc):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def download(relative, maximum, deadline, row):
    time_left = deadline - time.time()
    if time_left < 2:
        raise TimeoutError("original download window ended")
    url = BASE + relative
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "SF3D-training-input-audit"})
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=min(25, time_left)) as response:
        lengths = response.headers.get_all("Content-Length")
        etags = response.headers.get_all("ETag")
        encodings = response.headers.get_all("Content-Encoding")
        row.update(status=response.status, declared_bytes=lengths, ETag=etags)
        if (response.status != 200 or response.url != url or not lengths or len(lengths) != 1
                or not lengths[0].isascii() or not lengths[0].isdecimal()
                or not 0 < int(lengths[0]) <= maximum or not etags or len(etags) != 1
                or encodings not in (None, ["identity"])):
            raise ValueError("unexpected public asset response identity or length")
        size = int(lengths[0])
        raw = response.read(size + 1)
        row["body_bytes_read"] = len(raw)
        if len(raw) != size or response.read(1) or time.time() >= deadline:
            raise ValueError("public asset body size or deadline differs")
    (OUTPUT / Path(relative).name).write_bytes(raw)
    row.update(pin(raw), stored=True)
    return raw


def worker(deadline):
    import numpy as np
    from PIL import Image

    previous = json.loads(exact(REPO / PREVIOUS, PREVIOUS_SHA))
    if (previous["blind_groups"] or set(previous["objects"]) != {"dev", "train"}
            or previous["objects"]["train"]["group"] != "WaterBottle"):
        raise ValueError("original train and development split differs")
    for item in previous["objects"].values():
        for key in ("input", "target"):
            source = REPO / item[key]
            if not source.resolve(strict=True).is_relative_to(REPO / "local-only"):
                raise ValueError("prior cohort source is outside its private namespace")
            exact(source, item[key + "_pin"]["SHA256"])

    package = REPO / PBR_SOURCE
    exact(package / "pbr.py", "3bc2279874fefbe2ebdb36ffa74824ee83661c63ea39359099dc266f7437d1e9")
    exact(package / "read_targets.py", "baabee208e0f7d5fb6502934423d2b4f1057f1a58f9c26103ba6a1e5f0f65fba")
    sys.path.insert(0, str(package))
    from pbr import atlas_targets, uniform_uv_sample
    from read_targets import load_targets

    rows = []
    report = dict(request=REQUEST, passed=False, planned_GETs=len(ASSETS), actual_GETs=rows,
                  actual_optimizer_updates=0, actual_GPU_steps=0, READY=False)
    try:
        raw = {}
        for index, (name, maximum) in enumerate(ASSETS):
            row = dict(asset=name, body_bytes_read=0, stored=False)
            rows.append(row)
            raw[name] = download(name, maximum, deadline, row)
            if index == 1:
                license_text = raw[ASSETS[0][0]].decode("utf-8")
                legal, = json.loads(raw[name])["legal"]
                if "CC0-1.0" not in license_text or "CC-BY-4.0" not in license_text or legal["spdx"] != "CC0-1.0":
                    raise ValueError("public asset or metadocumentation license identity differs")
            if index == 2:
                gltf = json.loads(raw[name])
                material, = gltf["materials"]
                pbr = material["pbrMetallicRoughness"]
                for key, (expected, _) in zip(("baseColorTexture", "metallicRoughnessTexture"), ASSETS[3:]):
                    image = gltf["textures"][pbr[key]["index"]]["source"]
                    if gltf["images"][image]["uri"] != Path(expected).name or pbr[key].get("texCoord", 0) != 0:
                        raise ValueError("official single-material texture reference or UV set differs")
        decoded = []
        for name, _ in ASSETS[3:]:
            with Image.open(io.BytesIO(raw[name])) as image:
                if image.format != "PNG" or image.mode not in {"RGB", "RGBA"} or not all(0 < n <= 4096 for n in image.size):
                    raise ValueError("expected actual bounded 8-bit RGB/A PNG")
                decoded.append(np.array(image.convert("RGBA")))
        digests = [pin(array.tobytes())["SHA256"] for array in decoded]
        old = [item["RGBA_pixels_SHA256"] for item in previous["four_texture_pixel_pins"].values()]
        if len(set(old + digests)) != 6:
            raise ValueError("public atlas exact-pixel collision")
        color, orm = decoded
        fields, extrema, texels = atlas_targets(color, orm, material)
        uv, slots = uniform_uv_sample(color, orm, material)
        output_image = OUTPUT / "train-input-atlas-RGBA.png"
        Image.fromarray(color).save(output_image, format="PNG")
        target = OUTPUT / "train-material-targets-private.npz"
        np.savez_compressed(target, UV=uv, **{key: matrix.astype(np.float32) for key, matrix in slots.items()},
                            **{"native_batch_" + key: matrix for key, matrix in fields.items()})
        targets = load_targets(target, pin(target.read_bytes()), "sf3d-public-pbr-UV-atlas-v1")
        binding = dict(schema="SF3D_PUBLIC_ATLAS_THREE_OBJECTS_V1", prior_binding_SHA256=PREVIOUS_SHA,
                       train=[previous["objects"]["train"], dict(group="Corset",
                           input=output_image.relative_to(REPO).as_posix(), input_pin=pin(output_image.read_bytes()),
                           target=target.relative_to(REPO).as_posix(), target_pin=pin(target.read_bytes()))],
                       development=[previous["objects"]["dev"]], blind=[], native_features_generated=False)
        write_json(OUTPUT / "three-object-binding-private.json", binding)
        report.update(passed=True, license="CC0-1.0", owner=legal.get("owner"), artist=legal.get("artist"),
                      material_label_range=extrema, texture_shape=list(color.shape), uniform_texels=texels,
                      actual_new_UV_labels=len(uv), actual_new_train_objects=1, complete_train_objects=2,
                      complete_dev_objects=1, blind_objects=0, exact_decoded_texture_collisions=0,
                      native_target_field_shapes={key: list(value.shape) for key, value in targets.items()},
                      new_target=pin(target.read_bytes()),
                      private_three_object_binding=pin((OUTPUT / "three-object-binding-private.json").read_bytes()))
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)[:200]
    write_json(OUTPUT / "result.json", report)
    return int(not report["passed"])


def main():
    if sys.argv[1:2] == ["--worker"]:
        deadline = float(sys.argv[2])
        gate_path = OUTPUT / "job-gate.json"
        while not gate_path.is_file() and time.time() < deadline:
            time.sleep(0.02)
        if not gate_path.is_file():
            raise TimeoutError("verified local job controls not established")
        gate = json.loads(gate_path.read_bytes())
        intent = json.loads((OUTPUT / "intent.json").read_bytes())
        if (gate["CPU_count"] != 2 or gate["memory_limit_bytes"] != 2 * 1024**3
                or intent["request"] != REQUEST or intent["deadline"] != deadline
                or intent["source"] != pin(Path(__file__).read_bytes()) or time.time() >= deadline):
            raise ValueError("original scope or job controls differ")
        write_json(OUTPUT / "worker-claimed.json", dict(request=REQUEST, deadline=deadline))
        return worker(deadline)

    import importlib.util
    guard = REPO / "code/bw1000/training-USGS-multievent-render144-v2/job_guard.py"
    exact(guard, "c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71")
    spec = importlib.util.spec_from_file_location("frozen_sf3d_corset_guard", guard)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    OUTPUT.mkdir(mode=0o700)
    wall, mono = time.time(), time.monotonic()
    write_json(OUTPUT / "intent.json", dict(request=REQUEST, original_epoch=wall, deadline=wall + BUDGET,
                                          planned_members=list(ASSETS), source=pin(Path(__file__).read_bytes()),
                                          replay_allowed=False))
    job = helper.guarded_run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                              "--worker", str(wall + BUDGET)],
                             OUTPUT / "job-gate.json", OUTPUT, wall + BUDGET, mono + BUDGET)
    result_path = OUTPUT / "result.json"
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else None
    outcome = dict(request=REQUEST, job=job, passed=job["returncode"] == 0 and bool(result and result["passed"]),
                   result=pin(result_path.read_bytes()) if result is not None else None, training_READY=False)
    write_json(OUTPUT / "outcome.json", outcome)
    print(json.dumps(outcome))
    return int(not outcome["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
