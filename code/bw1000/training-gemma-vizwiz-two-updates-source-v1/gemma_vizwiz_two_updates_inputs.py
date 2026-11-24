"""Bind native per-rank forwards to the same actual template/collator source rows."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


FIVE = "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"
SEVEN = "3f9051bb78a71f49d1e8ef51b06e706c164ff01a7650bc80a15154182d9e2c66"
EXPECTED_VIEW = "cebeecad13db010a06f2ae7ef48260c6b26cc5ad42ea4993f6b8a60849ce1ad4"
EXPECTED_SCOPE = "7b494f529c5d202b522fb35cd5cf89af67bb97775534968518aee109668bd9d0"
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
BOUND_MEDIA = (
    ("training-gemma-vizwiz-five-native-input-audit-v1",
     "389c4d3d0284b153f25800ae66e78401b26b4c209ce3571f5cba2999c5fdee85", 2743),
    ("training-gemma-vizwiz-reviewed-train-native-v1",
     "456eeff45b37fa1c1bc81573816ebbc1bea12dc60003a83418e6e8870ccf8a3d", 1097),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def input_signature(batch):
    import torch
    from vlm.runtime.first_update_audit import fingerprint

    names = ("input_ids", "labels", "pixel_values")
    require(all(name in batch and isinstance(batch[name], torch.Tensor) for name in names),
            "exact native text, labels, or media tensor unavailable")
    selected = {name: batch[name].detach().cpu().contiguous() for name in names}
    require(selected["input_ids"].dtype == selected["labels"].dtype == torch.int64
            and selected["input_ids"].shape == selected["labels"].shape
            and selected["input_ids"].ndim == 2 and selected["input_ids"].shape[0] == 1
            and selected["pixel_values"].is_floating_point(), "native single-row collator signature differs")
    selected["pixel_values"] = selected["pixel_values"].to(torch.bfloat16)
    return fingerprint(selected)["SHA256"]


def prove_mounts(stream, declaration):
    stage, binding_sha, binding_size = declaration
    root = RESULTS / stage / "data"
    binding_path = root / "data-binding-private.json"
    require(binding_path.is_file() and binding_path.resolve(strict=True) == binding_path, "native media binding alias")
    with binding_path.open("rb") as stream_file:
        raw = stream_file.read(binding_size + 1)
    require(len(raw) == binding_size and sha256(raw).hexdigest() == binding_sha, "original media binding bytes differ")
    media = json.loads(raw)["media"]
    observed = {}
    for line in stream.splitlines():
        row = json.loads(line)
        (name,) = row["images"]
        path = Path(name)
        require(path.parent == root / "images" and path.name in media
                and path.resolve(strict=True) == path and path.is_file(), "unaliased original image path required")
        pin = media[path.name]
        require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= 524288, "invalid original image extent")
        with path.open("rb") as image:
            value = image.read(pin["bytes"] + 1)
        require(len(value) == pin["bytes"] and sha256(value).hexdigest() == pin["SHA256"],
                "original mounted image bytes differ")
        observed[path.name] = pin["SHA256"]
    require(set(observed) == set(media), "original mounted media denominator differs")
    return observed


def derive_index(capped_training_template, five, reviewed, view_sha):
    """Run only with the already verified original Gemma train-mode capped template."""
    require(len(five) == 2164 and len(reviewed) == 3022
            and sha256(five).hexdigest() == FIVE and sha256(reviewed).hexdigest() == SEVEN,
            "frozen VizWiz native index source differs")
    require(view_sha == EXPECTED_VIEW, "published VizWiz view binding differs")
    mounted = [prove_mounts(stream, declaration) for stream, declaration in zip((five, reviewed), BOUND_MEDIA)]
    require(set(mounted[0]).isdisjoint(mounted[1])
            and len(set(mounted[0].values()) | set(mounted[1].values())) == 12, "mounted source media overlaps")
    rows = five.splitlines(keepends=True) + reviewed.splitlines(keepends=True)
    result = {}
    for line in rows:
        row = capped_training_template.encode(json.loads(line))
        batch = capped_training_template.data_collator([row])
        signature = input_signature(batch)
        require(signature not in result, "native VizWiz collator input identities collide")
        result[signature] = sha256(line).hexdigest()
    require(len(result) == 12, "native lookup denominator differs")
    return {"schema": "gemma-vizwiz-two-updates-native-collator-index-v1", "view_SHA256": view_sha,
            "source_streams": {"five": FIVE, "reviewed": SEVEN}, "index": result,
            "published_scope_SHA256": EXPECTED_SCOPE, "original_media_pins_checked": 12,
            "original_binding_SHA256": [declaration[1] for declaration in BOUND_MEDIA]}


def load_index(config):
    from vlm.runtime.resume_cursor import read_pinned_json

    lookup = read_pinned_json(Path(config["vizwiz_native_index"]), config["vizwiz_native_index_SHA256"],
                              maximum_bytes=16384)
    lineage = read_pinned_json(Path(config["vizwiz_lineage"]), config["vizwiz_lineage_SHA256"], maximum_bytes=100000)
    expected = {"five": FIVE, "reviewed": SEVEN}
    index = lookup["index"]
    require(lookup["schema"] == "gemma-vizwiz-two-updates-native-collator-index-v1"
            and lineage["schema"] == "gemma-vizwiz-frozen-twelve-nominal25-two-updates-v1"
            and lookup["view_SHA256"] == lineage["view_SHA256"] == config["dataset_SHA256"] == EXPECTED_VIEW
            and lookup["published_scope_SHA256"] == lineage["published_scope_SHA256"] == EXPECTED_SCOPE
            and lookup["original_media_pins_checked"] == 12
            and lookup["original_binding_SHA256"] == [declaration[1] for declaration in BOUND_MEDIA]
            and lookup["source_streams"] == lineage["source_streams"] == expected
            and lineage["view_rows"] == 800 and len(lineage["ordered_source_line_SHA256"]) == 800
            and len(index) == 12 and set(index.values()) == {
                row["source_line_SHA256"] for row in lineage["source_rows"]}
            and set(lineage["ordered_source_line_SHA256"]) == set(index.values()), "native lookup or frozen lineage differs")
    return index


def identify(forward_kwargs, index):
    signature = input_signature(forward_kwargs)
    require(signature in index, "native forward did not match a frozen VizWiz training source row")
    return index[signature]
