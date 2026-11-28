"""Exact .2766 ImageFolder membership, native update and checkpoint cursor validation."""

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
import stat

ARCHIVE = "c1a5ed37c352ccf03f5a4ee9b4a3073e6b70dc8a63cdde1b960322b28efab472"
RELATIVE = "results/sdxl-cc0-four-imagefolder-native-ae669d303b204e93bf5e419225be27bb/imagefolder"
NAMES = ("metal_tool_chest_original.png", "metal_tool_chest_side.png",
         "painted_wooden_stool_original.png", "painted_wooden_stool_side.png")
FILE_NAMES = {*NAMES, "metadata.jsonl"}
COUNT = len(NAMES)
IMAGE_ID = "sha256:9906c8cc26fefb917b5067a792f976b8e672118e4ee1f52f5826ec7fe72a567e"
ORIGINAL_REQUESTS = {"abd5c98bc5dc", "ae669d303b20", "c5eda9cfa6ad", "fd93b793bbbd", "e2f599910efc"}
HEX = re.compile(r"[0-9a-f]{64}")
HOLDOUTS = {"cassette_player", "vintage_electric_kettle", "vintage_telephone_wall_clock"}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
         "CANONICAL_FOUR_ROW_FILE")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(len(raw) <= limit, "BOUNDED_FOUR_ROW_FILE")
    return raw


def archive(path):
    raw = read(path, 16 << 10)
    need(sha256(raw).hexdigest() == ARCHIVE, "EXACT_2766_RECORDED_EVIDENCE")
    doc = json.loads(raw)
    target = doc["target"]
    need(target["train_relative_scope"] == RELATIVE and target["actual_native_imagefolder_rows"] == COUNT
         and target["zero_gres"] is True and target["optimizer_updates"] == 0
         and target["accounting"][:4] == ["1194900.2766", "sdxl-four-cc0-native-imagefolder-ae669d303b20",
                                          "COMPLETED", "0:0"]
         and set(doc["render"]["files"]) == FILE_NAMES and set(doc["render"]["holdout_excluded"]) == HOLDOUTS
         and tuple(row["filename"] for row in target["rows"]) == NAMES, "FOUR_ROW_SOURCE_PROVENANCE")
    return raw, doc


def verify(root, proof, *, decode=False):
    _, doc = archive(proof)
    need(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir()
         and {path.name for path in root.iterdir()} == FILE_NAMES, "EXACT_FIVE_IMAGEFOLDER_MEMBERS")
    assets = {}
    for name, pin in doc["render"]["files"].items():
        raw = read(root / name, 1 << 20)
        need(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "EXACT_2766_MEMBER_BYTES")
        assets[name] = raw
    metadata = [json.loads(line) for line in assets["metadata.jsonl"].splitlines()]
    need(len(metadata) == COUNT and tuple(row["file_name"] for row in metadata) == NAMES
         and all(set(row) == {"file_name", "text"} for row in metadata), "EXACT_FOUR_ROW_ORDER")
    records = []
    for index, (row, baseline) in enumerate(zip(metadata, doc["target"]["rows"], strict=True)):
        name, caption = row["file_name"], row["text"]
        need(type(caption) is str and bool(caption.strip())
             and sha256(caption.encode()).hexdigest() == baseline["caption_SHA256"]
             and sha256(assets[name]).hexdigest() == baseline["original_file_SHA256"], "EXACT_NATIVE_ROW_IDENTITY")
        if decode:
            from PIL import Image

            with Image.open(BytesIO(assets[name])) as image:
                need(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256)
                     and sha256(image.tobytes()).hexdigest() == baseline["decoded_RGB_SHA256"], "EXACT_NATIVE_RGB")
        records.append(dict(id=index, path=name, sha256=baseline["original_file_SHA256"],
                            caption_sha256=json_sha(caption)))
    return records


def bind_data(root, proof, *, decode=False):
    records = verify(root, proof, decode=decode)
    _, evidence = archive(proof)
    files = evidence["render"]["files"]
    inventory = [{"path": name, "bytes": files[name]["bytes"], "sha256": files[name]["SHA256"]}
                 for name in sorted(files)]
    return dict(records=records, ordered_sha256=json_sha(records), tree_sha256=json_sha(inventory))


def check_updates(raw, start, stop, records):
    need(len(raw) <= 2 << 20 and len(records) == COUNT, "BOUNDED_FOUR_ROW_UPDATE_LEDGER")
    rows = [json.loads(line) for line in raw.splitlines()]
    need([row["global_step"] for row in rows] == list(range(start, stop + 1)), "COMPLETE_ACTUAL_UPDATE_LEDGER")
    epochs = {}
    for row in rows:
        witness = row["sample_witness"]
        step = row["global_step"]
        epoch, batch = divmod(step - 1, COUNT)
        need(type(step) is int and math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
             and row["grad_norm"] > 0 and (witness["epoch"], witness["batch_index"]) == (epoch, batch)
             and len(witness["sample_ids"]) == 1 and type(witness["sample_ids"][0]) is int,
             "NATIVE_UPDATE_OR_CURSOR_NOT_PROVEN")
        sample = witness["sample_ids"][0]
        need(0 <= sample < COUNT and witness["samples"] == [records[sample]], "EXACT_NATIVE_SAMPLE_RECORD")
        tensors = witness["tensors"]
        need(set(tensors) == {"pixel_values", "input_ids_one", "input_ids_two"}
             and all(type(value) is str and HEX.fullmatch(value) for value in tensors.values()),
             "EXACT_UPDATE_TENSOR_HASHES")
        used = epochs.setdefault(epoch, set())
        need(sample not in used, "DUPLICATE_SAMPLE_IN_NATIVE_EPOCH")
        used.add(sample)
    for epoch, used in epochs.items():
        if start <= epoch * COUNT + 1 and (epoch + 1) * COUNT <= stop:
            need(used == set(range(COUNT)), "INCOMPLETE_FOUR_ROW_NATIVE_EPOCH")
    return rows


def cp20_projection(helper, path, doc):
    payload = helper.restricted_load(path / helper.STATE, numpy_rng=True)
    state = payload["state"]
    data = state["data"]
    helper.validate_cursor(data, COUNT)
    need(set(payload) == {"contract", "step", "state"} and type(payload["step"]) is int and payload["step"] == 20
         and payload["contract"] == doc["contract"] and helper.state_sha(state) == doc["state_sha256"]
         and helper.preflight(path) == doc,
         "SAME_NEW_CP20_MANIFEST_AND_STATE")
    order = data["order"]
    need((data["epoch"], data["consumed"]) == (4, 4) and sorted(order) == list(range(COUNT)),
         "EXACT_NEW_CP20_EPOCH4_CONSUMED4")
    return dict(step=20, epoch=data["epoch"], consumed=data["consumed"], order=list(order),
                state_sha256=doc["state_sha256"], checkpoint_manifest_SHA256=helper.file_sha(path / helper.MANIFEST))
