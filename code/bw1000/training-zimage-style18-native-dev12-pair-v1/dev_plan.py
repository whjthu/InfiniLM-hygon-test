"""Freeze 12 Z-Image development pairs using the six original Style18 held-out rows."""

import csv
from hashlib import sha256
import io
import json
from pathlib import Path

BINDING = "2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35"
DEV_MANIFEST = "f19aa44cd8c0173e23c244b2b9d66a4585ca1a0440c240af6b931d0d8780a62c"
TRAIN_MANIFEST = "19e579a05ee06cee6bdfbb07abb18f14c40ecc999f9994465c94bdc98f42585d"
DEV_CSV = "a355200804403f5776ebb3344401ff43e267fbe038af7357307bfeead1a9ee3d"


def digest(value):
    return sha256(value).hexdigest()


def checked(root, relative, expected):
    path = root / relative
    raw = path.read_bytes()
    if path.is_symlink() or digest(raw) != expected:
        raise ValueError(f"Frozen development source differs: {relative}")
    return raw


def prepare(root, training_contract):
    binding = json.loads(checked(root, "data-binding-private.json", BINDING))
    train = json.loads(checked(root,"train/manifest.json",TRAIN_MANIFEST))["images"]
    dev = json.loads(checked(root,"dev/manifest.json",DEV_MANIFEST))["images"]
    if len(train) != 12 or len(dev) != 6:
        raise ValueError("Frozen split size differs")
    if training_contract["source_binding_SHA256"] != BINDING or [x["SHA256"] for x in train] != [
        x["image"]["SHA256"] for x in training_contract["rows"]
    ]:
        raise ValueError("Development split does not correspond to the actual training data")
    views = {k:[i for i in binding["images"] if i["split"] == k] for k in ("train","dev")}
    if [len(views[k]) for k in ("train","dev")] != [12,6]:
        raise ValueError("Media binding split differs")
    groups = {key:{x["group"] for x in rows} for key,rows in views.items()}
    pixels = {key:{x["RGB_pixels_SHA256"] for x in rows} for key,rows in views.items()}
    events = {key:{e for x in rows for e in x["event_ids"]} for key,rows in views.items()}
    if any(not values["train"].isdisjoint(values["dev"]) for values in (groups,pixels,events)):
        raise ValueError("Training data overlaps the development split")
    data = checked(root,"dev/metadata.csv",DEV_CSV)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"),newline="")))
    samples = []
    for index,(item,row) in enumerate(zip(dev,rows,strict=True)):
        if row["image"] != item["image"] or item["image"] != f"images/{index+12:03}.png":
            raise ValueError("Development CSV order differs")
        reference = checked(root,"dev/"+item["image"],item["SHA256"])
        if len(reference) != item["bytes"]:
            raise ValueError("Development reference extent differs")
        for replicate in range(2):
            key = digest(f"ZIMAGE_STYLE18_DEV_PAIR_V1/{BINDING}/{index}/{replicate}".encode())
            samples.append(dict(sample_id=key,ordinal=index,replicate=replicate,seed=int(key[:15],16),prompt=row["prompt"],
                                prompt_SHA256=digest(row["prompt"].encode()),
                                reference=dict(path=item["image"],bytes=item["bytes"],SHA256=item["SHA256"])))
    if len({r["seed"] for r in samples}) != 12:
        raise ValueError("Paired seeds repeat")
    return dict(scope="Style18 development only",binding_SHA256=BINDING,dev_images=6,train_images=12,width=256,
                height=256,denoising_steps=8,guidance_scale=1.0,paired_native_noise_required=True,
                blind_images_read=0,training_development_overlap=False,samples=samples)
