"""Freeze development and blind COCO groups for Z-Image without opening image files."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


ANNOTATIONS = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
TRAINING = "070013087a083bdec7defafaef92e781579e4ea491665536c48abe2336a21f55"
SOURCE_HOST = re.compile(r"(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com|farm[0-9]+\.static\.flickr\.com")
SOURCE_PATH = re.compile(r"/(?:[0-9]+/){1,3}([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg|png|gif)", re.I)
IMAGE_NAME = re.compile(r"images/([0-9]{12})\.jpg")
SPLIT_DOMAIN = b"T07.sft:COCO-val2017:photo-holdout:v1\0"
SEED_DOMAIN = b"T07.sft:COCO-val2017:paired-input:v1\0"
DEVELOPMENT_COUNT = 256


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read_pinned(path, expected, maximum):
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    need(0 < len(data) <= maximum and digest(data) == expected, "input identity changed")
    return json.loads(data)


def source_photo(url):
    link = urlsplit(url)
    match = SOURCE_PATH.fullmatch(link.path)
    need(link.scheme in ("http", "https") and not link.username and link.port is None
         and SOURCE_HOST.fullmatch(link.hostname or "") and match is not None, "unresolved Flickr source")
    return match[1]


def freeze(coco, training):
    train_files = training["row_order"]
    need(len(train_files) == len(set(train_files)) == 20 and set(train_files) == set(training["images"]),
         "original COCO20 membership changed")
    train_ids = set()
    for name in train_files:
        item = IMAGE_NAME.fullmatch(name)
        need(item is not None and re.fullmatch(r"[a-f0-9]{64}", training["images"][name]["SHA256"]),
             "original image identity cannot be bound")
        train_ids.add(int(item[1]))

    data_by_id = {}
    groups = defaultdict(list)
    licenses = {row["id"] for row in coco["licenses"]}
    for row in coco["images"]:
        image_id = row["id"]
        need(type(image_id) is int and image_id > 0 and image_id not in data_by_id
             and row["file_name"] == f"{image_id:012d}.jpg" and row["license"] in licenses,
             "unknown publisher image identity")
        photo = source_photo(row["flickr_url"])
        data_by_id[image_id] = (row, photo)
        groups[photo].append(image_id)
    need(len(data_by_id) == 5000 and train_ids <= data_by_id.keys(), "publisher is missing original training images")

    captions = defaultdict(list)
    annotation_ids = set()
    for row in coco["annotations"]:
        annotation_id, image_id, caption = row["id"], row["image_id"], row["caption"]
        need(type(annotation_id) is int and annotation_id > 0 and annotation_id not in annotation_ids
             and image_id in data_by_id and isinstance(caption, str) and caption.strip(), "invalid caption identity")
        annotation_ids.add(annotation_id)
        captions[image_id].append((annotation_id, caption))
    need(len(annotation_ids) == 25014 and captions.keys() == data_by_id.keys(), "unexpected published caption coverage")
    train_groups = {data_by_id[image_id][1] for image_id in train_ids}
    excluded = {image_id for source in train_groups for image_id in groups[source]}
    eligible = sorted((source for source in groups if source not in train_groups),
                      key=lambda source: (digest(SPLIT_DOMAIN + source.encode("ascii")), source))
    need(len(eligible) > DEVELOPMENT_COUNT + 2048, "blind roster lacks Inception-feature coverage")

    subsets = {"development": [], "blind": []}
    for index, source in enumerate(eligible):
        image_id = min(groups[source])
        row, _ = data_by_id[image_id]
        annotation_id, caption = min(captions[image_id])
        seed = int.from_bytes(sha256(SEED_DOMAIN + source.encode("ascii")).digest()[:8], "big") >> 1
        subset = "development" if index < DEVELOPMENT_COUNT else "blind"
        subsets[subset].append(dict(image_id=image_id, filename=row["file_name"],
                                    flickr_photo_id=source, license_id=row["license"],
                                    caption_id=annotation_id, prompt_SHA256=digest(caption.encode("utf-8")), seed=seed))
    selected = [row for rows in subsets.values() for row in rows]
    need(len({row["seed"] for row in selected}) == len(selected), "paired seed collision")
    duplicate_count = len(data_by_id) - len(excluded) - len(selected)
    return dict(schema="zimage-coco2017-holdout-roster-v1", case="T07.sft", purpose="evaluation only",
                publisher_annotations_SHA256=ANNOTATIONS, original_training_manifest_SHA256=TRAINING,
                publisher_images=len(data_by_id), publisher_captions=len(annotation_ids), original_training_ids=len(train_ids),
                excluded_training_source_groups=len(train_groups), excluded_images=len(excluded),
                duplicate_source_images_dropped=duplicate_count,
                subset_sizes={name: len(rows) for name, rows in subsets.items()},
                licenses_by_subset={name: dict(Counter(row["license_id"] for row in rows)) for name, rows in subsets.items()},
                split_order="SHA256 of the Flickr source ID in the pinned split domain",
                caption_policy="lowest publisher caption ID for the lowest image ID in each source group",
                frozen_membership_by_publisher_photo=True, image_bytes_verified=False,
                near_duplicates_checked=False, actual_shared_noise_pinned=False,
                scored_images=0, formal_effects_qualified=False, subsets=subsets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("original_training", type=Path)
    parser.add_argument("private_roster", type=Path)
    args = parser.parse_args()
    roster = freeze(read_pinned(args.annotations, ANNOTATIONS, 4 << 20),
                    read_pinned(args.original_training, TRAINING, 32 << 10))
    raw = (json.dumps(roster, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.private_roster.open("xb") as stream:
        stream.write(raw)
    del roster["subsets"]
    print(json.dumps(dict(roster, private_roster_SHA256=digest(raw), private_roster_bytes=len(raw)), sort_keys=True))


if __name__ == "__main__":
    main()
