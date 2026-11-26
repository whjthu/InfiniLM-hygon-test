"""Project source groups from the preserved, bounded official ABO sample."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import unicodedata
import zlib


SOURCE_PINS = {
    "download-ledger.jsonl": (16617, "79c5e9bd8e923d0cc3dea76a3cbf0098d41734703ab704ecfce40934273a5340"),
    "public-license.body": (14527, "419896aea50c15d6e40c5b4baf4bd346f78223b9a354c7357ad00262afdc08ec"),
    "listing-shard-0-prefix.body": (196608, "a606aa2af9332e7f69f411401e8ae6adc7efb9a635f516dd53d6ec9e7b70c1f5"),
    "image-index-prefix.body": (131072, "aaa872b5605c03b1056bad87d2c881a9a77c3104c7e4d752b46c026f348a9214"),
    "image-index-next-range.body": (32768, "640e98b6ef21ab4b48dd9693c4a06bf9cbb8b972529ea99147f5c2bcd6223469"),
    "mesh-index.body": (340528, "90d061c3e3efa555a55bfc8ed7fc55cdb322086397443de1b33e91f7924d7d92"),
}


def verified_files(root):
    result = {}
    for name, (size, digest) in SOURCE_PINS.items():
        with (root / name).open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            raise ValueError(f"Publisher response differs: {name}")
        result[name] = raw
    return result


def decompress(raw, *, complete):
    decoder = zlib.decompressobj(31)
    plain = decoder.decompress(raw, 8 * 1024 * 1024)
    if decoder.unconsumed_tail or decoder.unused_data or decoder.eof != complete:
        raise ValueError("Publisher archive prefix or EOF differs")
    if not complete:
        plain, separator, _partial_line = plain.rpartition(b"\n")
        if not separator:
            raise ValueError("Publisher prefix contains no complete line")
    return plain


def unique_index(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Missing or repeated publisher key: {key}")
        result[value] = row
    return result


def normalized_values(records):
    return {
        " ".join(unicodedata.normalize("NFKC", entry["value"]).casefold().split())
        for entry in records if entry.get("value")
    } - {""}


def image_ids(listing):
    return {listing.get("main_image_id"), *listing.get("other_image_id", [])} - {None, ""}


def listing_keys(listing, images):
    ids = image_ids(listing)
    keys = {("image_id", image_id) for image_id in ids}
    keys.update(("image_path", images[image_id]["path"]) for image_id in ids if image_id in images)
    if mesh_id := listing.get("3dmodel_id"):
        keys.add(("mesh_id", mesh_id))
    brands = normalized_values(listing.get("brand", []))
    models = normalized_values(listing.get("model_number", [])) - {"n/a", "na", "none", "unknown"}
    keys.update(("brand_model", brand, model) for brand in brands for model in models if brand not in {"generic", "unbranded"})
    return keys


def source_groups(listings, images):
    parents = {key: key for key in listings}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    by_key = {}
    for item_id in sorted(listings):
        for key in sorted(listing_keys(listings[item_id], images)):
            left, right = sorted((root(item_id), root(by_key.setdefault(key, item_id))))
            parents[right] = left
    groups = {}
    for item_id in sorted(listings):
        groups.setdefault(root(item_id), []).append(item_id)
    return list(groups.values())


def project(raw):
    rows = decompress(raw["listing-shard-0-prefix.body"], complete=False).splitlines()
    listings = unique_index(map(json.loads, rows), "item_id")
    image_table = decompress(raw["image-index-prefix.body"] + raw["image-index-next-range.body"], complete=False)
    images = unique_index(csv.DictReader(image_table.decode().splitlines()), "image_id")
    mesh_table = decompress(raw["mesh-index.body"], complete=True)
    meshes = unique_index(csv.DictReader(mesh_table.decode().splitlines()), "3dmodel_id")
    groups = source_groups(listings, images)
    entries = []
    for members in groups:
        candidate_rows = []
        for item_id in members:
            listing = listings[item_id]
            main_image = listing.get("main_image_id")
            indexed = sorted(image_ids(listing) & images.keys())
            mesh = listing.get("3dmodel_id")
            if indexed or mesh in meshes:
                candidate_rows.append({
                    "item_id": item_id,
                    "indexed_main_image_id": main_image if main_image in images else None,
                    "indexed_image_ids": indexed,
                    "indexed_mesh_id": mesh if mesh in meshes else None,
                })
        entries.append({"items": members, "candidate_rows": candidate_rows})
    candidates = [row for group in entries for row in group["candidate_rows"]]
    return {
        "state": "BOUNDED_OFFICIAL_ABO_SOURCE_GROUPS_PROJECTED",
        "source_SHA256": {name: digest for name, (_size, digest) in SOURCE_PINS.items()},
        "source_license": "CC BY 4.0",
        "source_attribution": "Amazon.com and ABO dataset authors",
        "listing_prefix_rows": len(listings),
        "indexed_image_prefix_rows": len(images),
        "complete_mesh_index_rows": len(meshes),
        "exact_metadata_groups": len(entries),
        "multi_listing_groups": sum(len(group["items"]) > 1 for group in entries),
        "largest_group_rows": max(map(len, groups)),
        "listings_with_indexed_images": sum(bool(row["indexed_image_ids"]) for row in candidates),
        "listings_with_indexed_mesh": sum(row["indexed_mesh_id"] is not None for row in candidates),
        "listings_with_indexed_main_image_and_mesh": sum(
            row["indexed_main_image_id"] is not None and row["indexed_mesh_id"] is not None for row in candidates
        ),
        "pilot_directions": [21, 26, 30, 38, 42, 45],
        "train_dev_blind_split_frozen": False,
        "media_downloads": 0,
        "training_rows": 0,
        "ready_runs": 0,
        "groups": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("private_output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not args.private_output.resolve().is_relative_to(repo / "local-only"):
        parser.error("Detailed source groups belong under the repository's private local-only directory")
    summary = project(verified_files(args.source))
    raw = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.private_output.open("xb") as stream:
        stream.write(raw)
    summary.pop("groups")
    print(json.dumps(dict(summary, private_index_SHA256=sha256(raw).hexdigest()), sort_keys=True))


if __name__ == "__main__":
    main()
