"""Screen the sealed Z-Image COCO holdout for reviewable perceptual matches."""

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import combinations, product
import json
from math import cos, pi
from pathlib import Path
import re
import time
from zipfile import ZipFile


MEDIA_SHA256 = "aa090e40fa22caef1dcf9fc3851af15c9734ac210e80dfb54c67b58d6c4cf7ac"
RECEIPT_SHA256 = "ebcc835c25a5c2022136bca91def57e52836019eb5ecc26a8788884571952f50"
TRAINING_SHA256 = "070013087a083bdec7defafaef92e781579e4ea491665536c48abe2336a21f55"
ARCHIVE_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ARCHIVE_BYTES = 815585330
COUNTS = {"training": 20, "development": 256, "blind": 4724}
DCT_SIZE = 32
DCT_COMPONENTS = 8
PHASH_MAX = 6
DHASH_MAX = 8
CANDIDATE_LIMIT = 10000
COSINES = tuple(tuple(cos(pi * u * (2 * i + 1) / (2 * DCT_SIZE)) for i in range(DCT_SIZE))
                for u in range(DCT_COMPONENTS))


def require(accepted, message):
    if not accepted:
        raise ValueError(message)


def pinned_json(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, f"wrong pinned file identity: {path.name}")
    return json.loads(raw)


def decode_hashes(raw, image, image_ops):
    with image.open(BytesIO(raw), formats=("JPEG",)) as opened:
        require(opened.width * opened.height <= 16 * 1024 * 1024, "image dimensions exceed audit limit")
        frame = image_ops.exif_transpose(opened).convert("L")
    diff = list(frame.resize((9, 8), image.Resampling.LANCZOS).getdata())
    dhash = sum((diff[y * 9 + x] > diff[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    small = list(frame.resize((DCT_SIZE, DCT_SIZE), image.Resampling.LANCZOS).getdata())
    lines = [small[y:y + DCT_SIZE] for y in range(0, len(small), DCT_SIZE)]
    horizontal = [[sum(pixel * c for pixel, c in zip(row, basis)) for row in lines] for basis in COSINES]
    coefficients = [sum(v * c for v, c in zip(row, basis))
                    for index, row in enumerate(horizontal) for j, basis in enumerate(COSINES) if index or j]
    middle = sorted(coefficients)[len(coefficients) // 2]
    phash = sum((value > middle) << i for i, value in enumerate(coefficients))
    return phash, dhash


def validated_roster(media, training):
    require(media["schema"] == "zimage-coco-val2017-media-v1" and media["archive_SHA256"] == ARCHIVE_SHA256
            and media["training_byte_overlaps"] == media["repeated_holdout_bytes"] == [],
            "media identity or exact-overlap audit differs")
    require(set(media["images"]) == {"development", "blind"}, "unexpected holdout set")
    rows = {"training": {}}
    for name, fields in training["images"].items():
        match = re.fullmatch(r"images/([0-9]{12})\.jpg", name)
        require(match is not None, "unexpected training archive path")
        image_id = int(match[1])
        require(image_id not in rows["training"], "duplicate training image id")
        rows["training"][image_id] = dict(fields, archive_member=f"val2017/{image_id:012d}.jpg")
    for cohort in ("development", "blind"):
        rows[cohort] = {int(k): v for k, v in media["images"][cohort].items()}
    require({k: len(v) for k, v in rows.items()} == COUNTS, "image cohort counts differ")
    require(len(set().union(*(set(v) for v in rows.values()))) == sum(COUNTS.values()), "image ID overlap")
    for cohort in rows.values():
        for image_id, fields in cohort.items():
            require(fields["archive_member"] == f"val2017/{image_id:012d}.jpg"
                    and 0 < fields["bytes"] <= 16 << 20 and re.fullmatch(r"[a-f0-9]{64}", fields["SHA256"]),
                    "invalid sealed image pin")
    return rows


def hash_archive(archive_file, roster, image, image_ops):
    expected = {row["archive_member"]: row for group in roster.values() for row in group.values()}
    signatures = {}
    with ZipFile(archive_file) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries if entry.filename != "val2017/" or not entry.is_dir()]
        require(len(names) == len(expected) and set(names) == set(expected), "ZIP contents differ")
        for name, pin in sorted(expected.items()):
            info = archive.getinfo(name)
            require(not info.flag_bits & 1 and info.file_size == pin["bytes"], "ZIP member size or flags changed")
            raw = archive.read(info)
            require(raw[:2] == b"\xff\xd8" and sha256(raw).hexdigest() == pin["SHA256"],
                    f"sealed image identity differs: {name}")
            signatures[name] = decode_hashes(raw, image, image_ops)
    return {cohort: {image_id: signatures[row["archive_member"]] for image_id, row in records.items()}
            for cohort, records in roster.items()}


def matches(groups):
    cohorts = (("training", "development"), ("training", "blind"),
               ("development", "development"), ("development", "blind"), ("blind", "blind"))
    results, candidates = {}, []
    for lhs, rhs in cohorts:
        a, b = sorted(groups[lhs].items()), sorted(groups[rhs].items())
        pairs = combinations(a, 2) if lhs == rhs else product(a, b)
        count = within = 0
        for (first, x), (second, y) in pairs:
            count += 1
            phash = (x[0] ^ y[0]).bit_count()
            if phash > PHASH_MAX:
                continue
            dhash = (x[1] ^ y[1]).bit_count()
            if dhash <= DHASH_MAX:
                within += 1
                if len(candidates) < CANDIDATE_LIMIT:
                    candidates.append({"first_cohort": lhs, "first_image_ID": first,
                                       "second_cohort": rhs, "second_image_ID": second,
                                       "phash_hamming": phash, "dhash_hamming": dhash})
        results[f"{lhs}:{rhs}"] = {"comparisons": count, "review_candidates": within}
    return results, candidates


def write_result(path, obj):
    raw = (json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest(), len(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--original-training-manifest", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    args = parser.parse_args()

    import PIL
    from PIL import Image, ImageOps, _imaging, features

    start = time.monotonic()
    receipt = pinned_json(args.archive_directory / "receipt.json", RECEIPT_SHA256)
    archive = args.archive_directory / "val2017.zip"
    require(receipt["archive_SHA256"] == ARCHIVE_SHA256 and receipt["archive_bytes"] == archive.stat().st_size == ARCHIVE_BYTES
            and receipt["private_media_SHA256"] == MEDIA_SHA256, "archive binding differs")
    media = pinned_json(args.archive_directory / "private-media.json", MEDIA_SHA256)
    training = pinned_json(args.original_training_manifest, TRAINING_SHA256)
    groups = hash_archive(archive, validated_roster(media, training), Image, ImageOps)
    counts, candidates = matches(groups)
    total = sum(item["review_candidates"] for item in counts.values())
    review_sha, review_bytes = write_result(args.result_directory / "candidates.private.json", candidates)
    summary = {
        "schema": "training.zimage.coco.perceptual_review_candidates.v1",
        "sealed_archive_SHA256": ARCHIVE_SHA256,
        "media_manifest_SHA256": MEDIA_SHA256,
        "training_manifest_SHA256": TRAINING_SHA256,
        "screen_code_SHA256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "decoder": {"Pillow": PIL.__version__, "JPEG": features.version_codec("jpg"),
                    "Pillow_Image_SHA256": sha256(Path(Image.__file__).read_bytes()).hexdigest(),
                    "Pillow_ImageOps_SHA256": sha256(Path(ImageOps.__file__).read_bytes()).hexdigest(),
                    "Pillow_native_SHA256": sha256(Path(_imaging.__file__).read_bytes()).hexdigest()},
        "verified_and_decoded_images": COUNTS,
        "policy": {"EXIF_transpose": True, "color": "L", "resample": "LANCZOS",
                   "phash": "32x32 DCT first 8x8 excluding DC, strictly above median",
                   "dhash": "9x8 horizontal differences",
                   "max_phash_hamming": PHASH_MAX, "max_dhash_hamming": DHASH_MAX,
                   "needs_both": True},
        "pair_counts": counts, "review_records": len(candidates),
        "review_records_truncated": len(candidates) != total,
        "private_review": {"filename": "candidates.private.json", "bytes": review_bytes, "SHA256": review_sha},
        "human_review_complete": False, "holdout_metrics_computed": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_result(args.result_directory / "perceptual-screen.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
