"""Freeze the next CLEVR val images for a two-arm OneVision checkpoint screen."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import zipfile

from PIL import Image
import build_clevr_grounded_caption as clevr


ORIGINAL_SOURCE = "62dbbf061ac30647afcdbc34aaee0dde270132de90113bfe2f568bd9e380cf4c"
CP25 = "b8b8a8b9dfe5012c4ce5f80499a1b3fc5e7b8b140a7a734cada07c4a39b83719"
VAL_PREFIX = f"{clevr.PREFIX}/images/val/"
SCENE_MEMBER = f"{clevr.PREFIX}/scenes/CLEVR_val_scenes.json"
OLD_EXPOSURES = 96
COUNT = 16


def digest(data):
    return sha256(data).hexdigest()


def frozen_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def pixels(raw):
    with Image.open(BytesIO(raw), formats=("PNG",)) as image:
        if image.size != (480, 320) or image.mode not in ("RGB", "RGBA"):
            raise ValueError("Official CLEVR image geometry differs")
        return digest(image.convert("RGB").tobytes())


def exposures(path):
    members = {}
    for group, expected in (("train", 64), ("dev", 16), ("calibration", 16)):
        raw = (path / f"{group}.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != expected:
            raise ValueError("Previously exposed cohort length differs")
        split = "train" if group == "train" else "val"
        for row in rows:
            relative, = row["images"]
            if re.fullmatch(rf"images/CLEVR_{split}_[0-9]{{6}}\.png", relative) is None:
                raise ValueError("Previously exposed cohort path differs")
            name = Path(relative).name
            if name in members:
                raise ValueError("Previously exposed cohort repeats an image")
            body = (path / relative).read_bytes()
            members[name] = (digest(body), pixels(body))
    if len(members) != OLD_EXPOSURES or len(set(members.values())) != OLD_EXPOSURES:
        raise ValueError("Previously exposed cohort identity differs")
    return members


def select(names, previously_exposed):
    eligible = [name for name in names if name.startswith(VAL_PREFIX)
                and re.fullmatch(r"CLEVR_val_[0-9]{6}\.png", name[len(VAL_PREFIX):])]
    if len(eligible) != 15000 or len(set(eligible)) != 15000:
        raise ValueError("Original validation membership differs")
    ordered = sorted(eligible, key=lambda name: digest(f"{clevr.SELECTION_RULE}/{name}".encode()))
    earlier = {Path(name).name for name in ordered[:32]}
    old_val = {name for name in previously_exposed if name.startswith("CLEVR_val_")}
    if earlier != old_val:
        raise ValueError("Prior 32 validation exposures differ from the original selection")
    return ordered[32:32 + COUNT]


def freeze(prior, output):
    if digest(Path(clevr.__file__).read_bytes()) != ORIGINAL_SOURCE:
        raise ValueError("Original verified CLEVR archive reader source differs")
    seen = exposures(prior)
    encoded = {value[0] for value in seen.values()}
    decoded = {value[1] for value in seen.values()}
    output.mkdir()
    target = output / "input"
    target.mkdir()
    images = target / "images"
    images.mkdir()
    with clevr.Archive() as source, zipfile.ZipFile(source) as archive:
        members = select(archive.namelist(), seen)
        protocol = {
            "arms": ["original base", "original base with .5000 CP25"],
            "candidate_adapter_sha256": CP25,
            "denominator_per_arm": COUNT,
            "decode": {"do_sample": False, "max_new_tokens": 128},
            "prompt": clevr.PROMPT,
            "metrics": ["exact_multiset", "false_color_shape_pairs", "missing_color_shape_pairs", "false_object_count",
                        "missing_object_count", "parse_failures", "generation_failures", "paired_win_tie_loss"],
            "failures": "A missing, invalid, capped or failed generation remains in the 16-slot denominator and cannot win a pair.",
            "interpretation": "independent synthetic-scene screen",
            "selection": "next 16 official validation entries at zero-based ranks 32 through 47 of the original hash order",
            "excluded_media_sha256": digest(frozen_json(seen)),
            "ordered_members": members,
        }
        plan_sha = exclusive(output / "PLAN.json", frozen_json(protocol))
        rows, image_members = [], []
        for slot, member in enumerate(members):
            info = archive.getinfo(member)
            if not 24 <= info.file_size <= 2 << 20:
                raise ValueError("Original validation PNG has unexpected extent")
            data = archive.read(info)
            raw_sha, pixel_sha = digest(data), pixels(data)
            if raw_sha in encoded or pixel_sha in decoded:
                raise ValueError("Independent validation roster contains duplicate media")
            encoded.add(raw_sha)
            decoded.add(pixel_sha)
            name = f"images/{slot:02d}.png"
            exclusive(target / name, data)
            image_members.append({"slot": slot, "name": name, "bytes": len(data), "SHA256": raw_sha})
            rows.append({"slot": slot, "images": [name], "messages": [{"role": "user", "content": clevr.PROMPT}]})
        requests_sha = exclusive(target / "requests.jsonl", b"".join(map(frozen_json, rows)))
        media_sha = exclusive(target / "IMAGES.json", frozen_json(image_members))

        scene_info = archive.getinfo(SCENE_MEMBER)
        if (scene_info.file_size, scene_info.CRC) != (33916162, 0x045958A1):
            raise ValueError("Original validation scenes have a different member identity")
        source.read_ahead = 2 * clevr.MIB
        scenes = clevr.stream_scenes(archive, SCENE_MEMBER, {Path(member).name for member in members})
        references = []
        for slot, member in enumerate(members):
            name = Path(member).name
            scene = scenes[name]
            if scene["split"] != "val" or scene["image_index"] != int(name[-10:-4]):
                raise ValueError("Scene belongs to another image")
            references.append({"slot": slot, "reference": clevr.caption(scene), "scene_sha256": digest(frozen_json(scene))})
        reference_sha = exclusive(output / "references.private.jsonl", b"".join(map(frozen_json, references)))
        report = {"schema": "LLAVA_CLEVR_VAL16_FROZEN_V1", "selection_frozen_before_scene_labels": True,
                  "prompt_media_contains_answers": False, "prior_exposed_images": OLD_EXPOSURES, "new_images": COUNT,
                  "planned_generations": 2 * COUNT, "encoded_overlap": 0, "decoded_pixel_overlap": 0,
                  "plan_SHA256": plan_sha, "answerless_requests_SHA256": requests_sha, "image_manifest_SHA256": media_sha,
                  "private_references_SHA256": reference_sha, "archive_range_bytes": source.transferred,
                  "producer_SHA256": digest(Path(__file__).read_bytes()), "completed_generations": 0,
                  "independent_real_image_quality": None}
    exclusive(output / "RECEIPT.json", frozen_json(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.prior, args.out), sort_keys=True))
