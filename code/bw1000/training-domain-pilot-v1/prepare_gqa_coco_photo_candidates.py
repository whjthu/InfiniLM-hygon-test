"""Join original training questions to unambiguous Visual Genome and COCO photo identities."""
import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from zipfile import ZipFile

from build_gqa_relation_train import deflate_parts, is_balanced_relation, load_pinned, pinned, question_rows


def flickr_photo_id(url):
    parsed = urlparse(url)
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or not parsed.hostname.endswith((".staticflickr.com", ".static.flickr.com"))):
        return None
    image_file = parsed.path.rsplit("/", 1)[-1]
    match = re.fullmatch(r"([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg)", image_file, re.IGNORECASE)
    return int(match[1]) if match else None


def join_metadata(visual, coco, relation_counts):
    by_vg = {str(row["image_id"]): row for row in visual}
    if len(by_vg) != len(visual):
        raise ValueError("duplicate Visual Genome image identity")
    coco_occurrences = Counter(row["coco_id"] for row in visual if row.get("coco_id") is not None)
    flickr_occurrences = Counter(row["flickr_id"] for row in visual if row.get("flickr_id") is not None)
    licenses = coco["licenses"]
    images = coco["images"]
    permitted = licenses["4"]
    if permitted["name"] != "Attribution License" or urlparse(permitted["url"]).path != "/licenses/by/2.0/":
        raise ValueError("COCO did not resolve license 4 to CC BY 2.0")
    rejected = Counter()
    candidates = []
    for image_id, count in relation_counts.items():
        visual_row = by_vg.get(image_id)
        if visual_row is None:
            rejected["missing_visual_genome"] += 1
            continue
        coco_id = visual_row.get("coco_id")
        flickr_id = visual_row.get("flickr_id")
        coco_row = images.get(str(coco_id))
        if coco_row is None:
            rejected["missing_coco_train2017"] += 1
            continue
        if coco_row["license"] != 4:
            rejected["not_coco_attribution_license"] += 1
            continue
        if (coco_occurrences[coco_id] != 1 or flickr_occurrences[flickr_id] != 1
                or flickr_photo_id(coco_row["flickr_url"]) != flickr_id):
            rejected["ambiguous_origin"] += 1
            continue
        candidates.append({"gqa_image_id":image_id, "coco_id":coco_id, "flickr_id":flickr_id,
                           "source_group":f"flickr:{flickr_id}", "source_url":f"https://www.flickr.com/photo.gne?id={flickr_id}",
                           "coco_flickr_url":coco_row["flickr_url"], "coco_license_url":permitted["url"],
                           "visual_genome_dimensions":[visual_row["width"],visual_row["height"]],
                           "coco_dimensions":[coco_row["width"],coco_row["height"]],
                           "relation_questions":count})
    candidates.sort(key=lambda row: row["flickr_id"])
    return candidates, dict(rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions","question-receipt","visual-archive","coco-projection"):
        parser.add_argument(f"--{name}",required=True,type=Path)
        parser.add_argument(f"--{name}-sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = load_pinned(args.question_receipt,args.question_receipt_sha)
    if args.questions_sha != receipt["compressed_SHA256"] or receipt["member"] != "train_balanced_questions.json":
        raise ValueError("official balanced question member differs")
    visual_zip = pinned(args.visual_archive,args.visual_archive_sha)
    with ZipFile(io.BytesIO(visual_zip)) as archive:
        member = archive.getinfo("image_data.json")
        if not 0 < member.file_size <= 25_000_000:
            raise ValueError("unexpected Visual Genome member extent")
        visual = json.loads(archive.read(member))
    coco = load_pinned(args.coco_projection,args.coco_projection_sha)
    counts = Counter(row["imageId"] for _,row in question_rows(deflate_parts(args.questions,
                     receipt["uncompressed_SHA256"],args.questions_sha,receipt["uncompressed_bytes"]))
                     if is_balanced_relation(row))
    candidates,rejected = join_metadata(visual,coco,counts)
    summary = {"relation_image_ids":len(counts),"relation_questions":sum(counts.values()),
               "unambiguous_coco_attribution_metadata_candidates":len(candidates),"excluded":rejected,
               "gqa_media_fetched":0,"training_rows":0,"holdout_bound":False,
               "visual_archive_SHA256":args.visual_archive_sha,"coco_projection_SHA256":args.coco_projection_sha,
               "question_member_SHA256":receipt["uncompressed_SHA256"]}
    if sum(rejected.values()) + len(candidates) != len(counts):
        raise ValueError("candidate denominator differs")
    raw = (json.dumps({"summary":summary,"candidates":candidates},sort_keys=True,separators=(",",":"))+"\n").encode()
    with args.output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({**summary,"private_candidates_SHA256":sha256(raw).hexdigest()},sort_keys=True))


if __name__ == "__main__":
    main()
