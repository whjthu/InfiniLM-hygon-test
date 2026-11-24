"""Read a bounded VizWiz-VQA training pilot from the publisher's pinned ZIPs."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile


PUBLISHER = "https://vizwiz.org/tasks-and-datasets/vqa/"
ROOT = "https://vizwiz.cs.colorado.edu/VizWiz_final/"
ANNOTATIONS = ("vqa_data/Annotations.zip", 1711273, '"1a1ca9-5f1efd408b3bc"')
ANNOTATION_ZIP_SHA256 = "dbc32fc1e3f1b1039aea906030d569d6c238b4ddb2eea5e3453a6a565c829ec9"
ANNOTATION_TRAIN_SHA256 = "f0725d0dfa790c48ee27fc3884a2a5d39dd79594da7b40a3a02e0612614ab351"
IMAGES = ("images/train.zip", 11298421598, '"2a1703b5e-59afdc1e58080"')
MAX_RANGE_BYTES = 6 * 1024 * 1024
MAX_ALL_RANGE_BYTES = 9 * 1024 * 1024
MAX_LABELS_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 384 * 1024
SAMPLES_PER_CLASS = 3
TRAIN_NAME = re.compile(r"VizWiz_train_(\d{8})\.jpg")


def pin(data):
    return {"bytes": len(data), "SHA256": sha256(data).hexdigest()}


class RangeSource:
    def __init__(self, entry):
        suffix, self.size, self.etag = entry
        self.url = ROOT + suffix
        self.position = 0
        self.network_bytes = 0
        self.requests = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        base = (0, self.position, self.size)[whence]
        position = base + offset
        if not 0 <= position <= self.size:
            raise OSError("invalid remote archive seek")
        self.position = position
        return position

    def read(self, count=-1):
        count = min(count, self.size - self.position) if count >= 0 else self.size - self.position
        if not count:
            return b""
        if count > MAX_RANGE_BYTES or self.network_bytes + count > MAX_ALL_RANGE_BYTES:
            raise OSError("public sample range budget exceeded")
        end = self.position + count - 1
        bounds = f"{self.position}-{end}"
        request = Request(self.url, headers={
            "Range": f"bytes={bounds}", "If-Range": self.etag, "Accept-Encoding": "identity",
        })
        self.requests += 1
        with urlopen(request, timeout=20) as response:
            if (response.url != self.url or response.status != 206
                    or response.headers.get("ETag") != self.etag
                    or response.headers.get("Content-Range") != f"bytes {bounds}/{self.size}"
                    or response.headers.get("Content-Length") != str(count)
                    or response.headers.get("Content-Encoding", "identity") != "identity"):
                raise OSError("public archive range or identity changed")
            payload = response.read(count + 1)
        self.network_bytes += len(payload)
        if len(payload) != count:
            raise EOFError("short public archive range")
        self.position = end + 1
        return payload


def labels(blob):
    with ZipFile(BytesIO(blob)) as archive:
        selected = [row for row in archive.infolist() if Path(row.filename).name == "train.json"
                    and not row.filename.startswith("__MACOSX/")]
        if len(selected) != 1 or selected[0].file_size > MAX_LABELS_BYTES:
            raise ValueError("unexpected training annotation member")
        raw = archive.read(selected[0])
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 20523:
        raise ValueError("unexpected publisher training denominator")
    accepted, excluded, seen = [], Counter(), set()
    for row in rows:
        image = row.get("image", "")
        match = TRAIN_NAME.fullmatch(image)
        if not match or image in seen:
            raise ValueError("unexpected or duplicate training image identity")
        seen.add(image)
        if 23431 <= int(match[1]) <= 23958:
            excluded["publisher_obfuscated_range"] += 1
            continue
        target = row.get("answerable")
        answers = row.get("answers")
        if type(target) is not int or target not in (0, 1) or not isinstance(answers, list) or len(answers) != 10:
            excluded["missing_answerability_or_answer_count"] += 1
            continue
        if (not isinstance(row.get("question"), str) or not row["question"].strip()
                or any(not isinstance(answer.get("answer"), str) or not answer["answer"].strip()
                       or answer.get("answer_confidence") not in ("yes", "maybe", "no") for answer in answers)):
            excluded["incomplete_annotation"] += 1
            continue
        accepted.append(row)
    return raw, accepted, dict(excluded)


def choose(archive, records):
    candidates = {}
    for member in archive.infolist():
        name = Path(member.filename).name
        if TRAIN_NAME.fullmatch(name):
            if name in candidates:
                raise ValueError("duplicate image in official archive")
            candidates[name] = member
    counts = Counter()
    sample = []
    for row in sorted(records, key=lambda item: sha256(item["image"].encode()).digest()):
        category = row["answerable"]
        if counts[category] == SAMPLES_PER_CLASS:
            continue
        member = candidates.get(row["image"])
        if member is None or not 0 < member.file_size <= MAX_IMAGE_BYTES or member.compress_size > MAX_IMAGE_BYTES:
            continue
        sample.append((row, member))
        counts[category] += 1
        if len(sample) == 2 * SAMPLES_PER_CLASS:
            return sample
    raise ValueError("balanced bounded image sample unavailable")


def save_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")


def collect(output):
    from PIL import Image

    output.mkdir()
    label_source = RangeSource(ANNOTATIONS)
    image_source = RangeSource(IMAGES)
    data = label_source.read()
    if pin(data)["SHA256"] != ANNOTATION_ZIP_SHA256:
        raise ValueError("official annotation ZIP digest changed")
    raw, records, excluded = labels(data)
    if pin(raw)["SHA256"] != ANNOTATION_TRAIN_SHA256:
        raise ValueError("official train member digest changed")
    save_json(output / "annotation-receipt.json", dict(
        published_url=label_source.url, etag=label_source.etag, zip=pin(data), train_json=pin(raw),
        planned_rows=20523, eligible_annotation_rows=len(records), excluded=excluded))
    images = output / "images"
    images.mkdir()
    assets = []
    private = []
    with ZipFile(image_source) as archive:
        for row, member in choose(archive, records):
            blob = archive.read(member)
            with Image.open(BytesIO(blob)) as image:
                if image.format != "JPEG" or any(size <= 0 or size > 6000 for size in image.size):
                    raise ValueError("unexpected JPEG sample format or dimensions")
                width, height = image.size
                image.load()
            with (images / row["image"]).open("xb") as handle:
                handle.write(blob)
            assets.append(dict(group=row["image"], answerable=row["answerable"], width=width,
                               height=height, image=pin(blob)))
            private.append(row)
    if len({item["image"]["SHA256"] for item in assets}) != len(assets):
        raise ValueError("pilot contains identical image bytes")
    save_json(output / "training-pilot-private.json", private)
    report = dict(publisher=PUBLISHER, published_dataset_license="CC BY 4.0", mode="TRAIN_ONLY_PILOT",
                  annotation_receipt=pin((output / "annotation-receipt.json").read_bytes()),
                  pilot_rows=len(assets), assets=assets, image_archive_bytes=IMAGES[1],
                  downloaded_image_archive_ranges_bytes=image_source.network_bytes,
                  downloaded_official_annotations_bytes=label_source.network_bytes,
                  requests=label_source.requests + image_source.requests,
                  evaluation_annotation_members_decoded=0, evaluation_images_read=0,
                  target_native_preprocessing_rows=0,
                  optimizer_updates=0, training_READY=False)
    save_json(output / "pilot-receipt.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "assets"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    collect(options.output)


if __name__ == "__main__":
    main()
