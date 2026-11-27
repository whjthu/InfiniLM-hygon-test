"""Stage a bounded, reproducible caption-training subset of official CLEVR v1.0."""

import argparse
import collections
import hashlib
import http.client
import io
import json
import re
import zipfile
from pathlib import Path


HOST = "dl.fbaipublicfiles.com"
ARCHIVE_PATH = "/clevr/CLEVR_v1.0.zip"
ARCHIVE_SIZE = 19_021_600_724
ARCHIVE_ETAG = '"3426f0a360618fcf2179966a8d1428c3-1210"'
PREFIX = "CLEVR_v1.0"
MIB = 1024 * 1024
SELECTION_RULE = "clevr-grounded-caption-v1"
PROMPT = (
    "<image>List the visible objects by color and shape with their counts. "
    "Only include objects that are present."
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Archive(io.RawIOBase):
    def __init__(self):
        self.connection = http.client.HTTPSConnection(HOST, timeout=25)
        self.position = 0
        self.transferred = 0
        self.buffer_start = 0
        self.buffer = b""
        self.read_ahead = 32 * 1024

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = {0: 0, 1: self.position, 2: ARCHIVE_SIZE}[whence] + offset
        if self.position < 0:
            raise OSError("negative range position")
        return self.position

    def read(self, size=-1):
        available = max(0, ARCHIVE_SIZE - self.position)
        size = available if size < 0 else min(size, available)
        if size == 0:
            return b""
        relative = self.position - self.buffer_start
        if relative < 0 or relative + size > len(self.buffer):
            start = self.position
            length = min(max(size, self.read_ahead), available)
            if self.transferred + length > 160 * MIB:
                raise ValueError("official archive range budget exceeded")
            end = start + length - 1
            self.connection.request("GET", ARCHIVE_PATH, headers={
                "Range": f"bytes={start}-{end}", "If-Match": ARCHIVE_ETAG,
                "Accept-Encoding": "identity",
            })
            response = self.connection.getresponse()
            if (response.status != 206 or response.getheader("ETag") != ARCHIVE_ETAG
                    or response.getheader("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}"):
                response.close()
                raise OSError(f"archive range or identity mismatch, HTTP {response.status}")
            self.buffer = response.read(length + 1)
            if len(self.buffer) != length:
                raise OSError("incomplete archive range")
            self.buffer_start, relative = start, 0
            self.transferred += length
        self.position += size
        return self.buffer[relative:relative + size]

    def close(self):
        self.connection.close()
        super().close()


def stream_scenes(zipped, name, selected):
    """Use one JSON decoder per scene, limiting retained data even on the training member."""
    decoder = json.JSONDecoder()
    size = 128 * 1024
    found = {}
    with io.TextIOWrapper(zipped.open(name), encoding="utf-8") as stream:
        head = stream.read(size)
        match = re.search(r'"scenes"\s*:\s*\[', head)
        if not match:
            raise ValueError("missing scene array")
        pending = head[match.end():]
        while True:
            if len(pending) < size:
                pending += stream.read(size)
            pending = pending.lstrip()
            if pending.startswith("]"):
                break
            scene, end = decoder.raw_decode(pending)
            image_name = scene["image_filename"]
            if image_name in selected:
                if image_name in found:
                    raise ValueError("duplicate official scene identity")
                found[image_name] = scene
            pending = pending[end:].lstrip()
            if not pending:
                pending = stream.read(size).lstrip()
            if pending.startswith(","):
                pending = pending[1:]
            elif not pending.startswith("]"):
                raise ValueError("invalid scene delimiter")
        # Force ZIP CRC validation even if metadata follows the scene array.
        while stream.read(size):
            pass
    if found.keys() != selected:
        raise ValueError("official scenes missing selected image IDs")
    return found


def caption(scene):
    counts = collections.Counter((obj["color"], obj["shape"]) for obj in scene["objects"])
    if not 3 <= sum(counts.values()) <= 10:
        raise ValueError("unexpected CLEVR object count")
    return "; ".join(f"{count} {color} {shape}{'s' if count > 1 else ''}"
                     for (color, shape), count in sorted(counts.items()))


def jsonl(path, rows):
    path.write_bytes(b"".join(
        (json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    ))


def stage(output):
    with Archive() as source, zipfile.ZipFile(source) as zipped:
        scene_members, ordered_images = {}, {}
        archive_names = zipped.namelist()
        for split, count in (("train", 64), ("val", 32)):
            stem = f"{PREFIX}/images/{split}/"
            eligible = [name for name in archive_names if name.startswith(stem)
                        and re.fullmatch(rf"CLEVR_{split}_[0-9]{{6}}\.png", name[len(stem):])]
            expected = 70_000 if split == "train" else 15_000
            if len(eligible) != expected or len(set(eligible)) != expected:
                raise ValueError("official image membership changed")
            ordered_images[split] = sorted(
                eligible, key=lambda name: sha(f"{SELECTION_RULE}/{name}".encode())
            )[:count]
            member = f"{PREFIX}/scenes/CLEVR_{split}_scenes.json"
            info = zipped.getinfo(member)
            scene_members[split] = {"path": member, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}

        groups = {"train": [], "dev": [], "calibration": []}
        roster, image_hashes = [], set()
        (output / "images").mkdir(parents=True)
        for split, names in ordered_images.items():
            source.read_ahead = 2 * MIB
            scenes = stream_scenes(zipped, scene_members[split]["path"], {Path(n).name for n in names})
            source.read_ahead = 32 * 1024
            for index, name in enumerate(names):
                group = "train" if split == "train" else ("dev" if index < 16 else "calibration")
                image_name = Path(name).name
                relative_path = f"images/{image_name}"
                info = zipped.getinfo(name)
                if not 24 <= info.file_size <= 2 * MIB:
                    raise ValueError("invalid official PNG member size")
                pixels = zipped.read(info)
                if pixels[:8] != b"\x89PNG\r\n\x1a\n" or pixels[16:24] != bytes.fromhex("000001e000000140"):
                    raise ValueError("official PNG signature or dimensions changed")
                digest = sha(pixels)
                if digest in image_hashes:
                    raise ValueError("duplicate image bytes across candidate groups")
                image_hashes.add(digest)
                (output / relative_path).write_bytes(pixels)
                scene = scenes[image_name]
                if scene["split"] != split or int(image_name[-10:-4]) != scene["image_index"]:
                    raise ValueError("image filename and scene identity disagree")
                answer = caption(scene)
                row = {"messages": [{"role": "user", "content": PROMPT}], "images": [relative_path]}
                if group == "train":
                    row["messages"].append({"role": "assistant", "content": answer})
                groups[group].append(row)
                roster.append({"group": group, "archive_member": name, "image_sha256": digest,
                               "bytes": len(pixels), "scene_sha256": sha(json.dumps(scene, sort_keys=True).encode()),
                               "reference": answer})
        for group, rows in groups.items():
            jsonl(output / f"{group}.jsonl", rows)
        jsonl(output / "references.private.jsonl", roster)
        return {
            "source": f"https://{HOST}{ARCHIVE_PATH}", "archive_size": ARCHIVE_SIZE,
            "archive_etag": ARCHIVE_ETAG, "http_range_bytes": source.transferred,
            "license": "CC BY 4.0", "attribution": "Justin Johnson et al., CLEVR v1.0 (2017)",
            "license_source": "https://cs.stanford.edu/people/jcjohns/clevr/",
            "selection_rule": SELECTION_RULE, "scene_members": scene_members,
            "counts": {group: len(rows) for group, rows in groups.items()},
            "sha256": {name: sha((output / name).read_bytes()) for name in (
                "train.jsonl", "dev.jsonl", "calibration.jsonl", "references.private.jsonl"
            )},
            "role": "synthetic-scene training candidate and public-data calibration",
            "real_image_blind_quality": None, "target_native_preflight": None, "optimizer_updates": 0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local-only directory")
    args = parser.parse_args()
    args.output.mkdir()
    receipt = stage(args.output)
    (args.output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
