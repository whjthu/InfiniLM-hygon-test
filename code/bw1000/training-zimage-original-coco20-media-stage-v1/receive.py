"""Install privately copied COCO20 source files after checking every original digest."""

import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def write(path, raw):
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "Short immutable input write")
        stream.flush()
        os.fsync(stream.fileno())


def install(root, request, wire):
    claim = json.loads((root/"CLAIM.json").read_bytes())
    raw = (root/"COCO20_ORIGINAL.json").read_bytes()
    need(request == claim["request"] and root.name == claim["label"]
         and sha256(raw).hexdigest() == claim["input_contract_SHA256"], "Original COCO20 scope differs")
    pins = json.loads(raw)
    inputs = {**pins["images"], "metadata-zimage-20.csv":pins["metadata"]}
    need(len(inputs) == 21 and sum(x["bytes"] for x in inputs.values()) == 3646189, "Original COCO20 budget differs")
    bytes_received = wire.read(4200001)
    need(len(bytes_received) <= 4200000, "Archive exceeds the original data scope")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(bytes_received), mode="r:gz") as archive:
        for member in archive:
            name = member.name
            path = PurePosixPath(name)
            need(name in inputs and name not in files and member.isfile() and not path.is_absolute()
                 and ".." not in path.parts, "Foreign, duplicate or unsafe original member")
            spec = inputs[name]
            need(member.size == spec["bytes"], "Original input size differs")
            stream = archive.extractfile(member)
            contents = stream.read(spec["bytes"]+1)
            need(len(contents) == member.size and sha256(contents).hexdigest() == spec["SHA256"],
                 "Original input digest differs")
            files[name] = contents
    need(set(files) == set(inputs), "Original image roster or CSV missing")
    reader = csv.DictReader(io.StringIO(files["metadata-zimage-20.csv"].decode("utf-8"), newline=""))
    need(reader.fieldnames == ["image", "prompt"], "Original CSV schema differs")
    rows = list(reader)
    need(len(rows) == len(pins["row_order"]) == 20, "Original CSV roster differs")
    for row, name, digest in zip(rows,pins["row_order"],pins["row_prompt_SHA256"],strict=True):
        identifier = PurePosixPath(row["image"])
        relative = identifier.relative_to(PurePosixPath(pins["root"])).as_posix() if identifier.is_absolute() else row["image"]
        need(relative == name and sha256(row["prompt"].encode()).hexdigest() == digest, "Original CSV row differs")
    staging = root/"private-candidate"
    staging.mkdir(mode=0o700)
    (staging/"images").mkdir(mode=0o700)
    published = root/"inputs"
    need(not published.exists() and not published.is_symlink(), "Original destination already exists")
    for name, data in files.items():
        write(staging/name,data)
    staging.rename(published)
    directory = os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    summary = dict(request=request,verified_images=20,verified_image_bytes=3644668,
                   original_csv_SHA256=pins["metadata"]["SHA256"],prompt_hashes_matched=20,
                   original_manifest_present=False,actual_GPU_steps=0,training_ready=False)
    write(root/"receiver-result.json",(json.dumps(summary,sort_keys=True)+"\n").encode())
    return summary


if __name__=="__main__":
    print(json.dumps(install(Path(__file__).resolve().parent,sys.argv[1],sys.stdin.buffer)))
