"""Pack, receive, or recheck the frozen SDXL style18 train-only view."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import zipfile

from preflight_sdxl_style18_target import load_portable


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
TARGET = ROOT / "results/training-sdxl-USGS-style18-domain25-data-v1"
REQUEST = "274f1a27b7ea44f198eaa08589b845b49"
MAX_PACKET = 320 * 1024
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RECEIPT = "stage-receipt.json"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def pins(portable):
    raw = Path(portable.__file__).with_name("DATA_BINDING.json").read_bytes()
    if sha256(raw) != portable.BINDING_SHA256:
        raise ValueError("train input binding differs")
    doc = json.loads(raw)
    return doc["train_images_by_filename_sha256"], doc["train_metadata"]


def checked_members(portable, names):
    images, metadata = pins(portable)
    names = list(names)
    if len(names) != len(images) + 1 or len(set(names)) != len(names):
        raise ValueError("unexpected train archive member count")
    members = {}
    for name in names:
        if not SAFE_NAME.fullmatch(name):
            raise ValueError("expected plain train archive basenames")
        expected = metadata if name == "metadata.jsonl" else images.get(sha256(name.encode("utf-8")))
        if expected is None:
            raise ValueError("archive member is outside the fixed training set")
        members[name] = expected
    return members


def equal(raw, pin):
    return len(raw) == pin["bytes"] and sha256(raw) == pin["SHA256"]


def pack(portable, train_dir):
    config = portable.bind_config(train_dir)
    train = Path(config["train_data_dir"])
    entries = checked_members(portable, sorted(file.name for file in train.iterdir()))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, pin in entries.items():
            path = train / name
            portable.local_stat(path, stat.S_ISREG)
            with path.open("rb") as stream:
                raw = stream.read(pin["bytes"] + 1)
            if not equal(raw, pin):
                raise ValueError("training file changed during packaging")
            bundle.writestr(zipfile.ZipInfo(name), raw)
    raw = output.getvalue()
    if len(raw) > MAX_PACKET:
        raise ValueError("train bundle exceeds fixed transfer budget")
    return raw


def unpack(portable, raw, expected_sha):
    if not raw or len(raw) > MAX_PACKET or sha256(raw) != expected_sha:
        raise ValueError("train bundle size or SHA256 differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        infos = bundle.infolist()
        pins_by_name = checked_members(portable, [info.filename for info in infos])
        confirmed = {}
        for info in infos:
            pin = pins_by_name[info.filename]
            if (info.compress_type != zipfile.ZIP_STORED or info.file_size != pin["bytes"]
                    or info.compress_size != pin["bytes"]):
                raise ValueError("archive length or storage type differs")
            with bundle.open(info) as stream:
                body = stream.read(pin["bytes"] + 1)
            if not equal(body, pin):
                raise ValueError("archive member bytes differ")
            confirmed[info.filename] = body
    return confirmed


def install(portable, raw, expected_sha, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("original installation directory exists; inspect it")
    contents = unpack(portable, raw, expected_sha)
    destination.mkdir(mode=0o700, exist_ok=False)
    train = destination / "train"
    train.mkdir(mode=0o700)
    for name, data in contents.items():
        with (train / name).open("xb") as stream:
            stream.write(data)
    portable.bind_config(train)
    receipt = dict(request=REQUEST, binding_SHA256=portable.BINDING_SHA256,
                   packet_SHA256=expected_sha, files=len(contents),
                   train_bytes=sum(map(len, contents.values())), training_READY=False)
    with (destination / RECEIPT).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def inspect(portable, destination, expected_sha):
    if not destination.exists() and not destination.is_symlink():
        return dict(state="ORIGINAL_DATA_NAMESPACE_ABSENT", request=REQUEST, training_READY=False)
    portable.local_stat(destination, stat.S_ISDIR)
    marker = destination / RECEIPT
    if not marker.exists() and not marker.is_symlink():
        return dict(state="ORIGINAL_DATA_NAMESPACE_INCOMPLETE", request=REQUEST, training_READY=False)
    images, metadata = pins(portable)
    expected = dict(request=REQUEST, binding_SHA256=portable.BINDING_SHA256,
                    packet_SHA256=expected_sha, files=len(images) + 1,
                    train_bytes=metadata["bytes"] + sum(p["bytes"] for p in images.values()),
                    training_READY=False)
    portable.local_stat(marker, stat.S_ISREG)
    with marker.open("rb") as stream:
        raw = stream.read(2049)
    if len(raw) > 2048 or json.loads(raw) != expected:
        raise ValueError("original stage receipt differs")
    portable.bind_config(destination / "train")
    return dict(expected, state="SDXL_TRAIN_DATA_STAGED_AND_RECHECKED")


def expired(*_):
    raise TimeoutError("original stage input deadline elapsed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    sender = actions.add_parser("pack")
    sender.add_argument("--train-dir", type=Path, required=True)
    sender.add_argument("--output", type=Path, required=True)
    for name in ("receive", "inspect"):
        receiver = actions.add_parser(name)
        receiver.add_argument("--packet-sha256", required=True)
    args = parser.parse_args()
    portable = load_portable()
    if args.action == "pack":
        raw = pack(portable, args.train_dir)
        with args.output.open("xb") as stream:
            stream.write(raw)
        report = dict(request=REQUEST, packet_SHA256=sha256(raw), packet_bytes=len(raw))
    else:
        if (os.name != "posix" or ROOT.resolve(strict=True) != ROOT
                or TARGET.parent.resolve(strict=True) != TARGET.parent
                or not re.fullmatch("[0-9a-f]{64}", args.packet_sha256)):
            raise ValueError("expected original target HOST and a pinned SHA256")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            if args.action == "receive":
                if TARGET.exists() or TARGET.is_symlink():
                    raise FileExistsError("original installation directory exists; inspect it")
                os.umask(0o077)
                start = time.monotonic()
                raw = sys.stdin.buffer.read(MAX_PACKET + 1)
                report = install(portable, raw, args.packet_sha256, TARGET)
                report.update(state="SDXL_TRAIN_DATA_STAGED", wall_seconds=time.monotonic() - start)
            else:
                report = inspect(portable, TARGET, args.packet_sha256)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
