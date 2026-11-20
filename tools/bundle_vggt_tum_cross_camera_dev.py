"""Pack and verify only the fixed VGGT development RGB224 files and their manifest."""

import argparse
import io
import json
from pathlib import Path
import zipfile

from bind_vggt_tum_dev_rgb import CALIBRATION, read_bounded, sha
from join_vggt_tum_cross_camera_dev import COMPLETE, FAILED


MANIFEST = "inputs-private.json"
MAX_MANIFEST_BYTES = 65536
MAX_RGB224_BYTES = 310000
MAX_BUNDLE_BYTES = 8 * 1024**2


def members(manifest_raw, expected):
    if sha(manifest_raw) != expected:
        raise ValueError("combined cohort manifest SHA differs")
    doc = json.loads(manifest_raw)
    windows = doc["windows"]
    identities = [(row["sequence"], row["window"]) for row in windows]
    planned = {(scene, window) for scene in CALIBRATION for window in range(1, 13)}
    if doc["scope"] != "VGGT_TUM_cross_camera_development" or len(identities) != 24 or set(identities) != planned:
        raise ValueError("expected all 24 windows from the combined scene manifest")
    files = {}
    admitted = set()
    for row in windows:
        if row["state"] == FAILED:
            continue
        if row["state"] != COMPLETE:
            raise ValueError("invalid original cohort window state")
        name = f"{row['sequence']}-window-{row['window']:02d}-rgb224.npy"
        if name != row["rgb_file"]:
            raise ValueError("manifest member path differs")
        size = row["rgb_file_bytes"]
        if type(size) is not int or not 0 < size <= MAX_RGB224_BYTES:
            raise ValueError("outside the RGB224 file byte bound")
        files[name] = (size, row["rgb_file_SHA256"])
        admitted.add(row["sequence"])
    if admitted != CALIBRATION.keys():
        raise ValueError("both official scenes need admitted RGB data")
    return files


def verify_media(raw, expected):
    if (len(raw), sha(raw)) != expected:
        raise ValueError("RGB224 file does not match the input manifest")
    return raw


def pack(inputs, inputs_sha, output):
    raw = read_bounded(inputs, MAX_MANIFEST_BYTES)
    files = members(raw, inputs_sha)
    blobs = {name: verify_media(read_bounded(inputs.parent / name, MAX_RGB224_BYTES), identity)
             for name, identity in files.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in [(MANIFEST, raw), *sorted(blobs.items())]:
            bundle.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    result = buffer.getvalue()
    if len(result) > MAX_BUNDLE_BYTES:
        raise ValueError("combined input bundle exceeds the finite byte bound")
    with output.open("xb") as stream:
        stream.write(result)
    return dict(bundle_bytes=len(result), bundle_SHA256=sha(result), inputs_SHA256=inputs_sha,
                packed_members=len(files)+1, input_pairs=len(files), planned_windows=24)


def unpack(source, bundle_sha, inputs_sha, output):
    raw = read_bounded(source, MAX_BUNDLE_BYTES)
    if sha(raw) != bundle_sha:
        raise ValueError("exact private input bundle SHA differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        items = {entry.filename: entry for entry in bundle.infolist()}
        if len(items) != len(bundle.infolist()) or MANIFEST not in items:
            raise ValueError("input bundle includes missing or duplicate members")
        if not 0 < items[MANIFEST].file_size <= MAX_MANIFEST_BYTES:
            raise ValueError("input bundle cohort manifest size differs")
        with bundle.open(MANIFEST) as stream:
            manifest = stream.read(MAX_MANIFEST_BYTES + 1)
        files = members(manifest, inputs_sha)
        if set(items) != {MANIFEST, *files}:
            raise ValueError("input bundle membership differs from the frozen manifest")
        blobs = {}
        for name, identity in files.items():
            if items[name].file_size != identity[0]:
                raise ValueError("input bundle RGB224 member size differs")
            with bundle.open(name) as stream:
                blobs[name] = verify_media(stream.read(MAX_RGB224_BYTES + 1), identity)
    output.mkdir(mode=0o700)
    for name, data in blobs.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    with (output / MANIFEST).open("xb") as stream:
        stream.write(manifest)
    return dict(bundle_SHA256=bundle_sha, inputs_SHA256=inputs_sha,
                verified_members=len(files)+1, input_pairs=len(files), planned_windows=24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    packing = sub.add_parser("pack")
    extracting = sub.add_parser("unpack")
    packing.add_argument("--inputs", type=Path, required=True)
    extracting.add_argument("--bundle", type=Path, required=True)
    extracting.add_argument("--bundle-sha", required=True)
    for command in (packing, extracting):
        command.add_argument("--inputs-sha", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (pack(args.inputs, args.inputs_sha, args.output) if args.mode == "pack"
              else unpack(args.bundle, args.bundle_sha, args.inputs_sha, args.output))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
