"""Stage the frozen, user-only GQA comparison on SCNet in one bound transfer."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from run_hygon_bound_payload import digest, document, send


REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "code/bw1000/training-domain-pilot-v1/internvl_gqa_val96.py"
sys.path.insert(0, str(EVAL.parent))
import internvl_gqa_val96 as evaluation


INPUT_SHA = "98f729fce9e6af91edb10bbf4dcc9b0d06f1322a38247c7001d05b02a21fc46b"
INPUT_BYTES = 14097308
REQUEST = "8fbbbada771e402a9ed57aa0229b27f9"
LABEL = "training-internvl-gqa-val96-user-only-v1"


REMOTE = r'''
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_STORED

def need(condition, reason):
    if not condition:
        raise ValueError(reason)

def digest(data):
    return sha256(data).hexdigest()

def stamp(data):
    return {"bytes": len(data), "SHA256": digest(data)}

def save(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def main():
    packet = sys.stdin.buffer.read(20 * 1024**2 + 1)
    need(len(packet) == PACKET_BYTES and digest(packet) == PACKET_SHA, "Archive extent or hash differs")
    media_name = re.compile(r"media/[0-9]+\.jpg")
    with ZipFile(BytesIO(packet)) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        need(len(names) == len(entries) == 98 and {"INPUT.json", "prompts.json"} <= names
             and all(media_name.fullmatch(name) for name in names - {"INPUT.json", "prompts.json"})
             and all(entry.compress_type == ZIP_STORED and 0 < entry.file_size <= 2 << 20
                     for entry in entries) and sum(entry.file_size for entry in entries) == INPUT_BYTES,
             "Unexpected archive surface")
        manifest_raw = archive.read("INPUT.json")
        need(digest(manifest_raw) == INPUT_SHA, "Frozen input manifest differs")
        manifest = json.loads(manifest_raw)
        need(manifest["schema"] == "INTERNVL_GQA_VAL96_INPUT_V1" and manifest["images"] == 96
             and manifest["gold_included"] is False and set(manifest["members"]) == names - {"INPUT.json"},
             "User-only input manifest required")
        data = {name: archive.read(name) for name in sorted(names)}
        need(all(stamp(data[name]) == pin for name, pin in manifest["members"].items()), "Input member differs")
    os.umask(0o077)
    stage = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / LABEL
    stage.mkdir(mode=0o700)
    root = stage / "input"
    root.mkdir(mode=0o700)
    (root / "media").mkdir(mode=0o700)
    for name, value in data.items():
        save(root / name, value)
    need(all(stamp((root / name).read_bytes()) == stamp(value) for name, value in data.items()),
         "Remote stage read-back differs")
    result = {"request": REQUEST, "rc": 0, "input_SHA256": INPUT_SHA, "archive_SHA256": PACKET_SHA,
              "images": 96, "files": len(data), "published_source": PUBLICATION, "gold_included": False,
              "files_read_back": len(data), "compute_native_executed": False}
    receipt = (json.dumps(result, sort_keys=True) + "\n").encode()
    save(stage / "verified-upload.json", receipt)
    print(json.dumps({**result, "receipt_SHA256": digest(receipt)}, sort_keys=True), flush=True)

try:
    main()
except (ValueError, OSError, KeyError) as error:
    print(json.dumps({"request": REQUEST, "rc": 1, "class": type(error).__name__,
                      "reason_SHA256": digest(str(error).encode())}, sort_keys=True), flush=True)
    sys.exit(1)
'''


def pack(root):
    assets = evaluation.frozen_members(root)
    if evaluation.digest(assets["INPUT.json"]) != INPUT_SHA:
        raise ValueError("Frozen input manifest differs")
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        for name, raw in assets.items():
            archive.writestr(ZipInfo(name), raw)
    if sum(map(len, assets.values())) != INPUT_BYTES:
        raise ValueError("Frozen input byte total differs")
    return output.getvalue()


def program(packet, publication):
    binding = dict(REQUEST=REQUEST, LABEL=LABEL, INPUT_SHA=INPUT_SHA, INPUT_BYTES=INPUT_BYTES,
                   PACKET_BYTES=len(packet), PACKET_SHA=digest(packet), PUBLICATION=publication)
    source = "\n".join(f"{key}={value!r}" for key, value in binding.items()) + "\n" + REMOTE
    compile(source, "GQA96 remote staging", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--publication")
    args = parser.parse_args()
    local = REPO / "local-only"
    packet = pack(local / "internvl-gqa-val96-blind-frozen-v1")
    source = program(packet, args.publication)
    if args.check:
        print(json.dumps({"request": REQUEST, "input_SHA256": INPUT_SHA, "archive_SHA256": digest(packet),
                          "archive_bytes": len(packet), "local_check_only": True}, sort_keys=True))
        return 0
    if not re.fullmatch("[0-9a-f]{40}", args.publication):
        raise ValueError("Exact published source revision required")
    for path in (Path(__file__).resolve(), EVAL, REPO / "tools/run_hygon_bound_payload.py"):
        key = path.relative_to(REPO).as_posix()
        published = subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}",
                                             "show", args.publication + ":" + key], cwd=REPO)
        if published != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Publication bytes differ")
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing project transport differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    result = send(transport.ssh(REPO.parent), request=REQUEST, program=source, data=packet,
                  send_token=local / f"internvl-gqa96-upload-{REQUEST}-send.json",
                  packet_cap=27 << 20, write_seconds=150, outcome_seconds=55)
    receipt = local / f"internvl-gqa96-upload-{REQUEST}-{uuid4().hex}.json"
    evaluation.write(receipt, document(result))
    response = result.get("response") or {}
    print(json.dumps({"request": REQUEST, "receipt": receipt.name, "remote_outcome": result["remote_outcome"],
                      "remote_received": result["remote_received"], "response": response}, sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED" or response.get("rc") != 0)


if __name__ == "__main__":
    raise SystemExit(main())
