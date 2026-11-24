"""Build and send the reviewed VizWiz train-only packet without persisting a local payload copy."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from uuid import uuid4

import receive_once as receiver


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
LOCAL = REPO / "local-only"
WORK = LOCAL / "specialized06-vizwiz-publisher-split-index-v1"
SOURCE = WORK / "frozen39-train20-calibration-supervision-first/train"
ROWS = {"bytes": 19062, "SHA256": "3ad4d278258df5264e7c2c2eec46a32eaf3a5e98c0c042d79420ba228d8343d6"}
MEMBERSHIP = {"bytes": 14227, "SHA256": "fe8ec456a4050b4603a53260b7e65ebe15e1fcb11b9c17dddf26afa095d61d8e"}
PHOTO_ROOT = WORK / "frozen72-jpeg-image-audit-first/train"
SSH_MODULE = REPO / "code/bw1000/training-gemma-vizwiz-five-data-stage-v1/launch_once.py"
SSH_MODULE_SHA256 = "98c9f1eea7cde26c575a13471accc0cf99144cad6b0267aa42de8734fe183389"
SOURCE_CONTROL = ("claim.json", "launch_once.py", "receive_once.py")


def read(path, pin):
    with path.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    if path.is_symlink() or receiver.identity(raw) != pin:
        raise ValueError("original private source differs")
    return raw


def reviewed_names(quality_file, quality_sha):
    if not receiver.SHA_PATTERN.fullmatch(quality_sha):
        raise ValueError("exact approval evidence SHA is required")
    with quality_file.open("rb") as stream:
        raw = stream.read(receiver.MAX_CONTROL + 1)
    if len(raw) > receiver.MAX_CONTROL or sha256(raw).hexdigest() != quality_sha:
        raise ValueError("exact private review receipt differs")
    approval = json.loads(raw)
    names = approval.get("train_allow")
    if (approval.get("schema") != "bw1000.vizwiz.reviewed-train-release.v1"
            or approval.get("eligible_for_target_staging") is not True
            or not isinstance(names, list) or not 0 < len(names) <= receiver.MAX_ROWS
            or len(set(names)) != len(names) or not all(receiver.FILE_PATTERN.fullmatch(n) for n in names)):
        raise ValueError("no exact reviewed training-only image allowlist")
    return set(names)


def prepare_packet(quality_file, quality_sha):
    names = reviewed_names(quality_file, quality_sha)
    membership = json.loads(read(SOURCE / "membership-and-status.json", MEMBERSHIP))
    manifest = {item["group_SHA256"]: item for item in membership}
    if len(membership) != 39 or len(manifest) != 39:
        raise ValueError("frozen train denominator changed")
    target_data = PurePosixPath(receiver.TARGET.as_posix()) / "data"
    source = read(SOURCE / "model-inputs/train-swift.jsonl", ROWS)
    target_rows, media, payloads = [], {}, {}
    for line in source.splitlines():
        row = json.loads(line)
        image, = row["images"]
        local = Path(image)
        name = local.name
        if name not in names:
            continue
        proof = manifest[row["group"]]
        ordinal = name.rsplit("_", 1)[1][:-4]
        if (name in media or local != PHOTO_ROOT / name or row["id"] != "vizwiz-" + ordinal
                or proof["id"] != "vizwiz-train-" + ordinal or proof["source_split"] != "train"
                or proof["status"] != "PASSED_VOTE_POLICY" or proof["projection"] != "TRAIN_ONLY_SWIFT_ROW"):
            raise ValueError("reviewed train media lost its original source binding")
        pin = proof["jpeg"]
        receiver.check_pin(pin, receiver.MAX_IMAGE)
        media[name] = pin
        payloads[name] = read(local, pin)
        row["images"] = [(target_data / "images" / name).as_posix()]
        target_rows.append(row)
    if set(media) != names:
        raise ValueError("quality receipt selected an unknown or cross-role image")
    binding = receiver.encode(dict(role="train", target_root=target_data.as_posix(),
                                   quality_receipt_SHA256=quality_sha, media=media))
    rows = b"".join(receiver.encode(row) for row in target_rows)
    data = binding + rows + b"".join(payloads[name] for name in sorted(media))
    pins = dict(target=receiver.TARGET.as_posix(), slurm=0, gpu=0, budget_seconds=receiver.BUDGET,
                replay_allowed=False, binding=receiver.identity(binding), rows=receiver.identity(rows),
                quality_receipt_SHA256=quality_sha, train_rows=len(media),
                payload_bytes=len(data), payload_SHA256=sha256(data).hexdigest())
    receiver.validate_scope(pins, binding, rows)
    return pins, data


def published(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("full published Git revision is required")
    command = ["git", "-c", f"safe.directory={REPO.as_posix()}"]
    subprocess.run([*command, "merge-base", "--is-ancestor", revision, "refs/remotes/origin/bw1000-experiments"],
                   cwd=REPO, capture_output=True, check=True, timeout=8)
    for name in SOURCE_CONTROL:
        relative = (PACKAGE / name).relative_to(REPO).as_posix()
        blob = subprocess.check_output([*command, "show", revision + ":" + relative], cwd=REPO,
                                       stderr=subprocess.PIPE, timeout=8)
        if blob != (PACKAGE / name).read_bytes():
            raise ValueError("exact staged source differs from Git")


def same_session():
    raw = SSH_MODULE.read_bytes()
    if sha256(raw).hexdigest() != SSH_MODULE_SHA256:
        raise ValueError("qualified stage launcher helper differs")
    scope = {"__name__": "frozen_vizwiz_transport", "__file__": str(SSH_MODULE)}
    exec(compile(raw, str(SSH_MODULE), "exec"), scope)
    return scope["ssh_command"]()


def send(claim, claim_raw, data, revision):
    published(revision)
    ssh = same_session()
    source = (PACKAGE / "receive_once.py").read_text(encoding="utf-8")
    embedded = b64encode(claim_raw).decode("ascii")
    driver = f"import base64,json\nCLAIM=json.loads(base64.b64decode({embedded!r},validate=True))\n" + source
    remote = "/usr/local/python3.12/bin/python3 -I -S -B -c " + shlex.quote(driver)
    out = LOCAL / ("specialized06-vizwiz-reviewed-train-stage-" + claim["request"])
    out.mkdir()
    report = dict(request=claim["request"], publication=revision, packet=receiver.identity(data),
                  state="UNKNOWN", training_READY=False, replay_allowed=False)
    receiver.save(out / "intent.json", receiver.encode(report))
    try:
        result = subprocess.run([*ssh, remote], input=data, capture_output=True,
                                timeout=receiver.BUDGET + 15)
        report.update(rc=result.returncode, stdout=receiver.identity(result.stdout),
                      stderr=receiver.identity(result.stderr))
        if result.returncode == 0 and 0 < len(result.stdout) <= 4096:
            receipt = json.loads(result.stdout)
            expected = dict(request=claim["request"], state="DATA_STAGED", role="train", gpu=0, slurm=0,
                            train_rows=claim["train_rows"], completed=claim["train_rows"] + 2,
                            payload_bytes=claim["payload_bytes"], optimizer_updates=0, training_READY=False)
            if receipt == expected:
                report.update(state="ORIGINAL_RECEIVER_REPORTED_STAGED", receipt=receipt)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["failure_type"] = type(error).__name__
    finally:
        receiver.save(out / "result.json", receiver.encode(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--quality-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--freeze-claim", action="store_true")
    mode.add_argument("--publication")
    args = parser.parse_args()
    parts, data = prepare_packet(args.quality, args.quality_sha256)
    if args.check:
        print(json.dumps(dict(parts, dispatched=False), sort_keys=True))
        return 0
    claim_path = PACKAGE / "claim.json"
    if args.freeze_claim:
        claim = dict(parts, request=uuid4().hex)
        receiver.check_claim(claim)
        receiver.save(claim_path, receiver.encode(claim))
        print(json.dumps(dict(claim, dispatched=False), sort_keys=True))
        return 0
    content = claim_path.read_bytes()
    claim = json.loads(content)
    receiver.check_claim(claim)
    if claim != dict(parts, request=claim["request"]):
        raise ValueError("actual source changed since the immutable claim")
    result = send(claim, content, data, args.publication)
    print(json.dumps(result, sort_keys=True))
    return int(result["state"] != "ORIGINAL_RECEIVER_REPORTED_STAGED")


if __name__ == "__main__":
    sys.exit(main())
