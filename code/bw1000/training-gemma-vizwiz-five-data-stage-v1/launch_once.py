"""Verify and send exactly the prepared VizWiz pilot bytes through the owned route."""

import argparse
import base64
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile

import receive_once as receiver


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
PRIVATE = REPO / "local-only"
BINDING_DIR = PRIVATE / "specialized06-vizwiz-native-target-binding-v1"
IMAGES = PRIVATE / "specialized06-vizwiz-official-train-pilot-v2/images"
SSH_HELPER = PRIVATE / "run-minicpm-sdpa-target-asset-preflight-original.py"
SSH_HELPER_SHA256 = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
MEMBERS = ("claim.json", "receive_once.py", "launch_once.py")


def read(path, expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected["bytes"]:
        raise ValueError("prepared input member differs")
    with path.open("rb") as stream:
        return receiver.take(stream, expected)


def payload(claim):
    receiver.check_claim(claim)
    binding = read(BINDING_DIR / "data-binding-private.json", receiver.BINDING)
    rows = read(BINDING_DIR / "train-swift.jsonl", receiver.ROWS)
    media = receiver.manifest(binding, rows, PurePosixPath(claim["target"]) / "data")
    payloads = [binding, rows]
    for name, pin in sorted(media.items()):
        payloads.append(read(IMAGES / name, {key: pin[key] for key in ("bytes", "SHA256")}))
    return b"".join(payloads)


def exact_publication(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("exact publication commit is required")
    prefix = PACKAGE.relative_to(REPO).as_posix()
    publication = subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "merge-base", "--is-ancestor", commit,
         "refs/remotes/origin/bw1000-experiments"], cwd=REPO, capture_output=True, timeout=8,
    )
    if publication.returncode != 0:
        raise ValueError("stage source has no matching published ancestry")
    for name in MEMBERS:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "show", f"{commit}:{prefix}/{name}"],
            cwd=REPO, capture_output=True, check=True, timeout=8,
        )
        if result.stdout != (PACKAGE / name).read_bytes():
            raise ValueError("local staged source differs from publication")


def ssh_command():
    raw = SSH_HELPER.read_bytes()
    if sha256(raw).hexdigest() != SSH_HELPER_SHA256:
        raise ValueError("qualified SSH helper differs")
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("owned_ssh", SSH_HELPER))
    exec(compile(raw, str(SSH_HELPER), "exec"), vars(module))
    ssh = module.ssh(REPO.parent)
    ssh.remove("-v")
    return ssh


def target_command(claim_raw):
    encoded = base64.b64encode(claim_raw).decode("ascii")
    prefix = f"import json,base64\nCLAIM=json.loads(base64.b64decode({encoded!r}))\n"
    return ("/usr/local/python3.12/bin/python3 -I -S -B -c "
            + shlex.quote(prefix + (PACKAGE / "receive_once.py").read_text(encoding="utf-8")))


def prepare_check(claim, data):
    request = claim["request"]
    scope = PurePosixPath(claim["target"]) / "data"
    with tempfile.TemporaryDirectory(prefix="gemma-vizwiz-stage-check-") as temp:
        root = Path(temp).resolve() / "staged"
        result = receiver.install(root, request, BytesIO(data), scope)
        if result["state"] != "DATA_STAGED" or result["completed"] != 7 or result["payload_bytes"] != len(data):
            raise ValueError("exact local roundtrip differs")
        try:
            receiver.install(root, request, BytesIO(data), scope)
        except FileExistsError:
            pass
        else:
            raise ValueError("original local namespace replay was accepted")
        for index, bad in enumerate((data[:-1], data + b"?", b"!" + data[1:],
                                     data[:-1] + bytes([data[-1] ^ 0xff]))):
            bad_root = Path(temp).resolve() / f"bad-{index}"
            try:
                receiver.install(bad_root, request, BytesIO(bad), scope)
            except ValueError:
                if (bad_root / "data").exists():
                    raise ValueError("bad packet reached the visible data root")
            else:
                raise ValueError("bad packet was accepted")
    return dict(request=request, ready_for_target_staging=True, local_roundtrip_files=7,
                bad_packet_rejections=4, duplicate_rejections=1, local_packet=receiver.identity(data),
                optimizer_updates=0, installed_target_files=0, training_READY=False)


def send(claim, claim_raw, data, publication):
    exact_publication(publication)
    command = ssh_command()
    request = claim["request"]
    out = PRIVATE / ("specialized06-vizwiz-stage-original-" + request)
    out.mkdir()
    proof = dict(request=request, publication=publication, packet=receiver.identity(data), replay_allowed=False)
    receiver.save(out / "intent.json", receiver.as_json(proof))
    result = dict(proof, state="UNKNOWN", target_installed_files=None, training_READY=False)
    try:
        attempt = subprocess.run([*command, target_command(claim_raw)], input=data, capture_output=True, timeout=110)
        result.update(returncode=attempt.returncode, stdout=receiver.identity(attempt.stdout),
                      stderr=receiver.identity(attempt.stderr))
        if attempt.returncode == 0 and 0 < len(attempt.stdout) <= 4096:
            receipt = json.loads(attempt.stdout)
            expected = dict(request=request, state="DATA_STAGED", completed=7, payload_bytes=len(data),
                            slurm=0, gpu=0, optimizer_updates=0, training_READY=False)
            if receipt == expected:
                result.update(state="ORIGINAL_RECEIVER_REPORTED_STAGED", target_installed_files=7,
                              receiver_receipt=receipt)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["failure_type"] = type(error).__name__
    finally:
        receiver.save(out / "result.json", receiver.as_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    options = parser.parse_args()
    claim_raw = (PACKAGE / "claim.json").read_bytes()
    claim = json.loads(claim_raw)
    data = payload(claim)
    result = prepare_check(claim, data) if options.check else send(claim, claim_raw, data, options.publication)
    print(json.dumps(result, sort_keys=True))
    return 0 if options.check or result["state"] == "ORIGINAL_RECEIVER_REPORTED_STAGED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
