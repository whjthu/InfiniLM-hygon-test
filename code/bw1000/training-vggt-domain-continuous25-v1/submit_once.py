"""Submit the published two-scene VGGT training and recovery scope exactly once."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import uuid


G = Path(__file__).resolve().parents[3]
CLAIM_PATH = "coordination/tasks/training-vggt-tum-continuous25-recovery-v1.json"
LABEL = "training-vggt-tum-continuous25-recovery-v1"
HELPER_SHA256 = "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"


def digest(raw):
    return sha256(raw).hexdigest()


def pin(raw, expected):
    if digest(raw) != expected:
        raise ValueError("Published source or data identity differs")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper", type=Path)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + G.as_posix(), "-C", str(G)]

    def published(path):
        return subprocess.check_output([*git, "show", args.publication + ":" + path], timeout=10)

    head = subprocess.check_output([*git, "ls-remote", "--exit-code", "origin",
                                    "refs/heads/bw1000-experiments"], timeout=12).decode().split()[0]
    if head != args.publication:
        raise ValueError("Expected the published claim at the current branch head")
    claim = json.loads(published(CLAIM_PATH))
    if (claim["label"], claim["parent"], claim["node"], claim["device_count"], claim["whole_seconds"]) != (
            LABEL, "1087656", "f11r4n19", 1, 1800):
        raise ValueError("Unexpected published resource scope")
    sources = {"src/" + name: pin(published(info["path"]), info["sha"])
               for name, info in claim["sources"].items()}
    if sources["src/submit_once.py"] != Path(__file__).read_bytes():
        raise ValueError("The invoked launcher differs from its publication")
    files = dict(sources)
    for name, info in claim["data_files"].items():
        path = args.bundle / name
        if path.is_symlink() or not path.resolve().is_relative_to(args.bundle.resolve()):
            raise ValueError("Data member escapes the frozen bundle")
        raw = pin(path.read_bytes(), info["sha"])
        if len(raw) != info["bytes"]:
            raise ValueError("Data member size differs")
        files["data/" + name] = raw
    if len(claim["data_files"]) != 17 or sum(len(v) for k, v in files.items() if k.startswith("data/")) > 4 * 1024**2:
        raise ValueError("Unexpected bounded training data payload")
    pin(args.ssh_helper.read_bytes(), HELPER_SHA256)
    ssh = runpy.run_path(str(args.ssh_helper))["ssh"]
    request = uuid.uuid4().hex
    payload = dict(publication=args.publication, request=request, files={name: dict(
        sha=digest(raw), bytes=len(raw), b64=base64.b64encode(raw).decode("ascii")) for name, raw in files.items()})
    body = ("PAYLOAD=" + repr(payload) + "\n").encode("ascii") + sources["src/run_tum_continuous_once.py"]
    directory = G / "local-only" / (LABEL + "-dispatch")
    directory.mkdir()
    (directory / "request.json").write_bytes((json.dumps(dict(
        request=request, publication=args.publication, body_SHA256=digest(body),
        original_label=LABEL, remote_outcome="UNKNOWN_UNTIL_ACTUAL_RECEIPT")) + "\n").encode("ascii"))
    print(json.dumps(dict(stage="request", request=request)), flush=True)
    with (directory / "stdout").open("xb") as out, (directory / "stderr").open("xb") as err:
        try:
            result = subprocess.run([*ssh, "/usr/local/python3.12/bin/python3 -I -S -B -"],
                                    input=body, stdout=out, stderr=err, timeout=1830)
        except subprocess.TimeoutExpired:
            result = None
    stdout = (directory / "stdout").read_bytes()
    # Raw training records remain remote. Controller stdout contains stage receipts only.
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith(b"{")]
    print(json.dumps(dict(request=request, ssh_rc=result.returncode if result else None,
                          observations=rows, stdout_SHA256=digest(stdout),
                          stderr_SHA256=digest((directory / "stderr").read_bytes()))))


if __name__ == "__main__":
    main()
