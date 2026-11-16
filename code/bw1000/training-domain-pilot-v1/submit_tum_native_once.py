"""Submit the published VGGT pilot once using the existing project SSH helper."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import uuid


G = Path(__file__).resolve().parents[3]
CLAIM_PATH = "coordination/tasks/training-vggt-tum-indoor-first-update-v2.json"


def digest(raw):
    return sha256(raw).hexdigest()


def pin(raw, expected):
    if digest(raw) != expected:
        raise ValueError("published asset digest differs")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper", type=Path)
    parser.add_argument("pair_directory", type=Path)
    args = parser.parse_args()
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + G.as_posix(), "-C", str(G)]

    def read_file(name):
        return subprocess.check_output([*git, "show", args.publication + ":" + name])

    ref = subprocess.check_output([*git, "ls-remote", "--exit-code", "origin", "refs/heads/bw1000-experiments"],
                                  timeout=12).decode().split()[0]
    if ref != args.publication:
        raise ValueError("expected published claim at remote branch head")
    claim = json.loads(read_file(CLAIM_PATH))
    if claim["node"] != "f11r4n19" or claim["parent"] != "1087656" or claim["device_count"] != 1:
        raise ValueError("unexpected published resource scope")
    sources = {"src/" + name: pin(read_file(row["path"]), row["sha"]) for name, row in claim["sources"].items()}
    pair = pin((args.pair_directory / "fixed-pair-manifest-mcap-v1.json").read_bytes(), claim["pair_sha"])
    files = {**sources, "data/pair.json": pair}
    for name, hash_value in json.loads(pair)["files_SHA256"].items():
        path = (args.pair_directory / name).resolve()
        if not path.is_relative_to(args.pair_directory.resolve()):
            raise ValueError("pair member leaves approved directory")
        files["data/" + name] = pin(path.read_bytes(), hash_value)
    if sum(len(raw) for name, raw in files.items() if name.startswith("data/")) > 2 * 1024**2:
        raise ValueError("published pilot data exceeds the transfer scope")
    pin(args.ssh_helper.read_bytes(), "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1")
    ssh = runpy.run_path(str(args.ssh_helper))["ssh"]
    request = uuid.uuid4().hex
    payload = dict(publication=args.publication, request=request, files={name: dict(
        sha=digest(raw), bytes=len(raw), b64=base64.b64encode(raw).decode("ascii")) for name, raw in files.items()})
    body = ("PAYLOAD=" + repr(payload) + "\n").encode() + sources["src/run_tum_native_once.py"]
    out = G / "local-only/training-vggt-tum-indoor-first-update-v2"
    out.mkdir()
    with (out / "request.json").open("x") as stream:
        json.dump(dict(request=request, publication=args.publication, body_SHA256=digest(body)), stream)
    print(json.dumps(dict(stage="request", request=request)), flush=True)
    with (out / "stdout").open("xb") as stdout, (out / "stderr").open("xb") as stderr:
        try:
            result = subprocess.run([*ssh, "/usr/local/python3.12/bin/python3 -I -S -B -"], input=body,
                                    stdout=stdout, stderr=stderr, timeout=700)
        except subprocess.TimeoutExpired:
            result = None
    raw = (out / "stdout").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
    print(json.dumps(dict(request=request, ssh_rc=result.returncode if result else None,
                          remote_observations=rows, stdout_SHA256=digest(raw),
                          stderr_SHA256=digest((out / "stderr").read_bytes()))))


if __name__ == "__main__":
    main()
