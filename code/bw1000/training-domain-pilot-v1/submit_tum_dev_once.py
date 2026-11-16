"""Submit the frozen VGGT before/after development comparison once."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import uuid

from render_tum_dev_controller import LABEL, render
from submit_tum_native_once import G, pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + G.as_posix(), "-C", str(G)]

    def from_git(name):
        return subprocess.check_output([*git, "show", args.publication + ":" + name])

    head = subprocess.check_output([*git, "ls-remote", "--exit-code", "origin", "refs/heads/bw1000-experiments"],
                                   timeout=12).decode().split()[0]
    if head != args.publication:
        raise ValueError("expected published development claim at remote head")
    claim = json.loads(from_git("coordination/tasks/" + LABEL + ".json"))
    files = {"src/" + name: pin(from_git(row["path"]), row["sha"]) for name, row in claim["sources"].items()}
    data = pin(args.manifest.read_bytes(), claim["pair_sha"])
    pair = json.loads(data)
    if claim["sequence"] != pair["sequence"] or (claim["parent"], claim["node"], claim["device_count"]) != (
            "1087656", "f11r4n19", 1):
        raise ValueError("development pair or resource scope differs")
    root = args.manifest.resolve().parent
    total = len(data)
    files["data/pair.json"] = data
    for name, sha in pair["files_SHA256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("frozen member is outside the development directory")
        raw = pin(path.read_bytes(), sha)
        files["data/" + name] = raw
        total += len(raw)
    if total != claim["pair_payload_bytes"] or total > 3 * 1024**2:
        raise ValueError("development pair byte budget differs")
    controller = pin(render(files["src/run_tum_native_once.py"], claim["pair_sha"], claim["sequence"]),
                     claim["generated_controller_sha"])
    files["src/run_tum_native_once.py"] = controller
    pin(args.ssh_helper.read_bytes(), "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1")
    ssh = runpy.run_path(str(args.ssh_helper))["ssh"]
    request = uuid.uuid4().hex
    payload = dict(publication=args.publication, request=request, files={name: dict(
        sha=sha256(raw).hexdigest(), bytes=len(raw), b64=base64.b64encode(raw).decode("ascii")) for name, raw in files.items()})
    body = ("PAYLOAD=" + repr(payload) + "\n").encode() + controller
    out = G / "local-only" / LABEL
    out.mkdir()
    with (out / "request.json").open("x") as stream:
        json.dump(dict(request=request, publication=args.publication, body_SHA256=sha256(body).hexdigest()), stream)
    print(json.dumps(dict(stage="request", request=request)), flush=True)
    with (out / "stdout").open("xb") as stdout, (out / "stderr").open("xb") as stderr:
        try:
            result = subprocess.run([*ssh, "/usr/local/python3.12/bin/python3 -I -S -B -"], input=body,
                                    stdout=stdout, stderr=stderr, timeout=700)
        except subprocess.TimeoutExpired:
            result = None
    stdout = (out / "stdout").read_bytes()
    observations = [json.loads(line) for line in stdout.splitlines() if line.startswith(b"{")]
    print(json.dumps(dict(request=request, ssh_rc=result.returncode if result else None,
                          remote_observations=observations, stdout_SHA256=sha256(stdout).hexdigest(),
                          stderr_SHA256=sha256((out / "stderr").read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
