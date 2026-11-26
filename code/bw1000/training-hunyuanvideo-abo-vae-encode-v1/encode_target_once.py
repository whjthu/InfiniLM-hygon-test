"""Authorize and observe one published native ABO VAE encode on the existing SSH route."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess


PKG = Path(__file__).resolve().parent
REPO = PKG.parents[2]
LOCAL = REPO / "local-only"
LABEL = "training-hunyuanvideo-abo-vae-native-encode-v1"
CLAIM = REPO / "coordination/tasks" / (LABEL + ".json")
ROUTE = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"


def pin(raw):
    return dict(SHA256=sha256(raw).hexdigest(), bytes=len(raw))


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def save(path, value):
    raw = encoded(value)
    with path.open("xb") as f:
        f.write(raw)
    return pin(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    args = parser.parse_args()
    pub = args.publication
    assert re.fullmatch(r"[0-9a-f]{40}", pub)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]

    def query(*parts):
        return subprocess.check_output([*git, *parts], cwd=REPO, stdin=subprocess.DEVNULL, timeout=18)

    assert query("remote", "get-url", "origin").strip() == b"git@github.com:whjthu/InfiniLM-hygon-test.git"
    assert query("ls-remote", "--exit-code", "origin", "refs/heads/bw1000-experiments").split()[0] == pub.encode()
    raw = CLAIM.read_bytes()
    assert query("show", pub + ":" + CLAIM.relative_to(REPO).as_posix()) == raw
    claim = json.loads(raw)
    assert claim["scope"] == dict(parent="1103606", node="f13r3n14", CPU=8, DCU=1, mem_MiB=128000, epoch_seconds=600)
    assert claim["natural_predecessor"] == "1103606.4128" and not claim["replay_allowed"] and not claim["training_GO"]
    source = {}
    for name, expected in claim["encode_sources"].items():
        path = PKG / name
        data = path.read_bytes()
        assert pin(data) == expected and query("show", pub + ":" + path.relative_to(REPO).as_posix()) == data
        compile(data, str(path), "exec")
        source[name] = data
    assert sha256(ROUTE.read_bytes()).hexdigest() == "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
    ssh = runpy.run_path(str(ROUTE))["ssh"](REPO.parent)
    assert ssh[1] == "-v"
    ssh[1:2] = ["-o", "LogLevel=ERROR"]
    request = claim["request"]
    assert re.fullmatch(r"[0-9a-f]{32}", request)
    placement = source["encode_placement.py"]
    wire = encoded(dict(request=request, publication=pub, claim_SHA256=pin(raw)["SHA256"],
                        inspected_members=claim["inspected_members"], placement_pin=pin(placement),
                        placement_source=b64encode(placement).decode()))
    directory = LOCAL / (LABEL + "-" + request)
    directory.mkdir(mode=0o700)
    save(directory / "intent.json", dict(request=request, publication=pub, claim=pin(raw), wire=pin(wire), replay_allowed=False))
    remote = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", source["encode_controller.py"].decode()])
    try:
        child = subprocess.run([*ssh, remote], input=wire, capture_output=True, timeout=690)
        response = json.loads(child.stdout) if child.returncode == 0 and 0 < len(child.stdout) <= 12000 else None
        assert response is None or response["request"] == request
        receipt = dict(request=request, rc=child.returncode, response=response, stderr=pin(child.stderr),
                       outcome="OBSERVED" if response else "UNKNOWN")
    except subprocess.TimeoutExpired:
        receipt = dict(request=request, rc=None, outcome="UNKNOWN")
    print(json.dumps(dict(local_receipt=save(directory / "result.json", receipt), **receipt), sort_keys=True))
    return int(receipt["outcome"] == "UNKNOWN")


if __name__ == "__main__":
    raise SystemExit(main())
