"""Launch the exact published CPU-only cache builder over the established project route."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
LOCAL = REPO / "local-only"
LABEL = "training-hunyuan-abo-native-text-cache-v1"
CLAIM = REPO / "coordination/tasks" / (LABEL + ".json")
ROUTE = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"


def pin(raw):
    return sha256(raw).hexdigest()


def save(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(data)
    return pin(data)


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--publication", required=True)
    publication = arg.parse_args().publication
    if not re.fullmatch("[a-f0-9]{40}", publication):
        raise ValueError("exact publication required")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]

    def query(*parts):
        return subprocess.check_output([*git, *parts], cwd=REPO, stdin=subprocess.DEVNULL, timeout=22)

    def published(path):
        raw = path.read_bytes()
        if query("show", publication + ":" + path.relative_to(REPO).as_posix()) != raw:
            raise ValueError("source differs from the published bytes")
        return raw

    ref = "refs/heads/bw1000-experiments"
    if (query("remote", "get-url", "origin").strip() != b"git@github.com:whjthu/InfiniLM-hygon-test.git"
            or query("ls-remote", "--exit-code", "origin", ref).split()[0] != publication.encode()):
        raise ValueError("experiment branch does not match this publication")
    claim = published(CLAIM)
    scope = json.loads(claim)
    request = scope["request"]
    if (not re.fullmatch("[a-f0-9]{32}", request) or
            scope["placement"] != dict(parent="1103606", node="f13r3n14", DCU=0, CPUs=12, memory_MiB=114688,
                                       single_epoch_seconds=900) or scope["training_GO"] or scope["replay_allowed"]):
        raise ValueError("exact CPU-only scope is required")
    members = scope["source_SHA256"]
    if set(members) != {"native_text.py", "build_cache.py", "cache_controller.py", "run_cache_once.py"}:
        raise ValueError("exact source set required")
    sources = {name: published(PACKAGE / name) for name in members}
    if any(pin(raw) != members[name] for name, raw in sources.items()):
        raise ValueError("source digest changed")
    if pin(ROUTE.read_bytes()) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("established route differs")
    ssh = runpy.run_path(str(ROUTE))["ssh"](REPO.parent)
    if ssh[1] != "-v":
        raise ValueError("unexpected authenticated route option")
    ssh[1:2] = ["-o", "LogLevel=ERROR"]
    files = {name: b64encode(sources[name]).decode("ascii") for name in ("native_text.py", "build_cache.py")}
    wire = json.dumps(dict(request=request, publication=publication, claim_SHA256=pin(claim),
                           files=files, pins={name: members[name] for name in files}), sort_keys=True).encode()
    local = LOCAL / (LABEL + "-" + request)
    local.mkdir(mode=0o700)
    save(local / "intent.json", dict(request=request, publication=publication, claim_SHA256=pin(claim),
                                   wire_SHA256=pin(wire), replay_allowed=False))
    remote = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c",
                         sources["cache_controller.py"].decode("utf-8")])
    try:
        child = subprocess.run([*ssh, remote], input=wire, capture_output=True, timeout=930)
        response = json.loads(child.stdout) if child.returncode == 0 and 0 < len(child.stdout) < 24_000 else None
        if response is not None and response["request"] != request:
            raise ValueError("remote response request changed")
        observation = dict(rc=child.returncode, response=response, stdout_SHA256=pin(child.stdout),
                           stderr_SHA256=pin(child.stderr), outcome="OBSERVED" if response else "UNKNOWN")
    except subprocess.TimeoutExpired:
        observation = dict(rc=None, outcome="UNKNOWN")
    receipt = dict(request=request, **observation)
    print(json.dumps(dict(local_receipt_SHA256=save(local / "result.json", receipt), **receipt), sort_keys=True))
    return int(observation["outcome"] != "OBSERVED")


if __name__ == "__main__":
    raise SystemExit(main())
