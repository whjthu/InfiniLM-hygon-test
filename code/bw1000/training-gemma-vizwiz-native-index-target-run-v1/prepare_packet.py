"""Build the source-only packet or prove a future actual publication by Git readback."""

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import subprocess

import runner


ROOT = Path(__file__).resolve().parents[3]
REF = "refs/remotes/origin/bw1000-experiments"


def encoded(raw):
    return b64encode(raw).decode("ascii")


def packet(root):
    raw = (root / runner.PACKAGE / "SOURCE_FILES.json").read_bytes()
    closure = json.loads(raw)
    files = {name: encoded((root / item["source"]).read_bytes()) for name, item in closure["members"].items()}
    body = runner.encode({"closure": encoded(raw), "files": files})
    runner.decode_packet(body, runner.pin(body)["SHA256"])
    return body


def expected_members(root, closure):
    paths = [item["source"] for item in closure["members"].values()]
    paths.extend(runner.PACKAGE + "/" + name for name in ("SOURCE_FILES.json", "runner.py"))
    return {name: runner.pin((root / name).read_bytes()) for name in paths}


def actual_publication(root, request, commit):
    runner.scope_name(request)
    runner.require(re.fullmatch("[0-9a-f]{40}", commit), "EXACT_PUBLISHED_COMMIT")

    def read(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=8)

    read("merge-base", "--is-ancestor", commit, REF)
    tree = read("rev-parse", commit + "^{tree}").decode("ascii").strip()
    raw = (root / runner.PACKAGE / "SOURCE_FILES.json").read_bytes()
    runner.require(runner.pin(raw)["SHA256"] == runner.CLOSURE_SHA, "SOURCE_CLOSURE_DIGEST")
    closure = json.loads(raw)
    expected = expected_members(root, closure)
    actual = {name: runner.pin(read("show", commit + ":" + name)) for name in expected}
    runner.require(actual == expected, "PUBLISHED_MEMBER_DIFFERS")
    proof = {"schema": runner.SCHEMA + "-publication", "request": request, "ref": REF, "commit": commit,
             "tree": tree, "source_closure_SHA256": runner.CLOSURE_SHA, "commit_verified_in_ref": True,
             "members": actual}
    body = runner.encode(proof)
    runner.check_publication(body, request, closure)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("packet")
    p.add_argument("--output", type=Path, required=True)
    proof = sub.add_parser("publication-proof")
    proof.add_argument("--request", required=True)
    proof.add_argument("--commit", required=True)
    proof.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = packet(ROOT) if args.mode == "packet" else actual_publication(ROOT, args.request, args.commit)
    runner.write(args.output, data)
    print(json.dumps(runner.pin(data), sort_keys=True))


if __name__ == "__main__":
    main()
