"""Adopt the original ordinary refill launcher with the published 48 GiB guard."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import ModuleType
import zlib


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEMORY = HERE.with_name("zimage-o01-eight-card-memory48-launch-v1")
PRIVATE = REPO / "local-only/o01-eight-card-refill-monitor-v1"
STAGE_SHA = "de5a309d8027d2cc8523c857afcb97186a8dbc4ada862d2d6db0e12611abc27e"
BRIDGE_SHA = "dcf3099f541bd5e9163db4f7d30f040acc0fc1faaea9fb677f49dec02eb5491e"
PENDING_SHA = "82987bab4be9ff5e527469bf39d08f7d1d03ca01ad0666fd7e3f0a0bf948014d"
PREVIOUS_STAGE = "O01-zimage-lt-eight-card-supervisor-v1-host-stage"
CURRENT_STAGE = "O01-zimage-lt-eight-card-memory48-v1-host-stage"
PREVIOUS_PENDING_SHA = "36a2bee98795384bb7ab4f7810e3dcdbfbe2e958aadecdb78bcd8576e8f1e8b9"
REQUEST = "274608fc6e0d410a8f545287843cb887"
PRIVATE_PINS = {"prepare.py": "dbf8d11030e1c2e2970c5a8f782867a61a2baef5beb5fe696797c02c03be554a",
                "launch.py": "538e4dcfd0c5afb46797073ec775a8f4845632386d626bafbd71e51b0af4c59f",
                "observe_refill.py": "3ef0d01908ac415b397f9c061f10e9836024780ccd5fd1c8b1e25c8a722788d0"}
TRANSPORT_SHA = "774650b555bc0d30b018c970a870edd120fe193dff21cb34309f74a447d816e5"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def exact(path, wanted, cap=65536):
    need(not path.is_symlink(), "unexpected source symlink")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    need(0 < len(raw) <= cap and sha256(raw).hexdigest() == wanted, "frozen bytes differ: " + path.name)
    return raw


def module(path, source, **names):
    adopted = ModuleType("memory48_refill_" + path.stem)
    adopted.__file__ = str(path)
    vars(adopted).update(names)
    exec(compile(source, str(path), "exec"), vars(adopted))
    return adopted


def replace(raw, before, after):
    x, y = before.encode(), after.encode()
    need(raw.count(x) == 1, "frozen source anchor differs")
    return raw.replace(x, y, 1)


def source(name):
    return exact(PRIVATE / name, PRIVATE_PINS[name])


def approved_publication(dispatch, publication):
    need(re.fullmatch("[0-9a-f]{40}", publication), "a verified ordinary publication is required")
    directories = (HERE, MEMORY, REPO / "code/bw1000/fleet-dcu-standby-v2",
                   REPO / "code/bw1000/fleet-dcu-standby-compute-bridge-v1")
    members = ("refill.py", "SOURCE_FILES.json", "render.py", "memory_link.py", "transport.py",
               "CLAIM.json", "controller.py", "worker.py", "dispatch.py", "bridge.py")
    for directory in directories:
        for name in members:
            path = directory / name
            if path.is_file():
                object_id = f"{publication}:{path.relative_to(REPO).as_posix()}"
                raw = subprocess.check_output(["git", "-c", "safe.directory=" + REPO.as_posix(),
                    "cat-file", "blob", object_id], cwd=REPO, timeout=8, stdin=subprocess.DEVNULL)
                need(raw == path.read_bytes(), "approved publication bytes differ")


def prepare_adapter(stage):
    manifest = json.loads(exact(stage / "HOST_STAGE_FILES.json", STAGE_SHA))
    need(manifest["original_request"] == REQUEST and
         manifest["files"]["guarded-standby-bridge.py"]["sha256"] == BRIDGE_SHA and
         manifest["files"]["pending.py"]["sha256"] == PENDING_SHA, "wrong 48 GiB stage pins")
    original = PRIVATE / "prepare.py"
    adopted = module(original, source("prepare.py"))
    adopted.GUARDED_SHA256 = BRIDGE_SHA

    def load_sources():
        pending = module(stage / "pending.py", exact(stage / "pending.py", PENDING_SHA))
        need(pending.REQUEST == REQUEST, "wrong pending request")
        dispatch = adopted.load_module("published_standard_dispatch", adopted.STANDBY / "dispatch.py")
        bridge = exact(stage / "guarded-standby-bridge.py", BRIDGE_SHA)
        need(len(bridge) == manifest["files"]["guarded-standby-bridge.py"]["bytes"], "wrong guarded bridge extent")
        return pending, dispatch, bridge

    adopted.load_sources = load_sources
    adopted.verify_publication = approved_publication
    return adopted


def launch_adapter(stage, receipts):
    launch = source("launch.py")
    launch = replace(launch, "import prepare\n", "")
    launch = replace(launch, PREVIOUS_STAGE, CURRENT_STAGE)
    launch = replace(launch, PREVIOUS_PENDING_SHA, PENDING_SHA)
    adopted = module(PRIVATE / "launch.py", launch, prepare=prepare_adapter(stage))
    adopted.HERE = receipts
    return adopted


def observer_source():
    raw = replace(source("observe_refill.py"), PREVIOUS_STAGE, CURRENT_STAGE)
    return replace(raw, "o01-pending-decision.json", "o01-memory48-pending-decision.json")


def transport():
    # Call the published transport's pinned environment without outputting any authentication setting.
    sys.path.insert(0, str(MEMORY))
    return module(MEMORY / "transport.py", exact(MEMORY / "transport.py", TRANSPORT_SHA)).adapter()


def verify_refill(args):
    approved_publication(None, args.publication)
    pending, _, _ = prepare_adapter(args.stage).load_sources()
    need(pending.refill_request(args.request), "invalid granted refill identity")
    source = observer_source()
    chosen = (args.request, args.marker_sha, args.grant_sha, STAGE_SHA)
    body = (f"source={source!r}\nexec(compile(source,'refill-observer.py','exec'))\n"
            f"print(json.dumps(observe_and_release(*{chosen!r},source),sort_keys=True))\n").encode()
    compile(body, "refill-observer-entry.py", "exec")
    encoded = b64encode(zlib.compress(body, 9)).decode("ascii")
    remote = "from base64 import b64decode;from zlib import decompress;exec(decompress(b64decode(" + repr(encoded) + ")))"
    adapter = transport()
    argv = [*adapter.ssh_args(), shlex.join([adapter.PYTHON, "-I", "-S", "-B", "-u", "-c", remote])]
    receipt = args.receipts / ("observe-release-" + args.request)
    receipt.mkdir(mode=0o700)
    with (receipt / "intent.json").open("x", encoding="utf8") as output:
        json.dump({"request": args.request, "observer_source_SHA256": sha256(source).hexdigest(),
                   "grant_SHA256": args.grant_sha, "stage_SHA256": STAGE_SHA, "replay_allowed": False}, output)
    with (receipt / "stdout.private").open("xb") as out, (receipt / "stderr.private").open("xb") as err:
        try:
            rc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=90, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = None
    result = {"request": args.request, "remote_outcome": "UNKNOWN", "rc": rc}
    if rc == 0:
        raw = (receipt / "stdout.private").read_bytes()
        need(0 < len(raw) <= 16384, "observer output exceeded bound")
        value = json.loads(raw)
        need(value["request"] == args.request and value["actual_computing_cards"] == 8,
             "unexpected refill terminal data")
        result = value
    print(json.dumps(result, sort_keys=True))
    return int(rc != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "consume"))
    for opt in ("request", "publication", "marker-sha", "grant-sha"):
        parser.add_argument("--" + opt, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    need(args.receipts.resolve(strict=True).is_relative_to(REPO / "local-only")
         and all(re.fullmatch("[0-9a-f]{64}", digest) for digest in (args.marker_sha, args.grant_sha)),
         "requires private receipts and exact grant hashes")
    if args.action == "consume":
        raise SystemExit(verify_refill(args))
    args.stage_sha = STAGE_SHA
    adapter = launch_adapter(args.stage, args.receipts)
    result = adapter.run(args, transport().ssh_args())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(int(result["stage"] != "BRIDGE_LAUNCH_PENDING"))
