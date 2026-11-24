"""Use the existing production bridge for one independently granted O01 renewal."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO / "local-only"
NODE = "f12r2n11"
REQUEST = "3f96a7be936a4a018ff9f83623895428"
ADMISSION = "74502903da9140449d1c764a840a117a"
RUNNER = REPO.parent / "run-dcu-standby-cycle-20261123.ps1"
PINS = {
    "../run-dcu-standby-cycle-20261123.ps1": "5705254ac745cfd36656cba60495425ffbd15f9629d478e1d3559418565da297",
    "code/bw1000/fleet-dcu-standby-v2/dispatch.py": "1d849fe990294c77b11b812992ab752ef23f3a47c612f2b9c2818302f64a92c7",
    "code/bw1000/fleet-dcu-standby-v2/controller.py": "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba",
    "code/bw1000/fleet-dcu-standby-v2/worker.py": "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059",
    "code/bw1000/fleet-dcu-standby-v2/CLAIM.json": "1f441531cc4c9fdbf3226fb887242d693475f4e4eca603cf0ce69ccc7f7299cd",
    "code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py": "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360",
    "code/bw1000/fleet-dcu-standby-compute-bridge-v1/CLAIM.json": "631552512b1fb0efe08626d5c1d3a40d9d8c77b801bc1dbf36ebc27483d04feb",
}
FROZEN_PREDECESSOR = "a6ef860d17203f5554f925c3a2f56275e64c8fdb6b17d188ba6a3597471eea22"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def patch(raw, before, after):
    need(raw.count(before) == 1, "qualified source boundary differs")
    return raw.replace(before, after)


def bridge(source, guardian):
    source = patch(source, "\n\nRESULTS = Path(",
                   "\n\n_o01_guard = {'__name__': 'o01_successor_guard'}\n"
                   f"exec(compile({guardian!r}, 'o01-successor-guard.py', 'exec'), _o01_guard)\n\nRESULTS = Path(")
    source = patch(source, '    envelope = json.loads(message)\n    request = envelope["request"]',
                   '    envelope = json.loads(message)\n'
                   '    return _o01_guard["launch"](RESULTS, envelope, b64decode(envelope["entry"], validate=True),\n'
                   '                                lambda: _start_bound(envelope, launch_only))\n\n'
                   'def _start_bound(envelope, launch_only):\n    request = envelope["request"]')
    source = patch(source, '    try:\n        namespace["run"](inputs)\n',
                   '    _o01_guard["install"](RESULTS, namespace, inputs, entry)\n'
                   '    try:\n        namespace["run"](inputs)\n')
    compile(source, "<scoped bridge>", "exec")
    return source.encode()


def build(publication):
    for path, pin in PINS.items():
        need(digest((REPO / path).read_bytes()) == pin, "existing production source differs")
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    guard = (HERE / "guard.py").read_bytes()
    need(claim["request"] == REQUEST and claim["fresh_admission_request"] == ADMISSION
         and claim["old_2296_and_2299_SHA256"] == FROZEN_PREDECESSOR
         and claim["files"]["guard.py"] == digest(guard), "successor claim differs")
    raw = RUNNER.read_bytes()
    source = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n@'\n", 1)[1].split("\n'@ | & ", 1)[0]
    frozen = (REPO / "code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py").read_text(encoding="utf8")
    patched_bridge = bridge(frozen, guard)
    changes = (
        ("req=uuid4().hex", f"req={ADMISSION!r}"),
        ("    request=uuid4().hex", f"    request={REQUEST!r}"),
        ('publication=current_git("rev-parse","HEAD")', f"publication={publication!r}"),
        ('    mod.published(repo,publication,[(pkg/name).relative_to(repo).as_posix() for name in ("CLAIM.json","worker.py","controller.py","dispatch.py","status.py")])\n', ""),
        ('        mod.published(repo,publication,[(bridge_pkg/name).relative_to(repo).as_posix()\n                        for name in ("bridge.py","CLAIM.json")])\n', ""),
        ("    body,intent,cutoff=mod.build(receipt,publication,request,10800)",
         "    body,intent,cutoff=mod.build(receipt,publication,request,10800)\n"
         f"    if intent['nodes'] != [{NODE!r}] or intent['requested_cards'] != 8:\n"
         "        raise ValueError('exact successor node and card count differ')"),
        ("    transport_body=json.dumps(dict(request=request,entry=base64.b64encode(body).decode(),",
         f"    bridge_source={patched_bridge!r}\n"
         "    bridge_claim={'bridge_SHA256':sha256(bridge_source).hexdigest()}\n"
         f"    o01_device=dict(receipt['nodes'][{NODE!r}],node={NODE!r},observed_epoch=receipt['observed_epoch'],\n"
         "                    request_id=receipt['request_id'],payload_SHA256=receipt['local_receipt']['payload_SHA256'])\n"
         "    transport_body=json.dumps(dict(request=request,o01_successor_device=o01_device,entry=base64.b64encode(body).decode(),"),
    )
    for before, after in changes:
        source = patch(source, before, after)
    return compile(source, "<qualified O01 successor runner>", "exec"), digest(patched_bridge)


def predecessor(path):
    raw = path.read_bytes()
    need(digest(raw) == FROZEN_PREDECESSOR, "old 2296 and 2299 observer differs")
    origin = json.loads(raw)
    rows = {r[0]: r for r in origin["observed"]["accounting"]}
    need(origin["rc"] == 0 and origin["authenticated"] and origin["sent"]
         and rows["1103606.2296"][2:4] == rows["1103606.2299"][2:4] == ["COMPLETED", "0:0"],
         "old steps are not independently terminal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publication")
    args = parser.parse_args()
    path = PRIVATE / "O01-original-8e398fe1-current-051708165534.json"
    predecessor(path)
    if args.check:
        _, bridge_sha = build("0" * 40)
        print(json.dumps(dict(offline_check=True, bridge_SHA256=bridge_sha, request=REQUEST, remote_calls=0)))
        return
    need(args.publication is not None and re.fullmatch(r"[a-f0-9]{40}", args.publication)
         and args.publication != "0" * 40, "root verified publication is required")
    program, bridge_sha = build(args.publication)
    intent = PRIVATE / f"training80-O01-2296-successor-{REQUEST}-single-call.json"
    with intent.open("x", encoding="utf8") as out:
        json.dump(dict(request=REQUEST, admission=ADMISSION, publication=args.publication, bridge_SHA256=bridge_sha,
                       nodes=[NODE], cards=8, replay_allowed=False, epoch=time.time()), out, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.environ["HYGON_STANDBY_ONLY_NODES"] = NODE
    os.environ["HYGON_STANDBY_VIA_COMPUTE"] = "1"
    os.chdir(REPO.parent)
    exec(program, {"__name__": "__main__"})


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print(json.dumps(dict(request=REQUEST, failure=type(error).__name__,
                              failure_SHA256=digest(str(error).encode("utf8", "replace")))))
        sys.exit(1)
