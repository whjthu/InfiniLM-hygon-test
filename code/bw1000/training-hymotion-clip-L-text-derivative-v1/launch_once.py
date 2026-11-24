"""Preflight and stage one SHA-bound CLIP text-only derivative through the existing route."""

import argparse
from base64 import b64encode
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from threading import Timer
import time
from urllib.request import Request, build_opener
import zlib

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(REPO / "tools"))
import receive_once as receiver

PRIVATE = REPO / "local-only"
HEADER = PRIVATE / "training-hymotion-clip-official-safetensors-header-only"
TRANSPORT = REPO / "code/bw1000/training-gemma-vizwiz-five-data-stage-v1/launch_once.py"
TRANSPORT_SHA = "98c9f1eea7cde26c575a13471accc0cf99144cad6b0267aa42de8734fe183389"
TRACKED = ("launch_once.py", "receive_once.py", "claim.json")
MAX_REPLY = 16384


def command(claim_bytes, claim, mode, observed=None):
    programs = {}
    for name, pin in claim["public_modules"].items():
        path = REPO / pin["path"]
        raw = path.read_bytes()
        if receiver.sig(raw) != {k: pin[k] for k in ("bytes", "SHA256")}:
            raise ValueError("published derivative dependency differs")
        programs[name] = b64encode(zlib.compress(raw, 9)).decode()
    code = (PACKAGE / "receive_once.py").read_bytes()
    packed = b64encode(zlib.compress(code, 9)).decode()
    wrapper = f"""
import base64,sys,types,zlib
for name,body in {programs!r}.items():
    mod=types.ModuleType(name)
    sys.modules[name]=mod
    exec(compile(zlib.decompress(base64.b64decode(body,validate=True)),name+'.py','exec'),vars(mod))
namespace={{"__name__":"__main__","MODE":{mode!r},"CLAIM":{claim!r},
           "CLAIM_SHA256":{sha256(claim_bytes).hexdigest()!r},"PREFLIGHT":{observed!r}}}
exec(compile(zlib.decompress(base64.b64decode({packed!r},validate=True)),"<clip_text_derivative>",'exec'),namespace)
"""
    compile(wrapper, "<bounded target bootstrap>", "exec")
    remote = "/usr/local/python3.12/bin/python3 -I -S -B -c " + shlex.quote(wrapper)
    if len(remote) > 24_000:
        raise ValueError("trusted target command exceeds the bounded SSH argument")
    return remote


def exact_publication(revision, claim):
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("exact published Git revision required")
    git = ["git", "-c", f"safe.directory={REPO.as_posix()}"]
    subprocess.run([*git,"merge-base","--is-ancestor",revision,"refs/remotes/origin/bw1000-experiments"],
                   cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=8)
    paths = [PACKAGE / name for name in TRACKED] + [REPO / p["path"] for p in claim["public_modules"].values()]
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        blob = subprocess.check_output([*git,"show",revision+":"+relative], cwd=REPO,
                                       stderr=subprocess.PIPE, timeout=8)
        if blob != path.read_bytes():
            raise ValueError("local derivative source differs from ordinary Git publication")


def local_transport():
    body = TRANSPORT.read_bytes()
    if sha256(body).hexdigest() != TRANSPORT_SHA:
        raise ValueError("existing project SSH entry differs")
    helper = {"__name__":"existing_clip_transport","__file__":str(TRANSPORT)}
    exec(compile(body,str(TRANSPORT),"exec"),helper)
    return helper["ssh_command"]()


def same_header(claim):
    with HEADER.open("rb") as stream:
        data = stream.read(receiver.HEADER_BYTES + 1)
    if receiver.sig(data) != {"bytes":receiver.HEADER_BYTES,"SHA256":claim["official_header_SHA256"]}:
        raise ValueError("cached official header differs")
    return data


def preflight(argv, claim, claim_raw, revision, output):
    output.mkdir()
    request = claim["request"]
    item = dict(request=request, publication=revision, state="UNKNOWN", writes=0, gpu=0, slurm=0)
    receiver.save(output/"preflight-intent.json", receiver.json_bytes(item))
    try:
        result = subprocess.run([*argv,command(claim_raw,claim,"preflight")], input=b"", capture_output=True, timeout=73)
        item.update(rc=result.returncode, stdout=receiver.sig(result.stdout), stderr=receiver.sig(result.stderr))
        if result.returncode == 0 and 0 < len(result.stdout) <= MAX_REPLY:
            event = json.loads(result.stdout)
            if (event["event"] == "PREFLIGHT_COMPLETE" and event["request"] == request
                    and event["metadata"] == claim["metadata"]
                    and event["verified_prefix_intervals"] == claim["source_ranges"][:2]
                    and event["retained_file_bytes"] == claim["retained_end"]
                    and event["writes"] == event["slurm"] == event["gpu"] == 0):
                item.update(state="ORIGINAL_RETAINED_PREFIX_VERIFIED", observed=event)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        item["failure_type"] = type(error).__name__
    finally:
        receiver.save(output/"preflight.json", receiver.json_bytes(item))
    return item


def response_event(pipe, expected_request):
    data = pipe.readline(4097)
    if not data or len(data) > 4096 or not data.endswith(b"\n"):
        raise EOFError("missing or oversized original receiver message")
    event = json.loads(data)
    if event.get("request") != expected_request:
        raise ValueError("receiver message is outside the frozen scope")
    return event


def transfer_tail(claim, local_sink, deadline):
    source = REPO / claim["public_modules"]["probe_hymotion_official_clip_range"]["path"]
    raw = source.read_bytes()
    pin = claim["public_modules"]["probe_hymotion_official_clip_range"]
    if receiver.sig(raw) != {k:pin[k] for k in ("bytes","SHA256")}:
        raise ValueError("official HTTP provider source differs")
    spec = importlib.util.spec_from_file_location("official_clip_source", source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), vars(module))
    opener = build_opener(module.OfficialRedirects())
    part = claim["source_ranges"][-1]
    start, stop = part["interval"]
    req = Request(module.BASE+"model.safetensors",headers={
        "Range":f"bytes={start}-{stop-1}", "Accept-Encoding":"identity"})
    remaining = part["bytes"]
    digest = sha256()
    with opener.open(req,timeout=11) as result:
        if (result.status != 206 or result.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{claim['official_bytes']}"
                or result.headers.get("Content-Length") not in (None,str(remaining))
                or result.headers.get("Content-Encoding","identity") != "identity"):
            raise ValueError("official returned wrong text suffix interval")
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline while forwarding official text suffix")
            chunk = result.read(min(256*1024,remaining))
            if not chunk or local_sink.write(chunk) != len(chunk):
                raise EOFError("official or SSH tail stream truncated")
            remaining -= len(chunk)
            digest.update(chunk)
        if result.read(1) or digest.hexdigest() != part["SHA256"]:
            raise ValueError("official text suffix length or digest differs")


def send(argv,claim,claim_bytes,revision,output,preflight_raw,header):
    inspected = json.loads(preflight_raw)
    if (inspected["state"] != "ORIGINAL_RETAINED_PREFIX_VERIFIED" or inspected["rc"] != 0
            or inspected["request"] != claim["request"] or inspected["publication"] != revision):
        raise ValueError("same-scope independent target preflight is required")
    request = claim["request"]
    item = dict(request=request, state="UNKNOWN", publication=revision, preflight=receiver.sig(preflight_raw))
    receiver.save(output/"install-intent.json",receiver.json_bytes(item))
    proc = None
    timer = None
    try:
        deadline = time.monotonic() + receiver.BUDGET
        proc = subprocess.Popen([*argv,command(claim_bytes,claim,"install",inspected["observed"])],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        timer = Timer(receiver.BUDGET,proc.kill)
        timer.daemon = True
        timer.start()
        first = response_event(proc.stdout,request)
        if first != dict(event="READY", request=request, header_bytes=receiver.HEADER_BYTES,
                         tail_bytes=claim["source_ranges"][-1]["bytes"]):
            raise ValueError("receiver did not reserve the expected derivative scope")
        item["ready_seen"] = True
        if proc.stdin.write(header)!=len(header):
            raise OSError("header stream write truncated")
        transfer_tail(claim,proc.stdin,deadline)
        item["sender_tail_verified"] = True
        proc.stdin.close()
        final = response_event(proc.stdout,request)
        item["rc"] = proc.wait(timeout=max(0.1,deadline-time.monotonic()))
        if (item["rc"] == 0 and final["event"] == "COMPLETE"
                and re.fullmatch("[0-9a-f]{64}",str(final.get("receipt_SHA256")))
                and 492242536 < final["derivative_bytes"] < 492242536 + 128*1024):
            item.update(state="ORIGINAL_DERIVATIVE_REPORTED", receipt=final)
        elif final["event"] == "FAILED_RETAINED":
            item.update(state="ORIGINAL_DERIVATIVE_FAILED_RETAINED", receipt=final)
    except (OSError, ValueError, EOFError, KeyError, subprocess.SubprocessError) as error:
        item["failure_type"] = type(error).__name__
    finally:
        if timer:
            timer.cancel()
        if proc:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            for pipe in (proc.stdin,proc.stdout):
                try:
                    pipe.close()
                except OSError:
                    pass
        receiver.save(output/"install-result.json",receiver.json_bytes(item))
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--check-local",action="store_true")
    choices.add_argument("--preflight")
    choices.add_argument("--publication")
    args = parser.parse_args()
    claim_raw = (PACKAGE/"claim.json").read_bytes()
    claim = json.loads(claim_raw)
    receiver.check_claim(claim)
    header = same_header(claim)
    check = command(claim_raw,claim,"preflight")
    if args.check_local:
        print(json.dumps(dict(request=claim["request"], claim=receiver.sig(claim_raw),
                              header=receiver.sig(header), command_bytes=len(check.encode()), dispatched=False)))
        return 0
    revision = args.preflight or args.publication
    exact_publication(revision,claim)
    argv = local_transport()
    output = PRIVATE/("hymotion-clip-text-derivative-"+claim["request"])
    if args.preflight:
        result = preflight(argv,claim,claim_raw,revision,output)
        ok = result["state"] == "ORIGINAL_RETAINED_PREFIX_VERIFIED"
    else:
        result = send(argv,claim,claim_raw,revision,output,(output/"preflight.json").read_bytes(),header)
        ok = result["state"] == "ORIGINAL_DERIVATIVE_REPORTED"
    # Device and inode identity is private and used for the matched install only.
    brief = {k:v for k,v in result.items() if k != "observed"}
    print(json.dumps(brief,sort_keys=True))
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
