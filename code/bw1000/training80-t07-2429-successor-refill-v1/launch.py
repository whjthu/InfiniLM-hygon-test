"""Use the existing production bridge for one guarded T07 renewal."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO / "local-only"
NODE = "f13r2n07"
REQUEST = "8dcecfc1023c4bb5bc05f4bccb850b1f"
ADMISSION = "42f59a4f4f004d6db5ac2c4560afc034"
RUNNER = REPO.parent / "run-dcu-standby-cycle-20261123.ps1"
PINS = {
    "../run-dcu-standby-cycle-20261123.ps1": "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73",
    "code/bw1000/fleet-dcu-standby-v2/dispatch.py": "1d849fe990294c77b11b812992ab752ef23f3a47c612f2b9c2818302f64a92c7",
    "code/bw1000/fleet-dcu-standby-v2/controller.py": "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba",
    "code/bw1000/fleet-dcu-standby-v2/worker.py": "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059",
    "code/bw1000/fleet-dcu-standby-v2/CLAIM.json": "1f441531cc4c9fdbf3226fb887242d693475f4e4eca603cf0ce69ccc7f7299cd",
    "code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py": "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360",
    "code/bw1000/fleet-dcu-standby-compute-bridge-v1/CLAIM.json": "631552512b1fb0efe08626d5c1d3a40d9d8c77b801bc1dbf36ebc27483d04feb",
}
FROZEN_PREDECESSOR = "7af90d018cfddbe8fa702df8b73a77cbaabe32040cbe5748337a50051b901dc7"
PRIOR_EXPECTED_SHA = "9636588dc03fdb5aa3d82eb71fc57260c3cf2b379e9bc26063b022d6a114a103"
PRIOR_WORKER_SHA = "0b4cf998e02f811481ca9ff6bdac167e67f1b73ff5f778c808fc74cfa6c95011"
AUDIT = PRIVATE / "run-training80-T05-2417-audited-refill-f67445c1a3204a708cd68006b03cbd9d.py"
AUDIT_SHA = "7d9ce780222e6b87ab54b3b08c3cdacd95b99fe44d384fc2cda6dbce7eaf4766"
GATE = r"""
from hashlib import sha256
import json,os,pathlib,shlex,subprocess,sys
request,tag,command,pin=sys.argv[1:]
base=pathlib.Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
def need(ok,reason):
    if not ok:raise ValueError(reason)
try:
    raw=subprocess.check_output(["sacct","-nP","-j","1103606.2429,1103606.2428",
       "-o","JobIDRaw,JobName%100,State,ExitCode,NodeList,AllocTRES,End"],text=True,timeout=6)
    wanted={"1103606.2429":"training80-dcu-standby-5f03331c61d9-f13r2n07",
            "1103606.2428":"training80-standby-compute-5f03331c61d9"}
    hosts={"1103606.2429":"f13r2n07","1103606.2428":"f12r2n13"}
    confirmed={r[0] for l in raw.splitlines() if len(r:=l.split("|"))==7 and r[0] in wanted
                and r[1:5]==[wanted[r[0]],"COMPLETED","0:0",hosts[r[0]]] and r[6]!="Unknown"}
    need(confirmed==set(wanted),"original standby and zero-GRES observation are not terminal")
    need(not any(os.path.lexists(base/(p+request)) for p in
       ("training80-dcu-standby-","training80-standby-compute-",".training80-t07-successor-admission-")),"original successor already exists")
    print(json.dumps(dict(event="certified-host-ready",request=request,old_step=tag,payload_SHA256=pin)),flush=True)
    packet=sys.stdin.buffer.readline(262146)
    need(1<len(packet)<=262145 and packet.endswith(b"\n") and sha256(packet[:-1]).hexdigest()==pin,"payload differs")
    print(json.dumps(dict(event="HOST_RECEIVED",request=request,payload_SHA256=pin)),flush=True)
    bridge=subprocess.run(shlex.split(command),input=packet[:-1],capture_output=True,timeout=35)
    sys.stdout.buffer.write(bridge.stdout);sys.stdout.flush()
    if bridge.returncode:
        print(json.dumps(dict(event="bridge-nonzero",request=request,bridge_rc=bridge.returncode,
                              bridge_stderr_SHA256=sha256(bridge.stderr).hexdigest())),flush=True)
    raise SystemExit(bridge.returncode)
except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as e:
    print(json.dumps(dict(event="host-gate-error",request=request,error=type(e).__name__,
                          detail_SHA256=sha256(str(e).encode()).hexdigest())),flush=True)
    raise SystemExit(1)
"""


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
                   "\n\n_t07_guard = {'__name__': 't07_successor_guard'}\n"
                   f"exec(compile({guardian!r}, 't07-successor-guard.py', 'exec'), _t07_guard)\n\nRESULTS = Path(")
    source = patch(source, '    envelope = json.loads(message)\n    request = envelope["request"]',
                   '    envelope = json.loads(message)\n'
                   '    return _t07_guard["launch"](RESULTS, envelope, b64decode(envelope["entry"], validate=True),\n'
                   '                                lambda: _start_bound(envelope, launch_only))\n\n'
                   'def _start_bound(envelope, launch_only):\n    request = envelope["request"]')
    source = patch(source, '    try:\n        namespace["run"](inputs)\n',
                   '    _t07_guard["install"](RESULTS, namespace, inputs, entry)\n'
                   '    try:\n        namespace["run"](inputs)\n')
    compile(source, "<scoped bridge>", "exec")
    return source.encode()


def build(publication):
    for path, pin in PINS.items():
        need(digest((REPO / path).read_bytes()) == pin, "existing production source differs")
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    guard = (HERE / "guard.py").read_bytes()
    need(claim["request"] == REQUEST and claim["fresh_admission_request"] == ADMISSION
         and claim["original_live_observation_SHA256"] == FROZEN_PREDECESSOR
         and claim["files"]["guard.py"] == digest(guard), "successor claim differs")
    need(digest(AUDIT.read_bytes()) == AUDIT_SHA, "existing authenticated bridge transport differs")
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
         f"    t07_device=dict(receipt['nodes'][{NODE!r}],node={NODE!r},observed_epoch=receipt['observed_epoch'],\n"
         "                    request_id=receipt['request_id'],payload_SHA256=receipt['local_receipt']['payload_SHA256'])\n"
         "    transport_body=json.dumps(dict(request=request,t07_successor_device=t07_device,entry=base64.b64encode(body).decode(),"),
        ("q=subprocess.run(inner+[command],input=transport_body,capture_output=True,timeout=120)",
         "q=audited_bridge(inner,command,transport_body,cutoff)"),
    )
    for before, after in changes:
        source = patch(source, before, after)
    return compile(source, "<bounded T07 successor runner>", "exec"), digest(patched_bridge)


def predecessor(path, pin, *, ready):
    path=path.resolve(strict=True)
    need(path.parent==PRIVATE and path.name.startswith("training80-f13r2n07-original2429-predecessor-current-"),
         "exact T07 scoped observer required")
    raw=path.read_bytes()
    need(0<len(raw)<=16384 and digest(raw)==pin,"original .2429/.2428 observation differs")
    src=json.loads(raw)
    r=src["observed"]
    standby,=r["accounting"]["1103606.2429"]
    sidecar,=r["accounting"]["1103606.2428"]
    need(src["rc"]==0 and r["parent_state"]=="RUNNING" and r["owner"] and r["alloc_320"]
         and r["expected"]["SHA256"]==PRIOR_EXPECTED_SHA and r["worker"]["SHA256"]==PRIOR_WORKER_SHA
         and standby[1]=="training80-dcu-standby-5f03331c61d9-f13r2n07"
         and standby[5]==NODE and "gres/dcu=8" in standby[6].split(",")
         and sidecar[1]=="training80-standby-compute-5f03331c61d9" and sidecar[5]=="f12r2n13"
         and not any(key.startswith("gres/") for key in sidecar[6].split(",")),
         "original T07 scope or bridge identity differs")
    if ready:
        term=r["terminal"]["document"]
        need(standby[2:4]==["COMPLETED","0:0"] and standby[8]!="Unknown"
             and sidecar[2:4]==["COMPLETED","0:0"] and sidecar[8]!="Unknown"
             and term["state"]=="WORKER_RETURNED" and term["step"]=="1103606.2429"
             and term["epoch"]>=1795513787.0696082 and r["live"]=={} and r["squeue"]=={}
             and 0<=time.time()-r["epoch"]<=150,"original worker or physical occupancy still unresolved")
    else:
        need(standby[2]=="RUNNING" and set(r["live"])=={"1103606.2429"}
             and set(r["squeue"]).issubset({"1103606.2429"}),"preparation scope differs")
    return standby

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publication")
    parser.add_argument("--terminal",type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.check:
        path=PRIVATE/"training80-f13r2n07-original2429-predecessor-current-17d90aba0e924be9b79c47afb6e6f7c8.json"
        predecessor(path,FROZEN_PREDECESSOR,ready=False)
        compile(GATE,"<authenticated exact T07 authority>","exec")
        _, bridge_sha = build("0" * 40)
        print(json.dumps(dict(offline_check=True, bridge_SHA256=bridge_sha, request=REQUEST, remote_calls=0)))
        return
    need(args.publication is not None and re.fullmatch(r"[a-f0-9]{40}", args.publication)
         and args.publication != "0" * 40, "root verified publication is required")
    need(args.terminal is not None and args.sha and re.fullmatch("[a-f0-9]{64}",args.sha),
         "original T07 terminal receipt and SHA required")
    prior=predecessor(args.terminal,args.sha,ready=True)
    program, bridge_sha = build(args.publication)
    local=runpy.run_path(str(AUDIT))
    audited=local["audited_bridge"]
    audited.__globals__.update(REFILL=REQUEST,STEP="1103606.2429",GATE=GATE)
    intent = PRIVATE / f"training80-T07-2429-successor-{REQUEST}-single-call.json"
    with intent.open("x", encoding="utf8") as out:
        json.dump(dict(request=REQUEST, admission=ADMISSION, publication=args.publication, bridge_SHA256=bridge_sha,
                       nodes=[NODE], cards=8, prior=prior, terminal_SHA256=args.sha, replay_allowed=False, epoch=time.time()), out, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.environ["HYGON_STANDBY_ONLY_NODES"] = NODE
    os.environ["HYGON_STANDBY_VIA_COMPUTE"] = "1"
    os.chdir(REPO.parent)
    exec(program, {"__name__": "__main__", "audited_bridge": audited})


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(request=REQUEST, failure=type(error).__name__,
                              failure_SHA256=digest(str(error).encode("utf8", "replace")))))
        sys.exit(1)
